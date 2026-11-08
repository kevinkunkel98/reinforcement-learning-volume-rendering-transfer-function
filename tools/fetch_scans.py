"""Download the scans the committed manifest lists, without the 3 GB zip.

`data/totalseg_manifest.json` (in the repository) names every subject, its
split and the sha256 of its CT. This reads only those subjects out of the
TotalSegmentator zip on Zenodo with HTTP range requests and writes
data/totalseg/<id>/ct.nii.gz and labels.nii.gz exactly as
`tools.select_totalseg` does from a local copy of the zip.

    python -m tools.fetch_scans --demo --splits      # the viewer's 7 showcase patients (make scans)
    python -m tools.fetch_scans                      # the 6 test patients (~250 MB; the tests use them)
    python -m tools.fetch_scans --names ts_s0358     # plus single scans by name
    python -m tools.fetch_scans --splits test val train
    python -m tools.fetch_scans --splits all         # all 100 (~3 GB of reads), for retraining

Scans already on disk with the manifest's checksum are skipped.
"""
import argparse
import http.client
import io
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import scans
from tools.select_totalseg import DEFAULT_OUT_DIR, ZENODO_RECORD, _sha256_file, extract_subject

ZIP_NAME = "Totalsegmentator_dataset_small_v201.zip"
ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ZIP_NAME}?download=1"
USER_AGENT = "tf-rl-fetch-scans/1.0"
ALL_SPLITS = ("train", "val", "test", "train_extra")


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, with a small block
    cache -- enough for `zipfile` to read the central directory and single
    members of a remote zip."""

    def __init__(self, url: str, block_size: int = 2 << 20, cached_blocks: int = 16, retries: int = 6,
                 size: int = None, blocks: dict = None):
        self.url, self.block_size, self.cached_blocks, self.retries = url, block_size, cached_blocks, retries
        self._blocks = OrderedDict(blocks or {})
        self._pos = 0
        self.bytes_fetched = 0
        self.size = size if size is not None else self._probe_size()

    def clone(self) -> "HttpRangeFile":
        """A second handle on the same URL that starts with this one's cached
        blocks, so another thread can open the zip without reading its central
        directory again."""
        return HttpRangeFile(self.url, self.block_size, self.cached_blocks, self.retries,
                             self.size, dict(self._blocks))

    def _get(self, start: int, end: int) -> tuple:
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                                            "User-Agent": USER_AGENT})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise OSError(f"{self.url} ignored the range request (HTTP {response.status})")
                    data = response.read()
                    if len(data) != end - start + 1:
                        raise http.client.IncompleteRead(data, end - start + 1 - len(data))
                    return data, response.headers.get("Content-Range", "")
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as exc:
                if attempt == self.retries - 1:
                    raise OSError(f"download failed after {self.retries} tries: {exc}") from exc
                time.sleep(2 ** attempt)

    def _probe_size(self) -> int:
        _, content_range = self._get(0, 0)  # "bytes 0-0/<size>"
        return int(content_range.rsplit("/", 1)[1])

    def _block(self, index: int) -> bytes:
        if index in self._blocks:
            self._blocks.move_to_end(index)
            return self._blocks[index]
        start = index * self.block_size
        data, _ = self._get(start, min(start + self.block_size, self.size) - 1)
        self.bytes_fetched += len(data)
        self._blocks[index] = data
        if len(self._blocks) > self.cached_blocks:
            self._blocks.popitem(last=False)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer):
        view = memoryview(buffer).cast("B")
        written = 0
        while written < len(view) and self._pos < self.size:
            index, offset = divmod(self._pos, self.block_size)
            chunk = self._block(index)[offset:offset + len(view) - written]
            view[written:written + len(chunk)] = chunk
            written += len(chunk)
            self._pos += len(chunk)
        return written


def wanted_subjects(splits, names=()) -> list:
    subjects = scans.manifest().get("subjects", [])
    return [s for s in subjects if s["split"] in splits or s["name"] in names]


def is_present(entry: dict) -> bool:
    return (os.path.exists(entry["path"]) and os.path.exists(entry["labels_path"])
            and _sha256_file(entry["path"]) == entry["sha256"])


def fetch(splits, out_dir: str = DEFAULT_OUT_DIR, url: str = ZIP_URL, workers: int = 4, names=()) -> list:
    entries = wanted_subjects(splits, names)
    if not entries:
        raise SystemExit(f"no subjects of split(s) {', '.join(splits)} in {scans.MANIFEST_PATH}")
    missing = [e for e in entries if not is_present(e)]
    print(f"[fetch_scans] {len(entries) - len(missing)}/{len(entries)} already present")
    if not missing:
        return []
    remote = HttpRangeFile(url)
    zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)).close()  # caches the central directory
    workers = max(1, min(workers, len(missing)))
    print(f"[fetch_scans] reading {len(missing)} subject(s) from {url}, {workers} at a time")
    local, lock, handles, fetched = threading.local(), threading.Lock(), [remote], []

    def extract(entry):
        if not hasattr(local, "zf"):
            handle = remote.clone()
            with lock:
                handles.append(handle)
            local.zf = zipfile.ZipFile(io.BufferedReader(handle, buffer_size=1 << 20))
        started = time.time()
        info = {k: entry[k] for k in ("study_type", "region", "shape", "spacing")}
        extract_subject(local.zf, entry["id"], info, entry["split"], out_dir)
        if _sha256_file(entry["path"]) != entry["sha256"]:
            raise SystemExit(f"{entry['id']}: checksum does not match the manifest")
        return entry, time.time() - started

    with ThreadPoolExecutor(workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(extract, e) for e in missing]), 1):
            entry, seconds = future.result()
            fetched.append(entry["name"])
            total = sum(h.bytes_fetched for h in handles) / 1e6
            print(f"[fetch_scans] {n}/{len(missing)} {entry['name']} ({entry['split']}) "
                  f"{seconds:.0f}s, {total:.0f} MB so far")
    return fetched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="*", default=["test"], choices=list(ALL_SPLITS) + ["all"])
    parser.add_argument("--names", nargs="*", default=[], help="single scans on top, e.g. ts_s0358")
    parser.add_argument("--demo", action="store_true", help="the viewer's showcase set, data/demo_scans.json")
    parser.add_argument("--candidates", action="store_true", help="the 24 patients the showcase set was picked from")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=4, help="scans downloaded in parallel")
    args = parser.parse_args()
    splits = ALL_SPLITS if "all" in args.splits else tuple(args.splits)
    names = (tuple(args.names) + (tuple(scans.demo_names()) if args.demo else ())
             + (tuple(scans.demo_names("candidates")) if args.candidates else ()))
    fetch(splits, args.out_dir, workers=args.workers, names=names)
    print(f"[fetch_scans] done: {len(scans.available_names())} scans available")


if __name__ == "__main__":
    main()
