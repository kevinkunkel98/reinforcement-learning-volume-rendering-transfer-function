"""Add larger held-out splits from the full TotalSegmentator dataset.

The small Zenodo subset's 100 patients are all spoken for (90 train, 4 val,
6 test), and six test patients make "works on unseen patients" a claim about
six people. This picks new patients from the full dataset (1228 CTs, Zenodo
record 10047292) that are not in the manifest yet, passes them through the
same extent filter, and adds them as two new splits, stratified by region
like the originals:

- `val_extra`  (VAL_PER_REGION per region): for the decisions still open on
  val (occluder weight, contrast by HU), so test stays untouched.
- `test_extra` (TEST_PER_REGION per region): evaluated once, with the frozen
  checkpoint.

The original splits are not touched, so every reported number stays as it
is and the shipped checkpoint stays valid (none of these patients was seen
in training). Reads only the needed members of the 24 GB zip with HTTP range
requests (~15 MB of metadata, a CT header per candidate, then ~20 MB per
selected patient). Each new entry records `source: "full"`, which
`tools.fetch_scans` uses to pick the zip, and the dataset's institute,
manufacturer, scanner and pathology for stratified analysis.

    python -m tools.extend_eval_splits --dry-run     # candidates and picks, no download
    python -m tools.extend_eval_splits               # extract and update the manifest
"""
import argparse
import io
import json
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tools.fetch_scans import FULL_ZIP_URL, HttpRangeFile
from tools.select_totalseg import (DEFAULT_MANIFEST, DEFAULT_OUT_DIR, extract_subject, passes_extent_filter,
                                   read_ct_header, read_meta, region_for_study_type)

FULL_ZENODO_RECORD = "10047292"
SOURCE = "full"
VAL_SPLIT, TEST_SPLIT = "val_extra", "test_extra"
VAL_PER_REGION, TEST_PER_REGION = 8, 15
REGIONS = ("whole_body", "trunk", "thorax", "abdomen_pelvis")
META_FIELDS = ("institute", "manufacturer", "scanner_model", "pathology", "age", "gender")


def candidate_rows(meta: list, known: set) -> list:
    """Rows of the dataset's meta.csv with a known region, not in `known`."""
    rows = []
    for row in meta:
        if row["image_id"] in known:
            continue
        try:
            region = region_for_study_type(row["study_type"])
        except ValueError:
            continue
        rows.append({**row, "region": region})
    return rows


def pick(candidates: list, val_per_region: int, test_per_region: int, seed: int = 0) -> dict:
    """id -> split: per region, a seeded permutation of the sorted ids, test
    first, then val (order-independent, like `select_totalseg.select_and_split`)."""
    rng = np.random.default_rng(seed)
    assignment = {}
    for region in REGIONS:
        ids = sorted(c["id"] for c in candidates if c["region"] == region)
        need = val_per_region + test_per_region
        if len(ids) < need:
            raise ValueError(f"region {region!r} has {len(ids)} eligible candidates, needs {need}")
        picked = [ids[i] for i in rng.permutation(len(ids))[:need]]
        assignment.update({sid: TEST_SPLIT for sid in picked[:test_per_region]})
        assignment.update({sid: VAL_SPLIT for sid in picked[test_per_region:]})
    return assignment


def _thread_zip(remote, local, lock, handles):
    if not hasattr(local, "zf"):
        handle = remote.clone()
        with lock:
            handles.append(handle)
        local.zf = zipfile.ZipFile(io.BufferedReader(handle, buffer_size=1 << 18))
    return local.zf


def extend(manifest: dict, url: str = FULL_ZIP_URL, out_dir: str = DEFAULT_OUT_DIR, seed: int = 0,
           val_per_region: int = VAL_PER_REGION, test_per_region: int = TEST_PER_REGION, workers: int = 2,
           dry_run: bool = False) -> dict:
    known = {s["id"] for s in manifest["subjects"]}
    if any(s["split"] in (VAL_SPLIT, TEST_SPLIT) for s in manifest["subjects"]):
        raise SystemExit(f"the manifest already has {VAL_SPLIT}/{TEST_SPLIT} entries; not re-drawing them")
    remote = HttpRangeFile(url, block_size=256 << 10)
    zf = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 18))
    rows = candidate_rows(read_meta(zf), known)
    print(f"[extend_eval_splits] {len(rows)} candidates with a known region outside the manifest")

    local, lock, handles = threading.local(), threading.Lock(), [remote]

    def header(row):
        return row, read_ct_header(_thread_zip(remote, local, lock, handles), row["image_id"])

    eligible, info = [], {}
    with ThreadPoolExecutor(workers) as pool:
        for row, (shape, zooms) in pool.map(header, rows):
            if not passes_extent_filter(shape, zooms):
                continue
            sid = row["image_id"]
            eligible.append({"id": sid, "region": row["region"]})
            info[sid] = {"study_type": row["study_type"], "region": row["region"],
                         "shape": [int(v) for v in shape[:3]], "spacing": [round(float(v), 6) for v in zooms[:3]]}
            info[sid]["dataset_meta"] = {k: row.get(k, "") for k in META_FIELDS}
    counts = {r: sum(c["region"] == r for c in eligible) for r in REGIONS}
    print(f"[extend_eval_splits] {len(eligible)} pass the extent filter: {counts}")
    assignment = pick(eligible, val_per_region, test_per_region, seed)
    for split in (VAL_SPLIT, TEST_SPLIT):
        print(f"  {split}: {sum(v == split for v in assignment.values())}")
    if dry_run:
        return manifest

    def extract(sid):
        entry_info = {k: v for k, v in info[sid].items() if k != "dataset_meta"}
        entry = extract_subject(_thread_zip(remote, local, lock, handles), sid, entry_info, assignment[sid],
                                out_dir)
        return {**entry, "source": SOURCE, **info[sid]["dataset_meta"]}

    added = []
    with ThreadPoolExecutor(workers) as pool:
        for n, entry in enumerate(pool.map(extract, sorted(assignment)), 1):
            added.append(entry)
            total = sum(h.bytes_fetched for h in handles) / 1e6
            print(f"[extend_eval_splits] {n}/{len(assignment)} {entry['name']} ({entry['split']}), "
                  f"{total:.0f} MB read so far")
    extended = dict(manifest)
    extended["subjects"] = list(manifest["subjects"]) + added
    extended["sources"] = {SOURCE: {"dataset": "TotalSegmentator v2.0.1 (full)",
                                    "zenodo_record": FULL_ZENODO_RECORD,
                                    "zip": os.path.basename(url.split("?")[0]), "license": "CC-BY-4.0"}}
    extended["selection"] = {**manifest.get("selection", {}), "eval_extra": {
        "seed": seed, "val_per_region": val_per_region, "test_per_region": test_per_region,
        "n_candidates": len(rows), "n_eligible": len(eligible), "eligible_per_region": counts,
        "numpy_version": np.__version__}}
    return extended


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-per-region", type=int, default=VAL_PER_REGION)
    parser.add_argument("--test-per-region", type=int, default=TEST_PER_REGION)
    parser.add_argument("--workers", type=int, default=2, help="parallel readers (Zenodo rate-limits)")
    parser.add_argument("--dry-run", action="store_true", help="print candidates and picks, download nothing")
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    extended = extend(manifest, out_dir=args.out_dir, seed=args.seed, val_per_region=args.val_per_region,
                      test_per_region=args.test_per_region, workers=args.workers, dry_run=args.dry_run)
    if args.dry_run:
        return
    tmp = args.manifest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(extended, stream, indent=2)
        stream.write("\n")
    os.replace(tmp, args.manifest)
    print(f"[extend_eval_splits] {len(extended['subjects'])} subjects in {args.manifest}")


if __name__ == "__main__":
    main()
