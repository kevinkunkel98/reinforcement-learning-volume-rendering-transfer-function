import io
import os
import zipfile

from tools import fetch_scans
from tools.fetch_scans import HttpRangeFile


class BytesRangeFile(HttpRangeFile):
    """HttpRangeFile over an in-memory payload instead of a URL."""

    def __init__(self, payload: bytes, **kwargs):
        self.payload, self.requests = payload, 0
        super().__init__("memory://zip", **kwargs)

    def _get(self, start, end):
        self.requests += 1
        return self.payload[start:end + 1], f"bytes {start}-{end}/{len(self.payload)}"

    def clone(self):
        twin = BytesRangeFile.__new__(BytesRangeFile)
        HttpRangeFile.__init__(twin, self.url, self.block_size, self.cached_blocks, self.retries,
                               self.size, dict(self._blocks))
        twin.payload, twin.requests = self.payload, 0
        return twin


def _zip_bytes(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_range_file_reads_members_of_a_remote_zip_across_blocks():
    members = {f"s{i:04d}/ct.nii.gz": os.urandom(5000 + 97 * i) for i in range(20)}
    remote = BytesRangeFile(_zip_bytes(members), block_size=1024, cached_blocks=4)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=512)) as zf:
        assert sorted(zf.namelist()) == sorted(members)
        for name in ("s0003/ct.nii.gz", "s0019/ct.nii.gz", "s0000/ct.nii.gz"):
            assert zf.read(name) == members[name]


def test_clone_opens_the_zip_from_the_cached_central_directory():
    members = {f"m{i}": os.urandom(3000) for i in range(30)}
    remote = BytesRangeFile(_zip_bytes(members), block_size=4096, cached_blocks=64)
    zipfile.ZipFile(io.BufferedReader(remote)).close()
    twin = remote.clone()
    with zipfile.ZipFile(io.BufferedReader(twin)) as zf:
        assert twin.requests == 0
        assert zf.read("m7") == members["m7"]


def test_wanted_subjects_filters_the_manifest_by_split(monkeypatch):
    manifest = {"subjects": [{"name": "ts_a", "split": "test"}, {"name": "ts_b", "split": "train"},
                             {"name": "ts_c", "split": "train_extra"}]}
    monkeypatch.setattr(fetch_scans.totalseg, "_manifest", lambda: manifest)
    assert [s["name"] for s in fetch_scans.wanted_subjects(("test",))] == ["ts_a"]
    assert [s["name"] for s in fetch_scans.wanted_subjects(fetch_scans.ALL_SPLITS)] == ["ts_a", "ts_b", "ts_c"]
