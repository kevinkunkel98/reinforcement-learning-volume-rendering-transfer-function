"""Scans: the manifest, loading, chunk transport, and the download tools."""
import io
import os
import zipfile

import numpy as np
import pytest

import scans
from tools import fetch_scans
from tools.fetch_scans import HttpRangeFile
from tools.select_totalseg import _subject_label_volume, build_label_volume


def test_manifest_and_loading():
    assert scans.available_names() == ["ts_s9001", "ts_s9002"]
    assert scans.split_names("test") == ["ts_s9001"] and scans.demo_names() == ["ts_s9001", "ts_s9002"]
    volume, spacing = scans.load_volume("ts_s9001")
    labels = scans.load_labels("ts_s9001")
    assert volume.dtype == np.float32 and labels.shape == volume.shape and spacing == (4.0, 4.0, 4.0)
    with pytest.raises(ValueError):
        scans.subject("ts_nope")


def test_chunks_reassemble_the_volume_in_fortran_order(monkeypatch):
    monkeypatch.setattr(scans, "CHUNK_BYTES", 64 * 1024)
    metadata = scans.volume_metadata("ts_s9001")
    raw = b"".join(scans.volume_chunk("ts_s9001", i) for i in range(metadata["chunk_count"]))
    volume = np.frombuffer(raw, dtype="<f4").reshape(metadata["dimensions"], order="F")
    assert np.array_equal(volume, scans.load_volume("ts_s9001")[0])
    labels = scans.label_metadata("ts_s9001")
    assert labels["classes"] == list(labels["class_ids"]) and len(labels["class_ids"]) == 8
    with pytest.raises(IndexError):
        scans.label_chunk("ts_s9001", labels["chunk_count"])


class BytesRangeFile(HttpRangeFile):
    """HttpRangeFile over an in-memory payload instead of a URL."""

    def __init__(self, payload: bytes, **kwargs):
        self.payload, self.requests = payload, 0
        super().__init__("memory://zip", **kwargs)

    def _get(self, start, end):
        self.requests += 1
        return self.payload[start:end + 1], f"bytes {start}-{end}/{len(self.payload)}"


def _zip_bytes(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_range_file_reads_a_remote_zip_across_blocks():
    members = {f"s{i:04d}/ct.nii.gz": os.urandom(5000 + 97 * i) for i in range(20)}
    remote = BytesRangeFile(_zip_bytes(members), block_size=1024, cached_blocks=4)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=512)) as zf:
        assert all(zf.read(name) == members[name] for name in ("s0003/ct.nii.gz", "s0019/ct.nii.gz"))
    twin = remote.clone()
    assert twin.size == remote.size and twin._blocks == remote._blocks


def test_fetch_selects_by_split_and_name():
    assert [s["name"] for s in fetch_scans.wanted_subjects(("test",))] == ["ts_s9001"]
    assert [s["name"] for s in fetch_scans.wanted_subjects((), ("ts_s9002",))] == ["ts_s9002"]
    assert fetch_scans.fetch(("test",)) == []          # already present, checksum matches


def test_label_volume_folds_masks_with_later_classes_on_top():
    shape = (4, 4, 4)
    masks = {"liver": np.zeros(shape, bool), "heart": np.zeros(shape, bool)}
    masks["liver"][:2] = True
    masks["heart"][1:3] = True
    labels = build_label_volume(masks, shape)
    import nibabel as nib
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, mask in masks.items():
            zf.writestr(f"s1/segmentations/{name}.nii.gz",
                        __import__("gzip").compress(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)).to_bytes()))
    with zipfile.ZipFile(buffer) as zf:
        streamed, _affine = _subject_label_volume(zf, "s1", shape)
    assert np.array_equal(streamed, labels) and set(np.unique(labels)) == {0, 3, 5}


def test_eval_extra_selection_skips_known_patients_and_is_stratified():
    from tools import extend_eval_splits as ext
    meta = ([{"image_id": f"s{i:04d}", "study_type": st} for i, st in enumerate(
        ["ct polytrauma", "ct thorax-abdomen-pelvis", "ct thorax", "ct abdomen-pelvis"] * 5)]
        + [{"image_id": "s9999", "study_type": "ct head"}])
    rows = ext.candidate_rows(meta, known={"s0000"})
    assert "s0000" not in {r["image_id"] for r in rows} and "s9999" not in {r["image_id"] for r in rows}
    candidates = [{"id": r["image_id"], "region": r["region"]} for r in rows]
    assignment = ext.pick(candidates, val_per_region=1, test_per_region=2, seed=0)
    assert assignment == ext.pick(list(reversed(candidates)), 1, 2, seed=0)   # order-independent
    for region in ext.REGIONS:
        splits = [assignment[c["id"]] for c in candidates if c["region"] == region and c["id"] in assignment]
        assert sorted(splits) == ["test_extra", "test_extra", "val_extra"]
    with pytest.raises(ValueError):
        ext.pick(candidates, val_per_region=3, test_per_region=3)


def test_fetch_reads_each_subject_from_its_source_zip(monkeypatch):
    entries = [{"name": "ts_a", "split": "test"}, {"name": "ts_b", "split": "test_extra", "source": "full"}]
    monkeypatch.setattr(fetch_scans, "wanted_subjects", lambda splits, names=(): entries)
    monkeypatch.setattr(fetch_scans, "is_present", lambda entry: False)
    calls = []
    monkeypatch.setattr(fetch_scans, "_fetch_from",
                        lambda url, missing, out_dir, workers: calls.append((url, [e["name"] for e in missing])) or [])
    fetch_scans.fetch(("test", "test_extra"))
    assert calls == [(fetch_scans.ZIP_URL, ["ts_a"]), (fetch_scans.FULL_ZIP_URL, ["ts_b"])]


def test_rate_limit_waits_until_the_window_resets():
    wait = fetch_scans._retry_after_seconds
    assert wait({"X-RateLimit-Reset": "1030", "Retry-After": "0"}, now=1000.0) == 31.0
    assert wait({"Retry-After": "12"}, now=1000.0) == 12.0
    assert wait({"Retry-After": "0"}, now=1000.0) == fetch_scans.DEFAULT_RETRY_AFTER
