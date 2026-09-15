import pytest

from tools.select_totalseg import (
    SPLIT_COUNTS, passes_extent_filter, region_for_study_type, select_and_split,
)


def test_region_for_study_type_maps_all_known_types():
    assert region_for_study_type("ct thorax-neck") == "thorax"
    assert region_for_study_type("ct angiography abdomen-pelvis") == "abdomen_pelvis"
    assert region_for_study_type("ct neck-thorax-abdomen-pelvis") == "trunk"
    assert region_for_study_type("CT Polytrauma ") == "whole_body"


def test_region_for_study_type_rejects_unknown():
    with pytest.raises(ValueError, match="unknown study type"):
        region_for_study_type("ct knee")


def test_passes_extent_filter_uses_physical_extent():
    assert passes_extent_filter((200, 200, 120), (1.5, 1.5, 1.5))       # 300 x 300 x 180 mm
    assert not passes_extent_filter((140, 200, 120), (1.5, 1.5, 1.5))   # 210 mm in-plane
    assert not passes_extent_filter((200, 200, 90), (1.5, 1.5, 1.5))    # 135 mm superior-inferior


def _candidates(region, n):
    return [{"id": f"{region}{i:02d}", "region": region} for i in range(n)]


def test_select_and_split_meets_counts_per_region():
    candidates = []
    for region in SPLIT_COUNTS:
        candidates += _candidates(region, 12)
    assignment = select_and_split(candidates, seed=0)
    for region, (n_train, n_val, n_test) in SPLIT_COUNTS.items():
        splits = [s for sid, s in assignment.items() if sid.startswith(region)]
        assert splits.count("train") == n_train
        assert splits.count("val") == n_val
        assert splits.count("test") == n_test
    assert sum(1 for s in assignment.values() if s == "train") == 20
    assert sum(1 for s in assignment.values() if s == "val") == 4
    assert sum(1 for s in assignment.values() if s == "test") == 6


def test_select_and_split_is_deterministic_and_seed_dependent():
    candidates = _candidates("thorax", 20)
    counts = {"thorax": (3, 1, 1)}
    first = select_and_split(candidates, counts, seed=0)
    assert first == select_and_split(list(reversed(candidates)), counts, seed=0)
    assert first != select_and_split(candidates, counts, seed=1)


def test_select_and_split_raises_when_region_too_small():
    with pytest.raises(ValueError, match="needs 5"):
        select_and_split(_candidates("thorax", 4), {"thorax": (3, 1, 1)}, seed=0)


import gzip
import json
import zipfile

import nibabel as nib
import numpy as np

from tools.select_totalseg import build_manifest

META_HEADER = "image_id;age;gender;institute;study_type;split;manufacturer;scanner_model;kvp;pathology;pathology_location"


def _nifti_gz(shape, zoom):
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.diag([zoom, zoom, zoom, 1.0]))
    return gzip.compress(image.to_bytes())


def _fake_zip(path):
    rows = [
        ("s0001", "ct thorax", (200, 200, 120)),
        ("s0002", "ct thorax-neck", (200, 200, 120)),
        ("s0003", "ct thorax", (200, 200, 120)),
        ("s0004", "ct thorax", (100, 100, 120)),   # too small in-plane: filtered out
        ("s0005", "ct pelvis", (200, 200, 120)),
    ]
    with zipfile.ZipFile(path, "w") as zf:
        meta = [META_HEADER] + [f"{sid};50;f;A;{st};train;x;y;120;none;none" for sid, st, _ in rows]
        zf.writestr("meta.csv", "﻿" + "\n".join(meta) + "\n")
        for sid, _, shape in rows:
            zf.writestr(f"{sid}/ct.nii.gz", _nifti_gz(shape, 1.5))
            zf.writestr(f"{sid}/segmentations/liver.nii.gz", b"not read")


def test_build_manifest_selects_extracts_and_describes(tmp_path):
    zip_path = tmp_path / "subset.zip"
    _fake_zip(zip_path)
    out_dir = tmp_path / "totalseg"

    manifest = build_manifest(str(zip_path), str(out_dir), seed=0,
                              split_counts={"thorax": (1, 1, 1)})

    subjects = manifest["subjects"]
    assert sorted(s["id"] for s in subjects) == ["s0001", "s0002", "s0003"]
    assert sorted(s["split"] for s in subjects) == ["test", "train", "val"]
    first = subjects[0]
    assert first["name"] == f"ts_{first['id']}"
    assert first["region"] == "thorax"
    assert first["shape"] == [200, 200, 120]
    assert first["spacing"] == [1.5, 1.5, 1.5]
    assert (out_dir / first["id"] / "ct.nii.gz").exists()
    assert first["path"] == str(out_dir / first["id"] / "ct.nii.gz")
    assert len(first["sha256"]) == 64
    assert not (out_dir / "s0005").exists()     # pelvis not in split_counts
    assert manifest["source"]["zenodo_record"] == "10047263"
    assert manifest["selection"]["seed"] == 0
    json.dumps(manifest)                        # serializable
