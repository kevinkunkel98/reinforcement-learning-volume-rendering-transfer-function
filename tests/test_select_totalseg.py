import pytest

from tools.select_totalseg import (
    CLASS_NAMES, SPLIT_COUNTS, build_label_volume, class_for_structure,
    passes_extent_filter, region_for_study_type, select_and_split,
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


def test_class_for_structure_maps_prefixes():
    assert class_for_structure("rib_left_4") == "skeleton"
    assert class_for_structure("lung_upper_lobe_left") == "lungs"
    assert class_for_structure("aorta") == "vessels"
    assert class_for_structure("autochthon_left") == "muscle"
    assert class_for_structure("liver") == "organs"
    assert class_for_structure("unknown_thing") is None


def test_build_label_volume_assigns_ids():
    shape = (4, 4, 4)
    mask_a = np.zeros(shape, dtype=bool)
    mask_a[0, 0, 0] = True
    mask_b = np.zeros(shape, dtype=bool)
    mask_b[1, 1, 1] = True
    mask_c = np.zeros(shape, dtype=bool)
    mask_c[2, 2, 2] = True
    masks = {"rib_left_1": mask_a, "liver": mask_b, "unknown": mask_c}

    labels = build_label_volume(masks, shape)

    assert labels.dtype == np.uint8
    assert labels[0, 0, 0] == CLASS_NAMES.index("skeleton") + 1
    assert labels[1, 1, 1] == CLASS_NAMES.index("organs") + 1
    assert labels[2, 2, 2] == 0


def test_build_label_volume_later_class_wins_on_overlap():
    shape = (2, 2, 2)
    rib = np.zeros(shape, dtype=bool)
    rib[0, 0, 0] = True
    aorta = np.zeros(shape, dtype=bool)
    aorta[0, 0, 0] = True
    masks = {"rib_left_1": rib, "aorta": aorta}

    labels = build_label_volume(masks, shape)

    assert labels[0, 0, 0] == CLASS_NAMES.index("vessels") + 1


import gzip
import hashlib
import json
import os
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from tools.select_totalseg import build_manifest, _subject_label_volume

META_HEADER = "image_id;age;gender;institute;study_type;split;manufacturer;scanner_model;kvp;pathology;pathology_location"


def _nifti_gz(shape, zoom, data=None):
    if data is None:
        data = np.zeros(shape, dtype=np.int16)
    image = nib.Nifti1Image(data, np.diag([zoom, zoom, zoom, 1.0]))
    return gzip.compress(image.to_bytes())


def _mask_gz(shape, zoom, corner):
    # A 12x12x12 block (1728 voxels) clears MIN_LABEL_VOXELS, so the mask
    # counts as "present" the same way a real anatomical structure would.
    data = np.zeros(shape, dtype=np.uint8)
    x, y, z = corner
    data[x:x + 12, y:y + 12, z:z + 12] = 1
    return _nifti_gz(shape, zoom, data)


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
            zf.writestr(f"{sid}/segmentations/rib_left_1.nii.gz",
                        _mask_gz(shape, 1.5, (0, 0, 0)))
            zf.writestr(f"{sid}/segmentations/liver.nii.gz",
                        _mask_gz(shape, 1.5, (20, 20, 20)))


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
    expected_path = os.path.relpath(out_dir / first["id"] / "ct.nii.gz")
    assert first["path"] == expected_path
    assert Path(expected_path).exists()
    assert first["sha256"] == hashlib.sha256(Path(first["path"]).read_bytes()).hexdigest()
    assert len(first["sha256"]) == 64
    assert not (out_dir / "s0005").exists()     # pelvis not in split_counts
    assert os.path.exists(first["labels_path"])
    assert first["classes_present"] == ["organs", "skeleton"]
    assert first["contrast"] is False
    labels = np.asarray(nib.load(first["labels_path"]).dataobj)
    assert labels[0, 0, 0] == CLASS_NAMES.index("skeleton") + 1
    assert labels[20, 20, 20] == CLASS_NAMES.index("organs") + 1
    assert manifest["source"]["zenodo_record"] == "10047263"
    assert manifest["selection"]["seed"] == 0
    assert manifest["selection"]["n_candidates"] == 4
    assert "numpy_version" in manifest["selection"]
    assert list(out_dir.rglob("*.tmp")) == []
    json.dumps(manifest)                        # serializable


def test_subject_label_volume_raises_clearly_when_nothing_matches(tmp_path):
    """Every mask filtered out (an unrecognized structure name, or none at
    all) used to leave `image` unbound, crashing on `image.affine` with a
    confusing NameError instead of saying what actually went wrong."""
    shape = (32, 32, 32)
    zip_path = tmp_path / "no_match.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("s0001/segmentations/unrecognized_structure.nii.gz",
                    _mask_gz(shape, 1.5, (0, 0, 0)))

    with zipfile.ZipFile(zip_path) as zf:
        with pytest.raises(ValueError, match="no recognized structure"):
            _subject_label_volume(zf, "s0001", shape)
