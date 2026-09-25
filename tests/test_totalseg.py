import json

import nibabel as nib
import numpy as np
import pytest

import totalseg
from anatomy import CLASS_LAYOUT_VERSION


def _write_subject(tmp_path, sid, split, data=None):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4) if data is None else data
    path = tmp_path / "totalseg" / sid / "ct.nii.gz"
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.5, 1.0])), str(path))
    return {"id": sid, "name": f"ts_{sid}", "split": split, "region": "thorax",
            "study_type": "ct thorax", "shape": list(data.shape), "spacing": [1.5, 1.5, 1.5],
            "path": str(path), "sha256": "ab" * 32}


def _write_labels(tmp_path, sid, shape):
    labels = np.zeros(shape, dtype=np.uint8)
    labels[0, 0, 0] = 1
    path = tmp_path / "totalseg" / sid / "labels.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.diag([1.5, 1.5, 1.5, 1.0])), str(path))
    return str(path)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    subjects = [_write_subject(tmp_path, "s0001", "train"),
                _write_subject(tmp_path, "s0002", "test")]
    subjects.append({**subjects[0], "id": "s0003", "name": "ts_s0003", "split": "val",
                     "path": str(tmp_path / "missing" / "ct.nii.gz")})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"subjects": subjects}))
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(path))
    return subjects


def test_is_totalseg_checks_prefix():
    assert totalseg.is_totalseg("ts_s0001")
    assert not totalseg.is_totalseg("ct_chest")


def test_available_names_lists_only_existing_files(manifest):
    assert totalseg.available_names() == ["ts_s0001", "ts_s0002"]


def test_split_names_returns_split_members(manifest):
    assert totalseg.split_names("train") == ["ts_s0001"]
    assert totalseg.split_names("test") == ["ts_s0002"]


def test_split_names_raises_when_a_member_is_missing(manifest):
    with pytest.raises(FileNotFoundError, match="tools.select_totalseg"):
        totalseg.split_names("val")


def test_split_names_rejects_unknown_split(manifest):
    with pytest.raises(ValueError):
        totalseg.split_names("holdout")


def test_version_uses_manifest_sha(manifest):
    assert totalseg.version("ts_s0001") == "sha256:" + "ab" * 32


def test_load_volume_returns_float32_ras_and_spacing(manifest):
    volume, spacing = totalseg.load_volume("ts_s0001")
    assert volume.dtype == np.float32
    assert volume.shape == (2, 3, 4)
    assert volume[1, 2, 3] == 23.0
    assert spacing == (1.5, 1.5, 1.5)


def test_load_volume_reorients_to_ras(tmp_path, monkeypatch):
    entry = _write_subject(tmp_path, "s0009", "train")
    image = nib.load(entry["path"])
    flipped = nib.Nifti1Image(np.asarray(image.dataobj)[::-1], np.diag([-1.5, 1.5, 1.5, 1.0]))
    nib.save(flipped, entry["path"])            # stored L->R flipped, affine says so
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"subjects": [entry]}))
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(path))
    volume, _ = totalseg.load_volume("ts_s0009")
    assert volume[1, 2, 3] == 23.0              # back in original (RAS) order


def test_load_volume_unknown_and_missing(manifest):
    with pytest.raises(ValueError, match="unknown TotalSegmentator volume"):
        totalseg.load_volume("ts_nope")
    with pytest.raises(FileNotFoundError, match="tools.select_totalseg"):
        totalseg.load_volume("ts_s0003")


def test_missing_manifest_means_no_volumes(tmp_path, monkeypatch):
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(tmp_path / "none.json"))
    assert totalseg.available_names() == []


@pytest.fixture
def labelled_manifest(tmp_path, monkeypatch):
    with_labels = _write_subject(tmp_path, "l0001", "train")
    with_labels["labels_path"] = _write_labels(tmp_path, "l0001", (2, 3, 4))
    with_labels["classes_present"] = ["liver", "skeleton"]
    with_labels["label_layout_version"] = CLASS_LAYOUT_VERSION
    with_labels["contrast"] = True

    no_labels_key = _write_subject(tmp_path, "l0002", "train")

    missing_labels_file = _write_subject(tmp_path, "l0003", "train")
    missing_labels_file["labels_path"] = str(tmp_path / "totalseg" / "l0003" / "labels.nii.gz")

    missing_labels_file["label_layout_version"] = CLASS_LAYOUT_VERSION
    subjects = [with_labels, no_labels_key, missing_labels_file]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"label_layout_version": CLASS_LAYOUT_VERSION,
                                "subjects": subjects}))
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(path))
    return subjects


def test_has_labels_true_only_when_file_exists(labelled_manifest):
    assert totalseg.has_labels("ts_l0001") is True
    assert totalseg.has_labels("ts_l0002") is False    # no labels_path key
    assert totalseg.has_labels("ts_l0003") is False    # labels_path set but file missing


def test_load_labels_matches_volume_shape_and_dtype(labelled_manifest):
    labels = totalseg.load_labels("ts_l0001")
    volume, _ = totalseg.load_volume("ts_l0001")
    assert labels.dtype == np.uint8
    assert labels.shape == volume.shape


def test_load_labels_raises_when_missing(labelled_manifest):
    with pytest.raises(FileNotFoundError, match="tools.select_totalseg"):
        totalseg.load_labels("ts_l0002")
    with pytest.raises(FileNotFoundError, match="tools.select_totalseg"):
        totalseg.load_labels("ts_l0003")


def test_classes_present_returns_manifest_list(labelled_manifest):
    assert totalseg.classes_present("ts_l0001") == ["liver", "skeleton"]
    assert totalseg.classes_present("ts_l0002") == []


def test_is_contrast_returns_flag(labelled_manifest):
    assert totalseg.is_contrast("ts_l0001") is True
    assert totalseg.is_contrast("ts_l0002") is False


def test_has_labels_rejects_stale_label_layout(labelled_manifest, monkeypatch):
    path = totalseg.MANIFEST_PATH
    manifest = json.loads(open(path).read())
    manifest["subjects"][0]["label_layout_version"] = "anatomy-v1"
    open(path, "w").write(json.dumps(manifest))

    with pytest.raises(ValueError, match="label_layout_version.*anatomy-v2"):
        totalseg.has_labels("ts_l0001")


def test_load_labels_rejects_missing_label_layout(labelled_manifest):
    path = totalseg.MANIFEST_PATH
    manifest = json.loads(open(path).read())
    del manifest["subjects"][0]["label_layout_version"]
    open(path, "w").write(json.dumps(manifest))

    with pytest.raises(ValueError, match="label_layout_version.*anatomy-v2"):
        totalseg.load_labels("ts_l0001")


def test_has_labels_accepts_current_fixture_subject_when_manifest_is_patched(
        labelled_manifest, monkeypatch):
    entry = json.loads(open(totalseg.MANIFEST_PATH).read())["subjects"][0]
    monkeypatch.setattr(totalseg, "_subjects", lambda: {entry["name"]: entry})
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", "/missing/manifest.json")

    assert totalseg.has_labels("ts_l0001") is True


def test_label_metadata_reports_anatomy_v2_layout_and_classes(labelled_manifest):
    metadata = totalseg.label_metadata("ts_l0001")

    assert metadata["dimensions"] == [2, 3, 4]
    assert metadata["scalar_type"] == "uint8"
    assert metadata["order"] == "F"
    assert metadata["label_layout_version"] == CLASS_LAYOUT_VERSION
    assert metadata["class_ids"] == {"skeleton": 1, "liver": 5}
    assert metadata["dataset_version"] == "sha256:" + "ab" * 32


def test_label_metadata_rejects_unlabeled_dataset(labelled_manifest):
    with pytest.raises(FileNotFoundError, match="no label volume"):
        totalseg.label_metadata("ts_l0002")


@pytest.mark.parametrize("bad", [
    np.zeros((2, 2), dtype=np.uint8),
    np.zeros((0, 2, 2), dtype=np.uint8),
    np.zeros((1, 1, 1), dtype=np.float32),
    np.array([[[np.nan]]], dtype=np.float64),
])
def test_iter_label_chunks_rejects_invalid_labels(bad):
    with pytest.raises(ValueError):
        list(totalseg.iter_label_chunks(bad, chunk_bytes=8))


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_load_labels_rejects_non_uint8_source(dtype, labelled_manifest):
    manifest = json.loads(open(totalseg.MANIFEST_PATH).read())
    entry = manifest["subjects"][0]
    source = nib.load(entry["labels_path"])
    invalid_path = entry["labels_path"].replace("labels.nii.gz", f"labels-{dtype.__name__}.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(source.dataobj, dtype=dtype), source.affine), invalid_path)
    entry["labels_path"] = invalid_path
    open(totalseg.MANIFEST_PATH, "w").write(json.dumps(manifest))

    with pytest.raises(ValueError, match="uint8"):
        totalseg.load_labels("ts_l0001")


def test_label_digest_changes_when_label_file_changes(labelled_manifest):
    first = totalseg.label_digest("ts_l0001")
    path = json.loads(open(totalseg.MANIFEST_PATH).read())["subjects"][0]["labels_path"]
    with open(path, "ab") as stream:
        stream.write(b"changed")

    second = totalseg.label_digest("ts_l0001")

    assert second != first
