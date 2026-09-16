import hashlib
import os
from unittest.mock import patch

import numpy as np
import pytest

import datasets
from phantom import SIZE_DEFAULT


def test_load_dataset_synthetic_matches_phantom():
    volume, spacing = datasets.load_dataset("synthetic")
    assert volume.shape == (SIZE_DEFAULT, SIZE_DEFAULT, SIZE_DEFAULT)
    assert spacing == (1.0, 1.0, 1.0)


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError):
        datasets.load_dataset("not_a_real_dataset")


def test_ensure_downloaded_skips_download_when_cached_and_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    content = b"fake nrrd bytes"
    digest = hashlib.sha256(content).hexdigest()
    fake_spec = {"filename": "fake.nrrd", "url": "https://example.invalid/fake.nrrd", "sha256": digest}
    os.makedirs("data", exist_ok=True)
    with open("data/fake.nrrd", "wb") as f:
        f.write(content)

    with patch.dict(datasets.DATASETS, {"fake": fake_spec}), \
         patch("datasets.urllib.request.urlretrieve") as mock_retrieve:
        path = datasets._ensure_downloaded("fake")
    mock_retrieve.assert_not_called()
    assert path == "data/fake.nrrd"


def test_ensure_downloaded_raises_and_cleans_up_on_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_spec = {"filename": "fake.nrrd", "url": "https://example.invalid/fake.nrrd",
                 "sha256": "0" * 64}  # deliberately wrong

    def fake_retrieve(url, path):
        with open(path, "wb") as f:
            f.write(b"whatever was downloaded")

    with patch.dict(datasets.DATASETS, {"fake": fake_spec}), \
         patch("datasets.urllib.request.urlretrieve", side_effect=fake_retrieve):
        with pytest.raises(ValueError, match="checksum mismatch"):
            datasets._ensure_downloaded("fake")
    assert not os.path.exists("data/fake.nrrd")


def test_dataset_metadata_matches_loaded_volume(monkeypatch):
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    monkeypatch.setattr(datasets, "load_dataset", lambda name: (volume, (0.5, 0.6, 0.7)))

    metadata = datasets.dataset_metadata("synthetic")

    assert metadata["name"] == "synthetic"
    assert metadata["version"] == "synthetic-v1"
    assert metadata["dimensions"] == [2, 3, 4]
    assert metadata["spacing"] == [0.5, 0.6, 0.7]
    assert metadata["scalar_type"] == "float32"
    assert metadata["orientation"] == "dataset-normalized"
    assert metadata["order"] == "F"
    assert metadata["axis_mapping"] == "numpy axis0 -> VTK X; axis1 -> VTK Y; axis2 -> VTK Z"
    assert metadata["intensity_range"] == [0.0, 23.0]
    assert metadata["total_bytes"] == 24 * 4
    assert metadata["chunks"]


def test_metadata_chunk_descriptors_use_arithmetic_offsets(monkeypatch):
    volume = np.arange(10, dtype=np.float32).reshape(1, 2, 5)
    monkeypatch.setattr(datasets, "load_dataset", lambda name: (volume, (1.0, 1.0, 1.0)))

    metadata = datasets.dataset_metadata("synthetic")

    assert metadata["chunk_count"] == 1
    assert metadata["chunks"] == [{"index": 0, "byte_offset": 0, "byte_length": 40}]

    volume = np.arange(10, dtype=np.float32).reshape(1, 2, 5)
    monkeypatch.setattr(datasets, "DEFAULT_CHUNK_BYTES", 16)
    metadata = datasets.dataset_metadata("synthetic")
    assert metadata["chunks"] == [
        {"index": 0, "byte_offset": 0, "byte_length": 16},
        {"index": 1, "byte_offset": 16, "byte_length": 16},
        {"index": 2, "byte_offset": 32, "byte_length": 8},
    ]


def test_metadata_descriptors_reconstruct_chunked_volume(monkeypatch):
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::-1]
    monkeypatch.setattr(datasets, "load_dataset", lambda name: (volume, (1.0, 1.0, 1.0)))
    monkeypatch.setattr(datasets, "DEFAULT_CHUNK_BYTES", 16)

    metadata = datasets.dataset_metadata("synthetic")
    payload = b"".join(
        datasets.get_volume_chunk("synthetic", descriptor["index"])
        for descriptor in metadata["chunks"]
    )
    restored = np.frombuffer(payload, dtype="<f4").reshape(metadata["dimensions"], order=metadata["order"])

    np.testing.assert_array_equal(restored, volume)
    assert [len(datasets.get_volume_chunk("synthetic", descriptor["index"]))
            for descriptor in metadata["chunks"]] == [descriptor["byte_length"] for descriptor in metadata["chunks"]]


def test_chunk_round_trip_reconstructs_float32_fortran_order_volume():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::-1]

    chunks = list(datasets.iter_volume_chunks(volume, chunk_bytes=16))
    restored = np.frombuffer(b"".join(chunks), dtype="<f4").reshape(volume.shape, order="F")

    np.testing.assert_array_equal(restored, volume)
    assert [len(chunk) for chunk in chunks] == [16, 16, 16, 16, 16, 16]


def test_get_volume_chunk_uses_metadata_chunk_descriptors(monkeypatch):
    volume = np.arange(12, dtype=np.float32).reshape(2, 2, 3)[:, ::-1, :]
    load_calls = 0

    def load_once(name):
        nonlocal load_calls
        load_calls += 1
        return volume, (1.0, 1.0, 1.0)

    monkeypatch.setattr(datasets, "load_dataset", load_once)
    monkeypatch.setattr(datasets, "DEFAULT_CHUNK_BYTES", 16)

    chunk = datasets.get_volume_chunk("synthetic", 1)

    expected = np.asfortranarray(volume, dtype=np.dtype("<f4")).tobytes(order="F")
    assert load_calls == 1
    assert chunk == expected[16:32]


def test_iter_volume_chunks_rejects_big_endian_float32():
    volume = np.arange(8, dtype=np.dtype(">f4")).reshape(2, 2, 2)

    with pytest.raises(ValueError, match="little-endian"):
        list(datasets.iter_volume_chunks(volume, chunk_bytes=16))


@pytest.mark.parametrize(
    "volume, error",
    [
        (np.arange(8, dtype=np.float64).reshape(2, 2, 2), "float32"),
        (np.array([[[0.0, np.nan]]], dtype=np.float32), "finite"),
    ],
)
def test_iter_volume_chunks_rejects_invalid_volume(volume, error):
    with pytest.raises(ValueError, match=error):
        list(datasets.iter_volume_chunks(volume, chunk_bytes=16))


def test_get_volume_chunk_rejects_invalid_index(monkeypatch):
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    monkeypatch.setattr(datasets, "load_dataset", lambda name: (volume, (1.0, 1.0, 1.0)))

    with pytest.raises(IndexError):
        datasets.get_volume_chunk("synthetic", 1)


@pytest.mark.parametrize("chunk_bytes", [0, -4, 3, 1.5, True])
def test_iter_volume_chunks_rejects_invalid_chunk_size(chunk_bytes):
    volume = np.zeros((1, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="chunk_bytes"):
        list(datasets.iter_volume_chunks(volume, chunk_bytes=chunk_bytes))


import json

import nibabel as nib

import totalseg


@pytest.fixture
def ts_volume(tmp_path, monkeypatch):
    path = tmp_path / "s0001" / "ct.nii.gz"
    path.parent.mkdir()
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    nib.save(nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.5, 1.0])), str(path))
    entry = {"id": "s0001", "name": "ts_s0001", "split": "train", "region": "thorax",
             "study_type": "ct thorax", "shape": [2, 3, 4], "spacing": [1.5, 1.5, 1.5],
             "path": str(path), "sha256": "cd" * 32}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"subjects": [entry]}))
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(manifest))
    datasets.load_dataset.cache_clear()
    yield "ts_s0001"
    datasets.load_dataset.cache_clear()


def test_list_datasets_includes_available_totalseg_volumes(ts_volume):
    assert ts_volume in datasets.list_datasets()


def test_load_dataset_totalseg_returns_ras_float32(ts_volume):
    volume, spacing = datasets.load_dataset(ts_volume)
    assert volume.dtype == np.float32 and volume.shape == (2, 3, 4)
    assert spacing == (1.5, 1.5, 1.5)
    canonical, _ = datasets.load_dataset(ts_volume, canonical=True)
    assert np.array_equal(canonical, volume)


def test_totalseg_version_metadata_and_camera(ts_volume):
    assert datasets._dataset_version(ts_volume) == "sha256:" + "cd" * 32
    metadata = datasets.dataset_metadata(ts_volume)
    assert metadata["orientation"] == "RAS"
    assert metadata["version"] == "sha256:" + "cd" * 32
    assert datasets.default_camera_for(ts_volume) == {"azimuth": 0.0, "elevation": 80.0, "zoom": 1.0}


def test_existing_datasets_keep_dataset_normalized_orientation(monkeypatch):
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    monkeypatch.setattr(datasets, "load_dataset", lambda name, canonical=False: (volume, (1.0, 1.0, 1.0)))
    assert datasets.dataset_metadata("synthetic")["orientation"] == "dataset-normalized"


def test_load_dataset_shares_one_cache_entry_for_equivalent_calls():
    # "synthetic" (and TotalSegmentator volumes) are RAS either way, so
    # load_dataset(n), load_dataset(n, False) and load_dataset(n, canonical=True)
    # must not each hold a separate multi-hundred-MB cache entry.
    datasets.load_dataset.cache_clear()
    datasets.load_dataset("synthetic")
    datasets.load_dataset("synthetic", False)
    datasets.load_dataset("synthetic", canonical=True)
    assert datasets.load_dataset.cache_info().currsize == 1
    datasets.load_dataset.cache_clear()


def test_volumes_for_split(ts_volume):
    assert datasets.volumes_for_split("train") == [ts_volume]
    assert datasets.volumes_for_split("out_of_source") == ["ct_chest", "ct_skull", "ct_cardio", "ct_abdomen"]
    with pytest.raises(ValueError):
        datasets.volumes_for_split("holdout")


def _write_nrrd_header(path, space, directions):
    text = ("NRRD0004\n# comment line\ntype: short\ndimension: 3\n"
            f"space: {space}\nsizes: 2 3 4\n"
            f"space directions: {directions}\n"
            "kinds: domain domain domain\nendian: little\nencoding: raw\n\n")
    path.write_bytes(text.encode("latin-1") + b"\x00" * 48)


def test_read_nrrd_header_parses_fields(tmp_path):
    path = tmp_path / "h.nrrd"
    _write_nrrd_header(path, "left-posterior-superior", "(1,0,0) (0,1,0) (0,0,2)")
    header = datasets._read_nrrd_header(str(path))
    assert header["space"] == "left-posterior-superior"
    assert header["space directions"] == "(1,0,0) (0,1,0) (0,0,2)"
    assert "type" in header and "NRRD0004" not in header


def test_reorient_lps_axis_aligned_flips_x_and_y():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    header = {"space": "left-posterior-superior", "space directions": "(1,0,0) (0,1,0) (0,0,2)"}
    ras, spacing = datasets._reorient_nrrd_to_ras(volume, header)
    assert np.array_equal(ras, volume[::-1, ::-1, :])
    assert spacing == (1.0, 1.0, 2.0)


def test_reorient_permutes_axes_and_spacing():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    # axis0 -> +superior (2 mm), axis1 -> +right (1 mm), axis2 -> -anterior (3 mm)
    header = {"space": "right-anterior-superior", "space directions": "(0,0,2) (1,0,0) (0,-3,0)"}
    ras, spacing = datasets._reorient_nrrd_to_ras(volume, header)
    expected = np.transpose(volume, (1, 2, 0))[:, ::-1, :]
    assert np.array_equal(ras, expected)
    assert spacing == (1.0, 3.0, 2.0)


def test_reorient_rejects_unsupported_space():
    with pytest.raises(ValueError, match="unsupported NRRD space"):
        datasets._reorient_nrrd_to_ras(np.zeros((2, 2, 2), np.float32),
                                       {"space": "scanner-xyz", "space directions": "(1,0,0) (0,1,0) (0,0,1)"})


def test_canonical_rejected_for_uncalibrated_and_mri():
    with pytest.raises(ValueError, match="no canonical RAS orientation"):
        datasets.load_dataset("mri_head", canonical=True)
    with pytest.raises(ValueError, match="no canonical RAS orientation"):
        datasets.load_dataset("stag_beetle", canonical=True)
    with pytest.raises(ValueError, match="no canonical RAS orientation"):
        datasets.load_dataset("ct_head", canonical=True)


@pytest.mark.slow
def test_load_nrrd_preserves_header_sizes():
    # The whole NRRD reorientation (_reorient_nrrd_to_ras) rests on VTK's
    # reader keeping the file's own axis order/extents; confirm that against
    # a real file's own declared "sizes" field.
    path = os.path.join("data", "CT-chest.nrrd")
    header = datasets._read_nrrd_header(path)
    expected_shape = tuple(int(x) for x in header["sizes"].split())
    assert datasets._load_nrrd(path)[0].shape == expected_shape
