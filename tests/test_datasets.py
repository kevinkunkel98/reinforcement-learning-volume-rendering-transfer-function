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
    assert metadata["intensity_range"] == [0.0, 23.0]
    assert metadata["total_bytes"] == 24 * 4
    assert metadata["chunks"]


def test_chunk_round_trip_reconstructs_float32_c_order_volume():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    chunks = list(datasets.iter_volume_chunks(volume, chunk_bytes=16))
    restored = np.frombuffer(b"".join(chunks), dtype=np.float32).reshape(volume.shape, order="C")

    np.testing.assert_array_equal(restored, volume)
    assert [len(chunk) for chunk in chunks] == [16, 16, 16, 16, 16, 16]


def test_get_volume_chunk_uses_metadata_chunk_descriptors(monkeypatch):
    volume = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    monkeypatch.setattr(datasets, "load_dataset", lambda name: (volume, (1.0, 1.0, 1.0)))

    metadata = datasets.dataset_metadata("synthetic")
    chunk = datasets.get_volume_chunk("synthetic", 0)

    assert metadata["chunks"][0]["index"] == 0
    assert len(chunk) == metadata["chunks"][0]["byte_length"]
    assert chunk == volume.tobytes(order="C")[: len(chunk)]


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
