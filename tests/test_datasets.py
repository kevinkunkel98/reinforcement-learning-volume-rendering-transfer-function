import hashlib
import os
from unittest.mock import patch

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
