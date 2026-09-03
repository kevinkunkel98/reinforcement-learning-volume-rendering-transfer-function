"""Real CT volumes, loaded as (numpy HU array, spacing) alongside the synthetic
phantom. Downloaded once, checksum-verified, and cached under data/ -- same
"download once, cache forever" precedent as the Whisper model in asr.py.

All four are de-identified public test data from the 3D Slicer project
(https://github.com/Slicer/SlicerTestingData), each verified by hand against
its registration in Slicer's own SampleData.py before adding it here: real
calibrated Hounsfield units (confirmed via the standard -1024/-3024 air-padding
value and realistic bone/soft-tissue HU distributions), not synthetic or
rescaled data.
"""
import hashlib
import os
import urllib.request

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from phantom import build_phantom

DATA_DIR = "data"

DATASETS = {
    "ct_chest": {
        "filename": "CT-chest.nrrd",
        "url": (
            "https://github.com/Slicer/SlicerTestingData/releases/download/"
            "SHA256/4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"
        ),
        "sha256": "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e",
    },
    "ct_skull": {
        "filename": "CT-brain.nrrd",
        "url": (
            "https://github.com/Slicer/SlicerTestingData/releases/download/"
            "SHA256/6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1"
        ),
        "sha256": "6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1",
    },
    "ct_cardio": {
        "filename": "CTA-cardio.nrrd",
        "url": (
            "https://github.com/Slicer/SlicerTestingData/releases/download/"
            "SHA256/3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2"
        ),
        "sha256": "3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2",
    },
    "ct_abdomen": {
        "filename": "Panoramix-cropped.nrrd",
        "url": (
            "https://github.com/Slicer/SlicerTestingData/releases/download/"
            "SHA256/146af87511520c500a3706b7b2bfb545f40d5d04dd180be3a7a2c6940e447433"
        ),
        "sha256": "146af87511520c500a3706b7b2bfb545f40d5d04dd180be3a7a2c6940e447433",
    },
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_downloaded(name: str) -> str:
    spec = DATASETS[name]
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, spec["filename"])
    if os.path.exists(path) and _sha256(path) == spec["sha256"]:
        return path
    print(f"[datasets] downloading {name} ({spec['url']}) -- one-time, cached afterward")
    urllib.request.urlretrieve(spec["url"], path)
    digest = _sha256(path)
    if digest != spec["sha256"]:
        os.remove(path)
        raise ValueError(f"checksum mismatch for {name}: got {digest}, expected {spec['sha256']}")
    return path


def _load_nrrd(path: str):
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(path)
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()
    spacing = image.GetSpacing()
    arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims[::-1]).astype(np.float32)
    arr = np.transpose(arr, (2, 1, 0))  # numpy axis0 -> VTK X, matching render()'s ravel(order="F")
    return arr, spacing


def load_dataset(name: str = "synthetic"):
    """Returns (volume: np.ndarray HU, spacing: (float, float, float))."""
    if name == "synthetic":
        return build_phantom(), (1.0, 1.0, 1.0)
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, choices: synthetic, {', '.join(DATASETS)}")
    path = _ensure_downloaded(name)
    return _load_nrrd(path)


def list_datasets() -> list:
    return ["synthetic"] + list(DATASETS.keys())
