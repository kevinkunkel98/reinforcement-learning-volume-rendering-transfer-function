"""Real CT/MRI volumes, loaded as (numpy array, spacing) alongside the
synthetic phantom. Downloaded once, checksum-verified, and cached under
data/ -- same "download once, cache forever" precedent as the Whisper model
in asr.py.

All entries are de-identified public test data from the 3D Slicer project
(https://github.com/Slicer/SlicerTestingData), each verified by hand against
its registration in Slicer's own SampleData.py before adding it here. The
CT entries carry real calibrated Hounsfield units (confirmed via the
standard -1024/-3024 air-padding value and realistic bone/soft-tissue HU
distributions). The MRI entry does NOT -- MRI has no calibrated absolute
intensity scale, and its own brightness ordering differs from CT's (T1 bone/
skull is *dark*, not the brightest structure as in CT). Its raw intensities
are linearly rescaled onto the same numeric range transfer.py's tissue bands
already use (see `_rescale_intensity_to_hu_range`), so every existing
command mechanically works and the render looks correct -- but the tissue
*labels* ("bone", "fat", ...) are not radiologically accurate for this
dataset, since that would require real MRI tissue segmentation, not a
one-line rescale. This is a known, deliberate approximation, not a bug.
"""
import hashlib
import os
import urllib.request

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from phantom import build_phantom
from transfer import CENTER_RANGE

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
    "mri_head": {
        "filename": "MR-head.nrrd",
        "url": (
            "https://github.com/Slicer/SlicerTestingData/releases/download/"
            "SHA256/cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93"
        ),
        "sha256": "cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93",
        "modality": "mri",  # not Hounsfield units -- see module docstring and _rescale_intensity_to_hu_range
        # This file's own NRRD "space directions" declare a different slice
        # acquisition axis than the CT files (sagittal vs. the CT datasets'
        # axial), so it loads upside down with no reorientation. Determined
        # empirically (rendered all three single-axis flips and compared),
        # not derived from the header math alone -- that math was tried
        # first and got the axis wrong twice. No axis permutation needed,
        # spacing is unaffected: just this one axis reversed.
        "flip_axis": 1,
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


def _rescale_intensity_to_hu_range(arr: np.ndarray, lo_percentile: float = 0.5,
                                    hi_percentile: float = 99.5) -> np.ndarray:
    """Linearly rescales an arbitrary-unit intensity volume (e.g. MRI, which
    has no calibrated Hounsfield-unit scale) onto transfer.py's full HU
    range, so the existing tissue bands/commands have something meaningful
    to act on mechanically. Percentile (not min/max) endpoints avoid a few
    outlier voxels compressing the useful dynamic range. This is a visual/
    mechanical approximation, not real tissue segmentation -- see this
    module's docstring."""
    lo, hi = np.percentile(arr, [lo_percentile, hi_percentile])
    normalized = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * (CENTER_RANGE[1] - CENTER_RANGE[0]) + CENTER_RANGE[0]).astype(np.float32)


def load_dataset(name: str = "synthetic"):
    """Returns (volume: np.ndarray HU, spacing: (float, float, float))."""
    if name == "synthetic":
        return build_phantom(), (1.0, 1.0, 1.0)
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, choices: synthetic, {', '.join(DATASETS)}")
    path = _ensure_downloaded(name)
    volume, spacing = _load_nrrd(path)
    if "flip_axis" in DATASETS[name]:
        volume = np.flip(volume, axis=DATASETS[name]["flip_axis"])
    if DATASETS[name].get("modality") == "mri":
        volume = _rescale_intensity_to_hu_range(volume)
    return volume, spacing


def list_datasets() -> list:
    return ["synthetic"] + list(DATASETS.keys())
