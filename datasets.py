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
import functools
import hashlib
import math
import os
import sys
import urllib.request

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from phantom import build_phantom
from transfer import CENTER_RANGE

DATA_DIR = "data"
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024

DATASETS = {
    "ct_chest": {
        "filename": "CT-chest.nrrd",
        "url": (
            "https://github.com/Slicer/SlicerTestingData/releases/download/"
            "SHA256/4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"
        ),
        "sha256": "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e",
        # This scan's own axis convention puts the body's long axis on the
        # default camera's view direction, so azimuth/elevation=(0,0) looks
        # straight down the body like a stack of axial slices, not at the
        # torso. elevation=80 (not 90 -- VTK's Azimuth/Elevation hits gimbal
        # lock exactly at 90, warns and resets the up-vector unpredictably)
        # tilts to a natural torso-facing view. Found empirically, not
        # derived from the NRRD header.
        "default_camera": {"azimuth": 0.0, "elevation": 80.0, "zoom": 1.0},
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
        # The global DEFAULT_CAMERA (camera.DEFAULT_CAMERA, azimuth=30/
        # elevation=20) is a side profile for this scan's axis convention,
        # not a face-forward view. Found empirically by sweeping azimuth.
        "default_camera": {"azimuth": 280.0, "elevation": 0.0, "zoom": 1.0},
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


@functools.lru_cache(maxsize=None)
def load_dataset(name: str = "synthetic"):
    """Returns (volume: np.ndarray HU, spacing: (float, float, float)).

    Cached per name: the viewer's chunked-transfer endpoints
    (dataset_metadata/get_volume_chunk in server.py) call this once per
    HTTP request, and a real CT/MRI volume re-reads + re-parses an NRRD
    file from disk plus a flip/rescale pass every call. Serving a real
    dataset's ~18 chunks uncached took ~6.5s of server time alone (each
    chunk request reloading the whole volume from scratch); no caller
    anywhere mutates the returned array in place, so caching by name is
    safe.
    """
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


def default_camera_for(name: str) -> dict:
    """Starting camera for `name`, overridden per-dataset where the shared
    camera.DEFAULT_CAMERA doesn't give a natural, face-forward framing for
    that scan's own axis convention (see DATASETS entries). Always returns
    a fresh dict -- safe to mutate."""
    from camera import DEFAULT_CAMERA
    override = DATASETS.get(name, {}).get("default_camera")
    return dict(override) if override else dict(DEFAULT_CAMERA)


def _dataset_version(name: str) -> str:
    if name == "synthetic":
        return "synthetic-v1"
    return f"sha256:{DATASETS[name]['sha256']}"


def _validate_chunk_bytes(chunk_bytes: int) -> None:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if chunk_bytes % np.dtype(np.float32).itemsize:
        raise ValueError("chunk_bytes must be a multiple of float32 item size")


def _validated_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.dtype.kind != "f" or volume.dtype.itemsize != 4:
        raise ValueError("volume must have dtype float32")
    if volume.dtype.byteorder == ">" or (
        volume.dtype.byteorder == "=" and sys.byteorder != "little"
    ):
        raise ValueError("volume must use little-endian float32")
    if volume.ndim != 3:
        raise ValueError("volume must have three dimensions")
    if volume.size == 0 or not np.isfinite(volume).all():
        raise ValueError("volume values must be finite and non-empty")
    return np.asfortranarray(volume, dtype=np.dtype("<f4"))


def iter_volume_chunks(volume: np.ndarray, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
    """Yield little-endian float32 chunks in render.py's Fortran order."""
    _validate_chunk_bytes(chunk_bytes)
    volume = _validated_volume(volume)
    raw = memoryview(volume.ravel(order="F")).cast("B")
    for offset in range(0, raw.nbytes, chunk_bytes):
        yield raw[offset:offset + chunk_bytes].tobytes()


def dataset_metadata(name: str) -> dict:
    """Return transport metadata for the normalized volume loaded by ``name``."""
    if name not in list_datasets():
        raise ValueError(f"unknown dataset {name!r}, choices: synthetic, {', '.join(DATASETS)}")
    volume, spacing = load_dataset(name)
    volume = _validated_volume(volume)
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != 3 or not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("spacing must contain three positive finite values")

    total_bytes = volume.nbytes
    chunk_count = math.ceil(total_bytes / DEFAULT_CHUNK_BYTES)
    descriptors = []
    for index in range(chunk_count):
        offset = index * DEFAULT_CHUNK_BYTES
        descriptors.append({
            "index": index,
            "byte_offset": offset,
            "byte_length": min(DEFAULT_CHUNK_BYTES, total_bytes - offset),
        })
    return {
        "name": name,
        "version": _dataset_version(name),
        "dimensions": list(volume.shape),
        "spacing": list(spacing),
        "scalar_type": "float32",
        "byte_order": "little",
        "order": "F",
        "axis_mapping": "numpy axis0 -> VTK X; axis1 -> VTK Y; axis2 -> VTK Z",
        "orientation": "dataset-normalized",
        "intensity_range": [float(volume.min()), float(volume.max())],
        "chunk_bytes": DEFAULT_CHUNK_BYTES,
        "total_bytes": total_bytes,
        "chunk_count": chunk_count,
        "chunks": descriptors,
    }


def get_volume_chunk(name: str, index: int) -> bytes:
    """Return one normalized volume chunk by dataset name and zero-based index."""
    if name not in list_datasets():
        raise ValueError(f"unknown dataset {name!r}, choices: synthetic, {', '.join(DATASETS)}")
    if isinstance(index, bool) or not isinstance(index, int):
        raise IndexError("chunk index must be an integer")
    volume, _ = load_dataset(name)
    volume = _validated_volume(volume)
    chunk_count = math.ceil(volume.nbytes / DEFAULT_CHUNK_BYTES)
    if index < 0 or index >= chunk_count:
        raise IndexError(f"chunk index out of range: {index}")
    offset = index * DEFAULT_CHUNK_BYTES
    raw = memoryview(volume.ravel(order="F")).cast("B")
    return raw[offset:offset + DEFAULT_CHUNK_BYTES].tobytes()
