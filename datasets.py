"""Real CT/MRI volumes, loaded as (numpy array, spacing) alongside the
synthetic phantom. Downloaded once, checksum-verified, and cached under
data/ -- same "download once, cache forever" precedent as the Whisper model
in asr.py.

Most entries are de-identified public test data from the 3D Slicer project
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

Two entries are classic volume-rendering benchmark datasets rather than
Slicer test data: `stag_beetle` (TU Wien industrial CT, 2005 -- a standard
reference in transfer-function research) and `ct_head` (the Stanford/UNC
"CThead" dataset, originally from the 1987-89 Marching Cubes-era UNC
scans -- probably the single most-cited volume dataset in the field's
history). Both ship as raw scanner-unit integers with no calibrated HU
scale, same situation as the MRI entry above, so they get the same
`_rescale_intensity_to_hu_range` treatment (marked via `modality:
"uncalibrated"` instead of `"mri"` -- same rescale, different reason: not
because the modality lacks a HU concept, but because these particular
scanner dumps were never calibrated to one).
"""
import functools
import hashlib
import math
import os
import re
import struct
import sys
import tarfile
import urllib.request
import zipfile

import nibabel as nib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy  # pyright: ignore[reportMissingImports]

from phantom import build_phantom
import totalseg
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
    "stag_beetle": {
        "filename": "dataset-stagbeetle-208x208x123.zip",
        "url": (
            "https://www.cg.tuwien.ac.at/research/publications/2005/"
            "dataset-stagbeetle/dataset-stagbeetle-208x208x123.zip"
        ),
        "sha256": "f41bf432fc2bb1f5678171f4924451f903127193ab264008d8d8b84134090bd8",
        "format": "stagbeetle_zip",
        "modality": "uncalibrated",  # see module docstring -- industrial CT, no HU calibration
        "spacing": (1.0, 1.0, 1.0),  # isotropic per the dataset's own documentation
    },
    "ct_head": {
        "filename": "CThead.tar.gz",
        "url": "https://graphics.stanford.edu/data/voldata/CThead.tar.gz",
        "sha256": "b176037ed1bde45eaff3b94fce1e02037680c515009df5bf00747210b449bbfa",
        "format": "cthead_tar",
        "modality": "uncalibrated",  # see module docstring -- 1980s scanner dump, no HU calibration
        "spacing": (1.0, 1.0, 2.0),  # per the archive's own info file: X:Y:Z voxel aspect is 1:1:2
    },
}

# RL v2 evaluates generalization on scans from another source: the four
# calibrated Slicer CTs, loaded with load_dataset(name, canonical=True).
OUT_OF_SOURCE_CT = ("ct_chest", "ct_skull", "ct_cardio", "ct_abdomen")

# TotalSegmentator volumes are RAS; this chat-UI camera shows them from the
# front with superior up (checked by rendering one, 2026-09-15).
TOTALSEG_DEFAULT_CAMERA = {"azimuth": 0.0, "elevation": 80.0, "zoom": 1.0}


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


# Maps an NRRD "space" to the 3x3 matrix converting its world axes to RAS.
_NRRD_SPACE_TO_RAS = {
    "left-posterior-superior": np.diag([-1.0, -1.0, 1.0]),
    "right-anterior-superior": np.eye(3),
}


def _read_nrrd_header(path: str) -> dict:
    """Header fields of an NRRD file (text lines up to the first blank line)."""
    fields = {}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").rstrip("\r\n")
            if not line:
                break
            if line.startswith("#") or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            fields[key.strip()] = value.strip()
    return fields


def _reorient_nrrd_to_ras(volume: np.ndarray, header: dict):
    """Reorder/flip a volume loaded by _load_nrrd (NRRD axis order) into RAS
    axis order using the header's space directions. Oblique directions snap to
    the closest RAS axes (CT-brain is off by a few degrees). Returns (volume,
    spacing) with spacing reordered to match."""
    space = header.get("space")
    if space not in _NRRD_SPACE_TO_RAS:
        raise ValueError(f"unsupported NRRD space {space!r}")
    vectors = re.findall(r"\(([^)]*)\)", header.get("space directions", ""))
    if len(vectors) != 3:
        raise ValueError("NRRD header needs three space direction vectors")
    columns = np.array([[float(v) for v in vector.split(",")] for vector in vectors]).T
    directions = _NRRD_SPACE_TO_RAS[space] @ columns     # column i: world direction of axis i
    affine = np.eye(4)
    affine[:3, :3] = directions
    orientation = nib.orientations.io_orientation(affine)
    reoriented = nib.orientations.apply_orientation(volume, orientation)
    zooms = np.linalg.norm(directions, axis=0)
    spacing = [0.0, 0.0, 0.0]
    for axis, (target, _) in enumerate(orientation):
        spacing[int(target)] = float(zooms[axis])
    return np.ascontiguousarray(reoriented, dtype=np.float32), tuple(spacing)


def _load_stagbeetle_zip(path: str) -> np.ndarray:
    """Loads the TU Wien Stag Beetle CT dataset: a zip containing one .dat
    file whose first 6 bytes are three little-endian uint16 dimensions
    (width, height, depth), followed by raw little-endian uint16 voxel
    data, X-fastest. Confirmed empirically against the known 208x208x123
    dataset -- those exact header bytes decode to that shape."""
    with zipfile.ZipFile(path) as zf:
        dat_name = next(n for n in zf.namelist() if n.endswith(".dat"))
        raw_bytes = zf.read(dat_name)
    width, height, depth = struct.unpack("<HHH", raw_bytes[:6])
    arr = np.frombuffer(raw_bytes[6:], dtype="<u2").reshape((depth, height, width))
    return np.transpose(arr.astype(np.float32), (2, 1, 0))  # -> (X, Y, Z), matching _load_nrrd


def _load_cthead_tar(path: str) -> np.ndarray:
    """Loads the classic Stanford/UNC CThead dataset: a gzipped tar of 113
    per-slice files (CThead.1 .. CThead.113, non-lexicographic numeric
    order), each a headerless 256x256 big-endian ("Mac byte ordering")
    int16 raster. Confirmed empirically: interpreting as big-endian gives a
    plausible intensity range (4-1457); little-endian gives full-range
    int16 noise (-32768-32515)."""
    with tarfile.open(path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.name.startswith("CThead.")}
        ordered_names = sorted(members, key=lambda n: int(n.split(".")[1]))
        slices = [
            np.frombuffer(tar.extractfile(members[n]).read(), dtype=">i2").reshape(256, 256)
            for n in ordered_names
        ]
    arr = np.stack(slices, axis=0)  # (depth, height, width)
    return np.transpose(arr.astype(np.float32), (2, 1, 0))  # -> (X, Y, Z), matching _load_nrrd


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


@functools.lru_cache(maxsize=4)
def _load_dataset_cached(name: str, canonical: bool):
    if name == "synthetic":
        return build_phantom(), (1.0, 1.0, 1.0)
    if totalseg.is_totalseg(name):
        return totalseg.load_volume(name)
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, choices: {', '.join(list_datasets())}")
    path = _ensure_downloaded(name)
    fmt = DATASETS[name].get("format", "nrrd")
    if canonical and (fmt != "nrrd" or DATASETS[name].get("modality") in ("mri", "uncalibrated")):
        raise ValueError(f"{name!r} has no canonical RAS orientation (only calibrated CT volumes do)")
    if fmt == "nrrd":
        volume, spacing = _load_nrrd(path)
    elif fmt == "stagbeetle_zip":
        volume, spacing = _load_stagbeetle_zip(path), DATASETS[name]["spacing"]
    elif fmt == "cthead_tar":
        volume, spacing = _load_cthead_tar(path), DATASETS[name]["spacing"]
    else:
        raise ValueError(f"unknown dataset format {fmt!r} for {name!r}")
    if canonical:
        return _reorient_nrrd_to_ras(volume, _read_nrrd_header(path))
    if "flip_axis" in DATASETS[name]:
        volume = np.flip(volume, axis=DATASETS[name]["flip_axis"])
    if DATASETS[name].get("modality") in ("mri", "uncalibrated"):
        volume = _rescale_intensity_to_hu_range(volume)
    return volume, spacing


def load_dataset(name: str = "synthetic", canonical: bool = False):
    """Returns (volume: np.ndarray HU, spacing: (float, float, float)).

    canonical=True returns calibrated CT volumes in RAS axis order (numpy
    axis 0 -> patient right, 1 -> anterior, 2 -> superior), which RL v2
    relies on. The default keeps each file's own axis order, which the chat
    UI's per-dataset cameras were tuned for. TotalSegmentator volumes are
    always RAS; the synthetic phantom is treated as RAS.

    Cached (in `_load_dataset_cached`) on (name, canonical), with `canonical`
    normalized to False first for volumes that are RAS either way (synthetic,
    TotalSegmentator) so load_dataset(n), load_dataset(n, False) and
    load_dataset(n, canonical=True) share one entry instead of three: the
    viewer's chunked-transfer endpoints (dataset_metadata/get_volume_chunk in
    server.py) call this once per HTTP request, and re-parsing an NRRD file
    from disk plus a flip/rescale pass on every chunk request of one dataset
    cost ~6.5s of server time uncached. Bounded to 4 entries because a real
    volume held as float32 is large -- the four Slicer CTs alone range
    38-337 MB (ct_cardio, the biggest) and the ~30 TotalSegmentator volumes
    are 38-239 MB (median ~109 MB), so 4 entries can hold up to ~1.1 GB. No
    caller mutates the returned array in place, so caching is safe.
    """
    if name == "synthetic" or totalseg.is_totalseg(name):
        canonical = False        # already RAS: one cache entry serves both calls
    return _load_dataset_cached(name, bool(canonical))


load_dataset.cache_clear = _load_dataset_cached.cache_clear  # pyright: ignore[reportFunctionMemberAccess]
load_dataset.cache_info = _load_dataset_cached.cache_info  # pyright: ignore[reportFunctionMemberAccess]


def list_datasets() -> list:
    return ["synthetic"] + list(DATASETS.keys()) + totalseg.available_names()


def default_camera_for(name: str) -> dict:
    """Starting camera for `name`, overridden per-dataset where the shared
    camera.DEFAULT_CAMERA doesn't give a natural, face-forward framing for
    that scan's own axis convention (see DATASETS entries). Always returns
    a fresh dict -- safe to mutate."""
    from camera import DEFAULT_CAMERA
    if totalseg.is_totalseg(name):
        return dict(TOTALSEG_DEFAULT_CAMERA)
    override = DATASETS.get(name, {}).get("default_camera")
    return dict(override) if override else dict(DEFAULT_CAMERA)


def volumes_for_split(split: str) -> list:
    """RL v2 volume names: TotalSegmentator "train" / "val" / "test" subjects,
    or the Slicer CTs as the "out_of_source" test set."""
    if split == "out_of_source":
        return list(OUT_OF_SOURCE_CT)
    return totalseg.split_names(split)


def _dataset_version(name: str) -> str:
    if name == "synthetic":
        return "synthetic-v1"
    if totalseg.is_totalseg(name):
        return totalseg.version(name)
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
        raise ValueError(f"unknown dataset {name!r}, choices: {', '.join(list_datasets())}")
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
        "orientation": "RAS" if totalseg.is_totalseg(name) else "dataset-normalized",
        "intensity_range": [float(volume.min()), float(volume.max())],
        "chunk_bytes": DEFAULT_CHUNK_BYTES,
        "total_bytes": total_bytes,
        "chunk_count": chunk_count,
        "chunks": descriptors,
    }


def get_volume_chunk(name: str, index: int) -> bytes:
    """Return one normalized volume chunk by dataset name and zero-based index."""
    if name not in list_datasets():
        raise ValueError(f"unknown dataset {name!r}, choices: {', '.join(list_datasets())}")
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
