"""The CT scans: TotalSegmentator patients with eight organ classes labelled.

`data/totalseg_manifest.json` (tracked) lists every patient, its split
(train / train_extra / val / test) and the checksum of its CT;
`tools.fetch_scans` downloads them to data/totalseg/<id>/. Volumes load in
canonical RAS axis order (numpy axis 0 -> patient right, 1 -> anterior,
2 -> superior) as float32 Hounsfield units, labels as uint8 class ids
(0 = unlabelled, i = anatomy.CANONICAL_CLASSES[i - 1]).

The browser gets volumes and labels in Fortran-order chunks
(`volume_metadata`/`volume_chunk`, `label_metadata`/`label_chunk`).
"""
import functools
import hashlib
import json
import math
import os

import nibabel as nib
import numpy as np

from anatomy import CANONICAL_CLASSES, CLASS_LAYOUT_VERSION

MANIFEST_PATH = "data/totalseg_manifest.json"
DEMO_SCANS_PATH = "data/demo_scans.json"
NAME_PREFIX = "ts_"
SPLITS = ("train", "val", "test", "train_extra")
CHUNK_BYTES = 8 * 1024 * 1024
_FETCH_HINT = "run `make scans` (python -m tools.fetch_scans) to download it"

# From the front, superior up, the patient's right on the viewer's left.
# Azimuth 180 first: elevation 80 alone keeps VTK's +y view-up, which then
# points to inferior and draws every scan upside down and mirrored.
DEFAULT_CAMERA = {"azimuth": 180.0, "elevation": 80.0, "zoom": 1.0}


# --- the manifest -----------------------------------------------------------------

def manifest() -> dict:
    # Read per call (the file is small), so tests can point MANIFEST_PATH elsewhere.
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as stream:
        return json.load(stream)


def _subjects() -> dict:
    return {s["name"]: s for s in manifest().get("subjects", [])}


def is_scan(name: str) -> bool:
    return name.startswith(NAME_PREFIX)


def subject(name: str) -> dict:
    subjects = _subjects()
    if name not in subjects:
        raise ValueError(f"unknown scan {name!r}")
    return subjects[name]


def available_names() -> list:
    """Manifest scans downloaded locally, sorted."""
    return sorted(n for n, s in _subjects().items() if os.path.exists(s["path"]))


def split_names(split: str) -> list:
    """Every scan of a split; raises if one is missing, so an experiment never
    silently runs on fewer scans than the manifest says."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, choices: {', '.join(SPLITS)}")
    subjects = _subjects()
    names = sorted(n for n, s in subjects.items() if s["split"] == split)
    missing = [n for n in names if not os.path.exists(subjects[n]["path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} scans missing (e.g. {missing[0]}); {_FETCH_HINT}")
    return names


def demo_names(key: str = "scans") -> list:
    """The viewer's showcase set (data/demo_scans.json), first = start scan;
    key="candidates" gives the patients it was picked from."""
    if not os.path.exists(DEMO_SCANS_PATH):
        return []
    with open(DEMO_SCANS_PATH, encoding="utf-8") as stream:
        return list(json.load(stream).get(key, []))


def version(name: str) -> str:
    return f"sha256:{subject(name)['sha256']}"


def classes_present(name: str) -> list:
    return list(subject(name).get("classes_present", []))


def is_contrast(name: str) -> bool:
    return bool(subject(name).get("contrast", False))


# --- loading ----------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def load_volume(name: str):
    """(float32 HU volume in RAS axis order, (sx, sy, sz) spacing in mm).
    Cached: the browser's chunk requests read it once per request."""
    entry = subject(name)
    if not os.path.exists(entry["path"]):
        raise FileNotFoundError(f"{entry['path']} missing; {_FETCH_HINT}")
    image = nib.as_closest_canonical(nib.load(entry["path"]))
    volume = image.get_fdata(dtype=np.float32)
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])
    return np.ascontiguousarray(volume), spacing


def load_labels(name: str) -> np.ndarray:
    """Class id per voxel, canonical RAS, matching `load_volume`."""
    entry = subject(name)
    path = entry.get("labels_path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{name} has no label volume; {_FETCH_HINT}")
    layout = (entry.get("label_layout_version"), manifest().get("label_layout_version"))
    if layout != (CLASS_LAYOUT_VERSION, CLASS_LAYOUT_VERSION):
        raise ValueError(f"label layout {layout} is not {CLASS_LAYOUT_VERSION!r}; "
                         "regenerate the labels with `python -m tools.select_totalseg`")
    labels = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj)
    if labels.dtype != np.uint8 or labels.ndim != 3 or labels.size == 0:
        raise ValueError(f"{path} is not a non-empty 3-D uint8 label volume")
    return np.ascontiguousarray(labels)


@functools.lru_cache(maxsize=64)
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_cache_identity(name: str) -> str:
    """Identity of a scan's labels for caches built on them (visibility.py)."""
    labels = load_labels(name)
    path = subject(name)["labels_path"]
    stat = os.stat(path)
    class_ids = _class_ids(name)
    classes = ",".join(f"{key}={value}" for key, value in sorted(class_ids.items()))
    return "|".join((CLASS_LAYOUT_VERSION, ",".join(map(str, labels.shape)), classes, version(name),
                     _file_digest(path, stat.st_mtime_ns, stat.st_size)))


# --- transport to the browser ---------------------------------------------------------

def _class_ids(name: str) -> dict:
    present = classes_present(name)
    return {c: index for index, c in enumerate(CANONICAL_CLASSES, 1) if c in present}


def _chunks(total_bytes: int) -> list:
    return [{"index": i, "byte_offset": i * CHUNK_BYTES, "byte_length": min(CHUNK_BYTES, total_bytes - i * CHUNK_BYTES)}
            for i in range(math.ceil(total_bytes / CHUNK_BYTES))]


def _chunk(array: np.ndarray, index: int) -> bytes:
    raw = memoryview(np.asfortranarray(array).ravel(order="F")).cast("B")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < math.ceil(raw.nbytes / CHUNK_BYTES):
        raise IndexError(f"chunk index out of range: {index}")
    return raw[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES].tobytes()


def volume_metadata(name: str) -> dict:
    volume, spacing = load_volume(name)
    return {"name": name, "version": version(name), "dimensions": list(volume.shape),
            "spacing": list(spacing), "scalar_type": "float32", "byte_order": "little", "order": "F",
            "axis_mapping": "numpy axis0 -> VTK X; axis1 -> VTK Y; axis2 -> VTK Z", "orientation": "RAS",
            "intensity_range": [float(volume.min()), float(volume.max())], "chunk_bytes": CHUNK_BYTES,
            "total_bytes": volume.nbytes, "chunk_count": len(_chunks(volume.nbytes)),
            "chunks": _chunks(volume.nbytes)}


def volume_chunk(name: str, index: int) -> bytes:
    return _chunk(load_volume(name)[0].astype("<f4", copy=False), index)


def label_metadata(name: str) -> dict:
    labels = load_labels(name)
    class_ids = _class_ids(name)
    return {"name": name, "dataset_version": version(name), "dimensions": list(labels.shape),
            "scalar_type": "uint8", "byte_order": "little", "order": "F",
            "label_layout_version": CLASS_LAYOUT_VERSION, "class_ids": class_ids, "classes": list(class_ids),
            "chunk_bytes": CHUNK_BYTES, "total_bytes": labels.nbytes, "chunk_count": len(_chunks(labels.nbytes)),
            "chunks": _chunks(labels.nbytes)}


def label_chunk(name: str, index: int) -> bytes:
    return _chunk(load_labels(name), index)
