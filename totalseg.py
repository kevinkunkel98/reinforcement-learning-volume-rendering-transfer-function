"""TotalSegmentator subset used by RL v2: the committed manifest written by
tools/select_totalseg.py, plus loading one selected CT volume.

Volumes are returned in canonical RAS axis order (numpy axis 0 -> patient
right, 1 -> anterior, 2 -> superior) as float32 Hounsfield units.
"""
import json
import os

import nibabel as nib
import numpy as np

from anatomy import CANONICAL_CLASSES, CLASS_LAYOUT_VERSION

MANIFEST_PATH = "data/totalseg_manifest.json"
NAME_PREFIX = "ts_"
SPLITS = ("train", "val", "test")
_FETCH_HINT = "run `python -m tools.select_totalseg` to extract the volumes"
LABEL_CHUNK_BYTES = 8 * 1024 * 1024


def _manifest() -> dict:
    # Resolved per call (not cached) so tests can point MANIFEST_PATH elsewhere;
    # the file is small.
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _subjects() -> dict:
    return {s["name"]: s for s in _manifest().get("subjects", [])}


def is_totalseg(name: str) -> bool:
    return name.startswith(NAME_PREFIX)


def subject(name: str) -> dict:
    subjects = _subjects()
    if name not in subjects:
        raise ValueError(f"unknown TotalSegmentator volume {name!r}")
    return subjects[name]


def available_names() -> list:
    """Manifest volumes whose CT file exists locally, sorted."""
    return sorted(n for n, s in _subjects().items() if os.path.exists(s["path"]))


def split_names(split: str) -> list:
    """All volumes of a split; raises if any of them is missing locally, so an
    experiment never silently runs on fewer volumes than the manifest says."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, choices: {', '.join(SPLITS)}")
    subjects = _subjects()
    names = sorted(n for n, s in subjects.items() if s["split"] == split)
    missing = [n for n in names if not os.path.exists(subjects[n]["path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} volumes missing (e.g. {missing[0]}); {_FETCH_HINT}")
    return names


def version(name: str) -> str:
    return f"sha256:{subject(name)['sha256']}"


def load_volume(name: str):
    """(float32 HU volume in RAS axis order, (sx, sy, sz) spacing in mm)."""
    entry = subject(name)
    if not os.path.exists(entry["path"]):
        raise FileNotFoundError(f"{entry['path']} missing; {_FETCH_HINT}")
    image = nib.as_closest_canonical(nib.load(entry["path"]))
    volume = image.get_fdata(dtype=np.float32)
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])
    return np.ascontiguousarray(volume), spacing


def has_labels(name: str) -> bool:
    entry = _subjects().get(name)
    if entry and entry.get("labels_path"):
        _validate_label_layout(entry, _manifest_version_for(entry))
    return bool(entry and entry.get("labels_path") and os.path.exists(entry["labels_path"]))


def classes_present(name: str) -> list:
    return list(subject(name).get("classes_present", []))


def is_contrast(name: str) -> bool:
    return bool(subject(name).get("contrast", False))


def load_labels(name: str) -> np.ndarray:
    """Anatomical class ids per voxel, canonical RAS, matching load_volume()."""
    entry = subject(name)
    path = entry.get("labels_path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{name} has no label volume; {_FETCH_HINT}")
    _validate_label_layout(entry, _manifest_version_for(entry))
    image = nib.as_closest_canonical(nib.load(path))
    labels = np.asarray(image.dataobj)
    if labels.dtype != np.dtype(np.uint8):
        raise ValueError("label volume must have dtype uint8")
    if labels.ndim != 3 or labels.size == 0:
        raise ValueError("label volume must be a non-empty three-dimensional array")
    if not np.isfinite(labels).all():
        raise ValueError("label volume values must be finite")
    return np.ascontiguousarray(labels)


def _validate_label_layout(entry: dict, manifest_version: str | None = None) -> None:
    """Reject label ids produced by a different anatomy registry."""
    version = entry.get("label_layout_version")
    if version != CLASS_LAYOUT_VERSION or manifest_version != CLASS_LAYOUT_VERSION:
        raise ValueError(
            "label_layout_version mismatch: "
            f"expected {CLASS_LAYOUT_VERSION!r}, got subject={version!r}, "
            f"manifest={manifest_version!r}; "
            "regenerate labels with `python -m tools.select_totalseg`"
        )


def _manifest_version_for(entry: dict) -> str | None:
    """Read root metadata, allowing tests to inject subject-only registries."""
    manifest = _manifest()
    if os.path.exists(MANIFEST_PATH):
        return manifest.get("label_layout_version")
    return entry.get("label_layout_version")


def _validated_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype != np.dtype(np.uint8):
        raise ValueError("labels must have dtype uint8")
    if labels.ndim != 3:
        raise ValueError("labels must have three dimensions")
    if labels.size == 0:
        raise ValueError("labels must be non-empty")
    return np.asfortranarray(labels, dtype=np.uint8)


def iter_label_chunks(labels: np.ndarray, chunk_bytes: int = LABEL_CHUNK_BYTES):
    """Yield uint8 label bytes in the same Fortran order as volume transport."""
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    labels = _validated_labels(labels)
    raw = memoryview(labels.ravel(order="F")).cast("B")
    for offset in range(0, raw.nbytes, chunk_bytes):
        yield raw[offset:offset + chunk_bytes].tobytes()


def label_metadata(name: str) -> dict:
    """Return validated label layout metadata for one labeled subject."""
    labels = _validated_labels(load_labels(name))
    present = classes_present(name)
    class_ids = {class_name: index for index, class_name in enumerate(CANONICAL_CLASSES, 1)
                 if class_name in present}
    total_bytes = labels.nbytes
    chunks = [{
        "index": index,
        "byte_offset": index * LABEL_CHUNK_BYTES,
        "byte_length": min(LABEL_CHUNK_BYTES, total_bytes - index * LABEL_CHUNK_BYTES),
    } for index in range((total_bytes + LABEL_CHUNK_BYTES - 1) // LABEL_CHUNK_BYTES)]
    return {
        "name": name,
        "dataset_version": version(name),
        "dimensions": list(labels.shape),
        "scalar_type": "uint8",
        "byte_order": "little",
        "order": "F",
        "label_layout_version": CLASS_LAYOUT_VERSION,
        "class_ids": class_ids,
        "classes": list(class_ids),
        "chunk_bytes": LABEL_CHUNK_BYTES,
        "total_bytes": total_bytes,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
