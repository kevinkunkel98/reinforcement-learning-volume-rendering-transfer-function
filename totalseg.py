"""TotalSegmentator subset used by RL v2: the committed manifest written by
tools/select_totalseg.py, plus loading one selected CT volume.

Volumes are returned in canonical RAS axis order (numpy axis 0 -> patient
right, 1 -> anterior, 2 -> superior) as float32 Hounsfield units.
"""
import json
import os

import nibabel as nib
import numpy as np

MANIFEST_PATH = "data/totalseg_manifest.json"
NAME_PREFIX = "ts_"
SPLITS = ("train", "val", "test")
_FETCH_HINT = "run `python -m tools.select_totalseg` to extract the volumes"


def _subjects() -> dict:
    # Resolved per call (not cached) so tests can point MANIFEST_PATH elsewhere;
    # the file is small.
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return {s["name"]: s for s in json.load(f)["subjects"]}


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
    names = sorted(n for n, s in _subjects().items() if s["split"] == split)
    missing = [n for n in names if not os.path.exists(subject(n)["path"])]
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
