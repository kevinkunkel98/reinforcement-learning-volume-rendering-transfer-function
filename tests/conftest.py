"""Shared test data: two small synthetic patients, so no test needs the
downloaded scans.

Every test runs in one scratch working directory holding data/ (the synthetic
scans, their manifest and the demo list, plus links to the repository's
lexicon and eval sets), models/ and static/ (links), and out/ (caches shared
across tests, so each visibility model is built once).
"""
import gzip
import hashlib
import json
import pathlib

import nibabel as nib
import numpy as np
import pytest

from anatomy import CANONICAL_CLASSES, CLASS_LAYOUT_VERSION

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHAPE = (40, 32, 48)
SPACING = (4.0, 4.0, 4.0)
# (class, HU, centre in voxels, radii in voxels)
ORGANS = (
    ("lungs", -800.0, (12, 16, 32), (7, 7, 9)), ("lungs", -800.0, (28, 16, 32), (7, 7, 9)),
    ("heart", 50.0, (22, 13, 28), (5, 4, 4)),
    ("liver", 65.0, (13, 15, 18), (7, 6, 5)),
    ("spleen", 55.0, (29, 18, 19), (3, 3, 3)),
    ("kidneys", 45.0, (14, 22, 12), (3, 3, 4)), ("kidneys", 45.0, (26, 22, 12), (3, 3, 4)),
    ("vessels", 300.0, (20, 19, 22), (2, 2, 16)),
    ("skeleton", 900.0, (20, 25, 24), (2, 2, 20)),
    ("soft", 40.0, (20, 28, 24), (8, 2, 14)),
)
SCANS = {"ts_s9001": ("s9001", "test", True), "ts_s9002": ("s9002", "train", False)}


def _phantom(seed: int):
    """A body-shaped volume: an unlabelled soft shell around labelled organs."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij"), -1).astype(np.float32)
    volume = np.full(SHAPE, -1000.0, dtype=np.float32)
    labels = np.zeros(SHAPE, dtype=np.uint8)
    body = (((grid - [20, 16, 24]) / [17, 13, 23]) ** 2).sum(-1) <= 1.0
    volume[body] = -100.0
    for name, hu, centre, radii in ORGANS:
        inside = (((grid - centre) / radii) ** 2).sum(-1) <= 1.0
        volume[inside] = hu
        labels[inside] = CANONICAL_CLASSES.index(name) + 1
    volume += rng.normal(0.0, 5.0, SHAPE).astype(np.float32)
    return volume, labels


def _write_nifti(array, path):
    affine = np.diag([*SPACING, 1.0])
    path.write_bytes(gzip.compress(nib.Nifti1Image(array, affine).to_bytes()))


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("work")
    data = root / "data"
    (data / "totalseg").mkdir(parents=True)
    for name in ("lexicon.json", "parser_eval_set.json", "parser_eval_realistic.json"):
        (data / name).symlink_to(ROOT / "data" / name)
    for name in ("models", "static"):
        (root / name).symlink_to(ROOT / name)
    subjects = []
    for seed, (name, (sid, split, contrast)) in enumerate(SCANS.items()):
        volume, labels = _phantom(seed)
        folder = data / "totalseg" / sid
        folder.mkdir()
        _write_nifti(volume, folder / "ct.nii.gz")
        _write_nifti(labels, folder / "labels.nii.gz")
        subjects.append({
            "id": sid, "name": name, "split": split, "label_layout_version": CLASS_LAYOUT_VERSION,
            "study_type": "ct thorax-abdomen-pelvis", "region": "trunk", "shape": list(SHAPE),
            "spacing": list(SPACING), "path": f"data/totalseg/{sid}/ct.nii.gz",
            "sha256": hashlib.sha256((folder / "ct.nii.gz").read_bytes()).hexdigest(),
            "labels_path": f"data/totalseg/{sid}/labels.nii.gz",
            "classes_present": sorted(CANONICAL_CLASSES), "contrast": contrast})
    (data / "totalseg_manifest.json").write_text(json.dumps(
        {"label_layout_version": CLASS_LAYOUT_VERSION, "subjects": subjects}))
    (data / "demo_scans.json").write_text(json.dumps({"scans": list(SCANS), "candidates": list(SCANS)}))
    return root


@pytest.fixture(autouse=True)
def _in_workdir(workdir, monkeypatch):
    monkeypatch.chdir(workdir)


@pytest.fixture
def model():
    import visibility
    return visibility.for_volume("ts_s9001")


@pytest.fixture(scope="session")
def policy():
    from rl.model_based import ModelBasedPolicy
    return ModelBasedPolicy.load(str(ROOT / "models/v3c_seed1/checkpoint.pt"))
