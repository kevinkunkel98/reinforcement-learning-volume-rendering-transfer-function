# RL v2 Plan 2: Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ~30 TotalSegmentator CT volumes plus the four Slicer CTs available to RL v2 in canonical RAS orientation, split by subject into train/val/test/out-of-source, without changing what the chat UI shows for existing datasets.

**Architecture:** A deterministic selection tool reads the downloaded TotalSegmentator zip, filters and stratifies subjects by body region, extracts only the selected CTs to `data/totalseg/` and writes a committed manifest. A small `totalseg.py` module owns that manifest. `datasets.py` gains TotalSegmentator entries, a `canonical=True` loading mode that reorients the Slicer NRRD CTs to RAS from their headers, and `volumes_for_split()`. A PNG projection tool verifies orientation visually.

**Tech Stack:** Python 3.14 (`.venv`), numpy, nibabel 5.4 (already installed in `.venv`), VTK, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md`, section "Data".

**Facts established before writing this plan (2026-09-15):**
- `data/Totalsegmentator_dataset_small_v201.zip` is downloaded (3.0 GB, sha256 `6c47636d9727cd97048616383365dd5a588d0bbc5723b5eb0fa736ed08480ffc`). It is git-ignored by the existing `data/*.zip` rule.
- Zip layout: `meta.csv` (semicolon-separated, UTF-8 with BOM; columns `image_id;age;gender;institute;study_type;split;manufacturer;scanner_model;kvp;pathology;pathology_location`), and per subject `s<id>/ct.nii.gz` plus `s<id>/segmentations/*.nii.gz` (117 masks). 102 subjects.
- Every CT is already RAS (`nib.aff2axcodes` → `('R','A','S')`), 1.5 mm isotropic, `int16` Hounsfield units. The spec's "slice thickness ≤ 3 mm, in-plane ≥ 256²" filter does not fit resampled data; it is replaced by a physical-extent filter (in-plane ≥ 250 mm, superior–inferior ≥ 150 mm), which 100 of 102 subjects pass.
- Study types map to four regions: thorax (23 subjects), abdomen_pelvis (20), trunk (39), whole_body (20). There are no head-only scans; `ct_skull` covers the head in the out-of-source set.
- The four Slicer NRRDs declare `space: left-posterior-superior` and explicit `space directions` (CT-brain is slightly oblique), so their RAS reorientation is computed from the header.
- Rendering a RAS TotalSegmentator volume with the chat camera `{"azimuth": 0, "elevation": 80, "zoom": 1.0}` shows an anterior view with superior up; that becomes the chat UI default camera for `ts_*` volumes.

**Repo rules:**
- Run everything from the repository root with `.venv/bin/python`.
- Commit messages: plain `git commit -m "..."`, **no** `Co-Authored-By` or other trailers.
- Stage files explicitly by path; never `git add -A` or `git add .`. Leave the untracked `docs/prompts_report.pdf` and `docs/vr-integration.md` alone.
- Branch `rl-v2`.

---

## File map

| File | Responsibility |
|---|---|
| `requirements.txt` | add `nibabel` |
| `.gitignore`, `.dockerignore` | ignore `data/totalseg/` |
| `tools/select_totalseg.py` (new) | region mapping, extent filter, stratified split, zip reading, extraction, manifest writing |
| `tests/test_select_totalseg.py` (new) | pure selection logic + end-to-end run on a tiny fake zip |
| `data/totalseg_manifest.json` (new, committed) | selected subjects, split, shape, spacing, sha256, path |
| `totalseg.py` (new) | read the manifest; names, splits, versions; load one volume (RAS, float32 HU) |
| `tests/test_totalseg.py` (new) | manifest-driven tests on tiny NIfTI files |
| `datasets.py` | TotalSegmentator entries in `list_datasets`/`load_dataset`/`_dataset_version`/`dataset_metadata`/`default_camera_for`; bounded cache; `canonical=True` NRRD→RAS; `volumes_for_split` |
| `tests/test_datasets.py` | integration and NRRD reorientation tests |
| `tools/check_orientation.py` (new) | 3-panel projection PNG per volume for visual orientation checks |
| `tests/test_check_orientation.py` (new) | projection layout test |
| `README.md` | "RL v2 volumes" subsection |

---

### Task 1: Dependency and ignore rules

**Files:**
- Modify: `requirements.txt`, `.gitignore`, `.dockerignore`

- [ ] **Step 1: Add nibabel**

In `requirements.txt`, add a line `nibabel` directly after the line `numpy`.

- [ ] **Step 2: Ignore extracted volumes**

In both `.gitignore` and `.dockerignore`, add the line `data/totalseg/` directly after the line `data/*.tar.gz`.

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -c "import nibabel; print(nibabel.__version__)" && git check-ignore -v data/totalseg/x.nii.gz`
Expected: a version (5.4.x), then a line showing `.gitignore` matched `data/totalseg/`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore .dockerignore
git commit -m "chore(data): add nibabel and ignore extracted TotalSegmentator volumes"
```

---

### Task 2: Selection logic

**Files:**
- Create: `tools/select_totalseg.py`
- Test: `tests/test_select_totalseg.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_select_totalseg.py`:

```python
import pytest

from tools.select_totalseg import (
    SPLIT_COUNTS, passes_extent_filter, region_for_study_type, select_and_split,
)


def test_region_for_study_type_maps_all_known_types():
    assert region_for_study_type("ct thorax-neck") == "thorax"
    assert region_for_study_type("ct angiography abdomen-pelvis") == "abdomen_pelvis"
    assert region_for_study_type("ct neck-thorax-abdomen-pelvis") == "trunk"
    assert region_for_study_type("CT Polytrauma ") == "whole_body"


def test_region_for_study_type_rejects_unknown():
    with pytest.raises(ValueError, match="unknown study type"):
        region_for_study_type("ct knee")


def test_passes_extent_filter_uses_physical_extent():
    assert passes_extent_filter((200, 200, 120), (1.5, 1.5, 1.5))       # 300 x 300 x 180 mm
    assert not passes_extent_filter((140, 200, 120), (1.5, 1.5, 1.5))   # 210 mm in-plane
    assert not passes_extent_filter((200, 200, 90), (1.5, 1.5, 1.5))    # 135 mm superior-inferior


def _candidates(region, n):
    return [{"id": f"{region}{i:02d}", "region": region} for i in range(n)]


def test_select_and_split_meets_counts_per_region():
    candidates = []
    for region in SPLIT_COUNTS:
        candidates += _candidates(region, 12)
    assignment = select_and_split(candidates, seed=0)
    for region, (n_train, n_val, n_test) in SPLIT_COUNTS.items():
        splits = [s for sid, s in assignment.items() if sid.startswith(region)]
        assert splits.count("train") == n_train
        assert splits.count("val") == n_val
        assert splits.count("test") == n_test
    assert sum(1 for s in assignment.values() if s == "train") == 20
    assert sum(1 for s in assignment.values() if s == "val") == 4
    assert sum(1 for s in assignment.values() if s == "test") == 6


def test_select_and_split_is_deterministic_and_seed_dependent():
    candidates = _candidates("thorax", 20)
    counts = {"thorax": (3, 1, 1)}
    first = select_and_split(candidates, counts, seed=0)
    assert first == select_and_split(list(reversed(candidates)), counts, seed=0)
    assert first != select_and_split(candidates, counts, seed=1)


def test_select_and_split_raises_when_region_too_small():
    with pytest.raises(ValueError, match="needs 5"):
        select_and_split(_candidates("thorax", 4), {"thorax": (3, 1, 1)}, seed=0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_select_totalseg.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.select_totalseg'`.

- [ ] **Step 3: Implement the selection logic**

Create `tools/select_totalseg.py`:

```python
"""Select, split and extract a TotalSegmentator subset for RL v2.

Reads the official small-subset zip (Zenodo record 10047263) directly, picks
subjects stratified by body region, extracts only the selected CT volumes to
data/totalseg/<id>/ct.nii.gz and writes data/totalseg_manifest.json.
Deterministic for a given --seed.
"""
import numpy as np

REGION_BY_STUDY_TYPE = {
    "ct thorax": "thorax",
    "ct thorax-neck": "thorax",
    "ct abdomen": "abdomen_pelvis",
    "ct pelvis": "abdomen_pelvis",
    "ct hip right": "abdomen_pelvis",
    "ct angiography abdomen-pelvis": "abdomen_pelvis",
    "ct thorax-abdomen-pelvis": "trunk",
    "ct neck-thorax-abdomen-pelvis": "trunk",
    "ct angiography thorax-abdomen-pelvis": "trunk",
    "ct polytrauma": "whole_body",
    "ct whole body": "whole_body",
}

# (train, val, test) subjects per region; totals 20 / 4 / 6.
SPLIT_COUNTS = {
    "whole_body": (5, 1, 2),
    "trunk": (5, 1, 2),
    "thorax": (5, 1, 1),
    "abdomen_pelvis": (5, 1, 1),
}

# The dataset is resampled to 1.5 mm isotropic, so the filter is on physical
# extent: it drops heavily cropped scans (e.g. a single hip).
MIN_INPLANE_MM = 250.0
MIN_SUPERIOR_INFERIOR_MM = 150.0


def region_for_study_type(study_type: str) -> str:
    try:
        return REGION_BY_STUDY_TYPE[study_type.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown study type {study_type!r}") from None


def passes_extent_filter(shape, zooms) -> bool:
    extent = [float(n) * float(z) for n, z in zip(shape[:3], zooms[:3])]
    return min(extent[0], extent[1]) >= MIN_INPLANE_MM and extent[2] >= MIN_SUPERIOR_INFERIOR_MM


def select_and_split(candidates, split_counts=SPLIT_COUNTS, seed=0) -> dict:
    """Map subject id -> "train" | "val" | "test".

    `candidates` is a list of {"id", "region"} dicts. Each region contributes
    exactly its (train, val, test) counts, drawn with a seeded permutation of
    the region's sorted ids, so the result does not depend on input order.
    """
    rng = np.random.default_rng(seed)
    by_region = {}
    for candidate in candidates:
        by_region.setdefault(candidate["region"], []).append(candidate["id"])
    assignment = {}
    for region, (n_train, n_val, n_test) in sorted(split_counts.items()):
        ids = sorted(by_region.get(region, []))
        need = n_train + n_val + n_test
        if len(ids) < need:
            raise ValueError(f"region {region!r} has {len(ids)} candidates, needs {need}")
        picked = [ids[i] for i in rng.permutation(len(ids))[:need]]
        for sid in picked[:n_test]:
            assignment[sid] = "test"
        for sid in picked[n_test:n_test + n_val]:
            assignment[sid] = "val"
        for sid in picked[n_test + n_val:]:
            assignment[sid] = "train"
    return assignment
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_select_totalseg.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/select_totalseg.py tests/test_select_totalseg.py
git commit -m "feat(data): stratified TotalSegmentator subject selection"
```

---

### Task 3: Selection tool I/O, real run, manifest

**Files:**
- Modify: `tools/select_totalseg.py`
- Test: `tests/test_select_totalseg.py`
- Create (by running the tool): `data/totalseg_manifest.json`

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/test_select_totalseg.py`:

```python
import gzip
import json
import zipfile

import nibabel as nib
import numpy as np

from tools.select_totalseg import build_manifest

META_HEADER = "image_id;age;gender;institute;study_type;split;manufacturer;scanner_model;kvp;pathology;pathology_location"


def _nifti_gz(shape, zoom):
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.diag([zoom, zoom, zoom, 1.0]))
    return gzip.compress(image.to_bytes())


def _fake_zip(path):
    rows = [
        ("s0001", "ct thorax", (200, 200, 120)),
        ("s0002", "ct thorax-neck", (200, 200, 120)),
        ("s0003", "ct thorax", (200, 200, 120)),
        ("s0004", "ct thorax", (100, 100, 120)),   # too small in-plane: filtered out
        ("s0005", "ct pelvis", (200, 200, 120)),
    ]
    with zipfile.ZipFile(path, "w") as zf:
        meta = [META_HEADER] + [f"{sid};50;f;A;{st};train;x;y;120;none;none" for sid, st, _ in rows]
        zf.writestr("meta.csv", "﻿" + "\n".join(meta) + "\n")
        for sid, _, shape in rows:
            zf.writestr(f"{sid}/ct.nii.gz", _nifti_gz(shape, 1.5))
            zf.writestr(f"{sid}/segmentations/liver.nii.gz", b"not read")


def test_build_manifest_selects_extracts_and_describes(tmp_path):
    zip_path = tmp_path / "subset.zip"
    _fake_zip(zip_path)
    out_dir = tmp_path / "totalseg"

    manifest = build_manifest(str(zip_path), str(out_dir), seed=0,
                              split_counts={"thorax": (1, 1, 1)})

    subjects = manifest["subjects"]
    assert sorted(s["id"] for s in subjects) == ["s0001", "s0002", "s0003"]
    assert sorted(s["split"] for s in subjects) == ["test", "train", "val"]
    first = subjects[0]
    assert first["name"] == f"ts_{first['id']}"
    assert first["region"] == "thorax"
    assert first["shape"] == [200, 200, 120]
    assert first["spacing"] == [1.5, 1.5, 1.5]
    assert (out_dir / first["id"] / "ct.nii.gz").exists()
    assert first["path"] == str(out_dir / first["id"] / "ct.nii.gz")
    assert len(first["sha256"]) == 64
    assert not (out_dir / "s0005").exists()     # pelvis not in split_counts
    assert manifest["source"]["zenodo_record"] == "10047263"
    assert manifest["selection"]["seed"] == 0
    json.dumps(manifest)                        # serializable
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_select_totalseg.py::test_build_manifest_selects_extracts_and_describes`
Expected: FAIL with `ImportError: cannot import name 'build_manifest'`.

- [ ] **Step 3: Implement zip reading, extraction and the CLI**

In `tools/select_totalseg.py`, replace the import line `import numpy as np` with:

```python
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import zipfile

import nibabel as nib
import numpy as np

DEFAULT_ZIP = "data/Totalsegmentator_dataset_small_v201.zip"
DEFAULT_OUT_DIR = "data/totalseg"
DEFAULT_MANIFEST = "data/totalseg_manifest.json"
ZENODO_RECORD = "10047263"
NIFTI_HEADER_BYTES = 352
```

Append to the end of `tools/select_totalseg.py`:

```python
def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_meta(zf: zipfile.ZipFile) -> list:
    lines = zf.read("meta.csv").decode("utf-8-sig").splitlines()
    header = lines[0].split(";")
    return [dict(zip(header, line.split(";"))) for line in lines[1:] if line.strip()]


def read_ct_header(zf: zipfile.ZipFile, subject_id: str):
    """(shape, zooms) from the NIfTI header only, without decompressing the volume."""
    with zf.open(f"{subject_id}/ct.nii.gz") as raw, gzip.open(raw) as stream:
        header = nib.Nifti1Header.from_fileobj(io.BytesIO(stream.read(NIFTI_HEADER_BYTES)))
    return header.get_data_shape(), header.get_zooms()


def build_manifest(zip_path, out_dir=DEFAULT_OUT_DIR, seed=0, split_counts=SPLIT_COUNTS) -> dict:
    """Select subjects, extract their ct.nii.gz into out_dir, return the manifest dict."""
    with zipfile.ZipFile(zip_path) as zf:
        candidates, info = [], {}
        for row in read_meta(zf):
            sid = row["image_id"]
            shape, zooms = read_ct_header(zf, sid)
            if not passes_extent_filter(shape, zooms):
                continue
            region = region_for_study_type(row["study_type"])
            candidates.append({"id": sid, "region": region})
            info[sid] = {
                "study_type": row["study_type"],
                "region": region,
                "shape": [int(v) for v in shape[:3]],
                "spacing": [float(v) for v in zooms[:3]],
            }
        assignment = select_and_split(candidates, split_counts, seed)
        subjects = []
        for sid in sorted(assignment):
            path = os.path.join(out_dir, sid, "ct.nii.gz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with zf.open(f"{sid}/ct.nii.gz") as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            subjects.append({"id": sid, "name": f"ts_{sid}", "split": assignment[sid],
                             **info[sid], "path": path, "sha256": _sha256_file(path)})
    return {
        "source": {
            "dataset": "TotalSegmentator small subset v2.0.1",
            "zenodo_record": ZENODO_RECORD,
            "zip": os.path.basename(zip_path),
            "zip_sha256": _sha256_file(zip_path),
            "license": "CC-BY-4.0",
        },
        "selection": {
            "seed": seed,
            "split_counts": {region: list(counts) for region, counts in split_counts.items()},
            "min_inplane_mm": MIN_INPLANE_MM,
            "min_superior_inferior_mm": MIN_SUPERIOR_INFERIOR_MM,
            "n_candidates": len(candidates),
        },
        "subjects": subjects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=DEFAULT_ZIP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.zip):
        raise SystemExit(
            f"{args.zip} not found. Download it first:\n"
            f"  curl -L -o {args.zip} "
            f"'https://zenodo.org/records/{ZENODO_RECORD}/files/{os.path.basename(args.zip)}?download=1'"
        )
    manifest = build_manifest(args.zip, args.out_dir, args.seed)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    counts = {}
    for s in manifest["subjects"]:
        counts[(s["split"], s["region"])] = counts.get((s["split"], s["region"]), 0) + 1
    print(f"[select_totalseg] {len(manifest['subjects'])} subjects from "
          f"{manifest['selection']['n_candidates']} candidates -> {args.manifest}")
    for (split, region), n in sorted(counts.items()):
        print(f"  {split:5s} {region:15s} {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_select_totalseg.py`
Expected: 7 passed.

- [ ] **Step 5: Run the tool on the real zip**

Run: `.venv/bin/python -m tools.select_totalseg`
Expected: `[select_totalseg] 30 subjects from 100 candidates -> data/totalseg_manifest.json`, then 12 lines of split/region counts summing to 20 train, 4 val, 6 test. Then:

```bash
ls data/totalseg | wc -l                        # 30
.venv/bin/python -c "import json; m=json.load(open('data/totalseg_manifest.json')); print(m['source']['zip_sha256'])"
git status --short data/                        # only data/totalseg_manifest.json is new
```

Expected: `30`; `6c47636d9727cd97048616383365dd5a588d0bbc5723b5eb0fa736ed08480ffc`; only the manifest listed (extracted volumes are ignored).

- [ ] **Step 6: Commit**

```bash
git add tools/select_totalseg.py tests/test_select_totalseg.py data/totalseg_manifest.json
git commit -m "feat(data): extract 30 TotalSegmentator CTs and commit the split manifest"
```

---

### Task 4: `totalseg.py` manifest registry

**Files:**
- Create: `totalseg.py`
- Test: `tests/test_totalseg.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_totalseg.py`:

```python
import json

import nibabel as nib
import numpy as np
import pytest

import totalseg


def _write_subject(tmp_path, sid, split, data=None):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4) if data is None else data
    path = tmp_path / "totalseg" / sid / "ct.nii.gz"
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.5, 1.0])), str(path))
    return {"id": sid, "name": f"ts_{sid}", "split": split, "region": "thorax",
            "study_type": "ct thorax", "shape": list(data.shape), "spacing": [1.5, 1.5, 1.5],
            "path": str(path), "sha256": "ab" * 32}


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    subjects = [_write_subject(tmp_path, "s0001", "train"),
                _write_subject(tmp_path, "s0002", "test")]
    subjects.append({**subjects[0], "id": "s0003", "name": "ts_s0003", "split": "val",
                     "path": str(tmp_path / "missing" / "ct.nii.gz")})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"subjects": subjects}))
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(path))
    return subjects


def test_is_totalseg_checks_prefix():
    assert totalseg.is_totalseg("ts_s0001")
    assert not totalseg.is_totalseg("ct_chest")


def test_available_names_lists_only_existing_files(manifest):
    assert totalseg.available_names() == ["ts_s0001", "ts_s0002"]


def test_split_names_returns_split_members(manifest):
    assert totalseg.split_names("train") == ["ts_s0001"]
    assert totalseg.split_names("test") == ["ts_s0002"]


def test_split_names_raises_when_a_member_is_missing(manifest):
    with pytest.raises(FileNotFoundError, match="tools.select_totalseg"):
        totalseg.split_names("val")


def test_split_names_rejects_unknown_split(manifest):
    with pytest.raises(ValueError):
        totalseg.split_names("holdout")


def test_version_uses_manifest_sha(manifest):
    assert totalseg.version("ts_s0001") == "sha256:" + "ab" * 32


def test_load_volume_returns_float32_ras_and_spacing(manifest):
    volume, spacing = totalseg.load_volume("ts_s0001")
    assert volume.dtype == np.float32
    assert volume.shape == (2, 3, 4)
    assert volume[1, 2, 3] == 23.0
    assert spacing == (1.5, 1.5, 1.5)


def test_load_volume_reorients_to_ras(tmp_path, monkeypatch):
    entry = _write_subject(tmp_path, "s0009", "train")
    image = nib.load(entry["path"])
    flipped = nib.Nifti1Image(np.asarray(image.dataobj)[::-1], np.diag([-1.5, 1.5, 1.5, 1.0]))
    nib.save(flipped, entry["path"])            # stored L->R flipped, affine says so
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"subjects": [entry]}))
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(path))
    volume, _ = totalseg.load_volume("ts_s0009")
    assert volume[1, 2, 3] == 23.0              # back in original (RAS) order


def test_load_volume_unknown_and_missing(manifest):
    with pytest.raises(ValueError, match="unknown TotalSegmentator volume"):
        totalseg.load_volume("ts_nope")
    with pytest.raises(FileNotFoundError, match="tools.select_totalseg"):
        totalseg.load_volume("ts_s0003")


def test_missing_manifest_means_no_volumes(tmp_path, monkeypatch):
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(tmp_path / "none.json"))
    assert totalseg.available_names() == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_totalseg.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'totalseg'`.

- [ ] **Step 3: Implement `totalseg.py`**

Create `totalseg.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_totalseg.py`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add totalseg.py tests/test_totalseg.py
git commit -m "feat(data): TotalSegmentator manifest registry and RAS volume loading"
```

---

### Task 5: TotalSegmentator volumes in `datasets.py`

**Files:**
- Modify: `datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
import json

import nibabel as nib

import totalseg


@pytest.fixture
def ts_volume(tmp_path, monkeypatch):
    path = tmp_path / "s0001" / "ct.nii.gz"
    path.parent.mkdir()
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    nib.save(nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.5, 1.0])), str(path))
    entry = {"id": "s0001", "name": "ts_s0001", "split": "train", "region": "thorax",
             "study_type": "ct thorax", "shape": [2, 3, 4], "spacing": [1.5, 1.5, 1.5],
             "path": str(path), "sha256": "cd" * 32}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"subjects": [entry]}))
    monkeypatch.setattr(totalseg, "MANIFEST_PATH", str(manifest))
    datasets.load_dataset.cache_clear()
    yield "ts_s0001"
    datasets.load_dataset.cache_clear()


def test_list_datasets_includes_available_totalseg_volumes(ts_volume):
    assert ts_volume in datasets.list_datasets()


def test_load_dataset_totalseg_returns_ras_float32(ts_volume):
    volume, spacing = datasets.load_dataset(ts_volume)
    assert volume.dtype == np.float32 and volume.shape == (2, 3, 4)
    assert spacing == (1.5, 1.5, 1.5)
    canonical, _ = datasets.load_dataset(ts_volume, canonical=True)
    assert np.array_equal(canonical, volume)


def test_totalseg_version_metadata_and_camera(ts_volume):
    assert datasets._dataset_version(ts_volume) == "sha256:" + "cd" * 32
    metadata = datasets.dataset_metadata(ts_volume)
    assert metadata["orientation"] == "RAS"
    assert metadata["version"] == "sha256:" + "cd" * 32
    assert datasets.default_camera_for(ts_volume) == {"azimuth": 0.0, "elevation": 80.0, "zoom": 1.0}


def test_existing_datasets_keep_dataset_normalized_orientation(monkeypatch):
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    monkeypatch.setattr(datasets, "load_dataset", lambda name, canonical=False: (volume, (1.0, 1.0, 1.0)))
    assert datasets.dataset_metadata("synthetic")["orientation"] == "dataset-normalized"


def test_volumes_for_split(ts_volume):
    assert datasets.volumes_for_split("train") == [ts_volume]
    assert datasets.volumes_for_split("out_of_source") == ["ct_chest", "ct_skull", "ct_cardio", "ct_abdomen"]
    with pytest.raises(ValueError):
        datasets.volumes_for_split("holdout")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_datasets.py`
Expected: the five new tests FAIL (e.g. `ts_s0001` not in `list_datasets()`, `load_dataset() got an unexpected keyword argument 'canonical'`); the existing 18 still pass.

- [ ] **Step 3: Implement**

In `datasets.py`:

1. After the line `from phantom import build_phantom`, add the line `import totalseg`.

2. After the `DATASETS = { … }` dict, add:

```python
# RL v2 evaluates generalization on scans from another source: the four
# calibrated Slicer CTs, loaded with load_dataset(name, canonical=True).
OUT_OF_SOURCE_CT = ("ct_chest", "ct_skull", "ct_cardio", "ct_abdomen")

# TotalSegmentator volumes are RAS; this chat-UI camera shows them from the
# front with superior up (checked by rendering one, 2026-09-15).
TOTALSEG_DEFAULT_CAMERA = {"azimuth": 0.0, "elevation": 80.0, "zoom": 1.0}
```

3. Replace the decorator and signature/docstring/start of `load_dataset`:

```python
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
```

with:

```python
@functools.lru_cache(maxsize=4)
def load_dataset(name: str = "synthetic", canonical: bool = False):
    """Returns (volume: np.ndarray HU, spacing: (float, float, float)).

    canonical=True returns calibrated CT volumes in RAS axis order (numpy
    axis 0 -> patient right, 1 -> anterior, 2 -> superior), which RL v2
    relies on. The default keeps each file's own axis order, which the chat
    UI's per-dataset cameras were tuned for. TotalSegmentator volumes are
    always RAS; the synthetic phantom is treated as RAS.

    Cached: the viewer's chunked-transfer endpoints (dataset_metadata/
    get_volume_chunk in server.py) call this once per HTTP request, and a
    real volume re-reads + re-parses its file every call (~6.5s of server
    time uncached for one dataset's chunks). Bounded to 4 entries because
    the RL v2 registry holds ~30 real CT volumes of 50-150 MB each. No
    caller mutates the returned array in place, so caching is safe.
    """
    if name == "synthetic":
        return build_phantom(), (1.0, 1.0, 1.0)
    if totalseg.is_totalseg(name):
        return totalseg.load_volume(name)
```

(Task 6 adds the `canonical` handling for NRRD volumes; until then the flag is accepted and ignored for them.)

4. Replace `list_datasets`, `default_camera_for` and `_dataset_version` with:

```python
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
```

5. In `dataset_metadata`, replace the line:

```python
        "orientation": "dataset-normalized",
```

with:

```python
        "orientation": "RAS" if totalseg.is_totalseg(name) else "dataset-normalized",
```

- [ ] **Step 4: Run the dataset and server tests**

Run: `.venv/bin/python -m pytest -q tests/test_datasets.py tests/test_totalseg.py tests/test_server.py`
Expected: all pass.

- [ ] **Step 5: Check the real manifest integrates**

Run:

```bash
.venv/bin/python -c "
import datasets
names = datasets.list_datasets()
print(len([n for n in names if n.startswith('ts_')]), 'ts volumes listed')
print({s: len(datasets.volumes_for_split(s)) for s in ('train', 'val', 'test', 'out_of_source')})
v, sp = datasets.load_dataset(datasets.volumes_for_split('test')[0])
print(v.shape, v.dtype, sp, float(v.min()), float(v.max()))
"
```

Expected: `30 ts volumes listed`, `{'train': 20, 'val': 4, 'test': 6, 'out_of_source': 4}`, then a shape like `(3xx, 3xx, 3xx) float32 (1.5, 1.5, 1.5)` with a min near −1024 or below and a max above 1000.

- [ ] **Step 6: Commit**

```bash
git add datasets.py tests/test_datasets.py
git commit -m "feat(data): register TotalSegmentator volumes and RL v2 splits in datasets"
```

---

### Task 6: Canonical RAS loading for the Slicer CTs

**Files:**
- Modify: `datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
def _write_nrrd_header(path, space, directions):
    text = ("NRRD0004\n# comment line\ntype: short\ndimension: 3\n"
            f"space: {space}\nsizes: 2 3 4\n"
            f"space directions: {directions}\n"
            "kinds: domain domain domain\nendian: little\nencoding: raw\n\n")
    path.write_bytes(text.encode("latin-1") + b"\x00" * 48)


def test_read_nrrd_header_parses_fields(tmp_path):
    path = tmp_path / "h.nrrd"
    _write_nrrd_header(path, "left-posterior-superior", "(1,0,0) (0,1,0) (0,0,2)")
    header = datasets._read_nrrd_header(str(path))
    assert header["space"] == "left-posterior-superior"
    assert header["space directions"] == "(1,0,0) (0,1,0) (0,0,2)"
    assert "type" in header and "NRRD0004" not in header


def test_reorient_lps_axis_aligned_flips_x_and_y():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    header = {"space": "left-posterior-superior", "space directions": "(1,0,0) (0,1,0) (0,0,2)"}
    ras, spacing = datasets._reorient_nrrd_to_ras(volume, header)
    assert np.array_equal(ras, volume[::-1, ::-1, :])
    assert spacing == (1.0, 1.0, 2.0)


def test_reorient_permutes_axes_and_spacing():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    # axis0 -> +superior (2 mm), axis1 -> +right (1 mm), axis2 -> -anterior (3 mm)
    header = {"space": "right-anterior-superior", "space directions": "(0,0,2) (1,0,0) (0,-3,0)"}
    ras, spacing = datasets._reorient_nrrd_to_ras(volume, header)
    expected = np.transpose(volume, (1, 2, 0))[:, ::-1, :]
    assert np.array_equal(ras, expected)
    assert spacing == (1.0, 3.0, 2.0)


def test_reorient_rejects_unsupported_space():
    with pytest.raises(ValueError, match="unsupported NRRD space"):
        datasets._reorient_nrrd_to_ras(np.zeros((2, 2, 2), np.float32),
                                       {"space": "scanner-xyz", "space directions": "(1,0,0) (0,1,0) (0,0,1)"})


def test_canonical_rejected_for_uncalibrated_and_mri():
    with pytest.raises(ValueError, match="no canonical RAS orientation"):
        datasets.load_dataset("mri_head", canonical=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_datasets.py`
Expected: the five new tests FAIL (`_read_nrrd_header` / `_reorient_nrrd_to_ras` missing; `mri_head` canonical not rejected — note: if `data/MR-head.nrrd` were missing this test would try to download; it is present in this checkout).

- [ ] **Step 3: Implement**

In `datasets.py`, add `import re` to the standard-library imports and `import nibabel as nib` after `import numpy as np`.

After `_load_nrrd`, add:

```python
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
```

In `load_dataset`, replace the block starting at `path = _ensure_downloaded(name)` through the final `return volume, spacing` with:

```python
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
```

In `dataset_metadata`, the `orientation` value stays as set in Task 5 (the chat UI serves non-canonical volumes; RL v2 code loads canonical ones directly).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_datasets.py`
Expected: all pass.

- [ ] **Step 5: Check the four real volumes load canonically**

Run:

```bash
.venv/bin/python -c "
import datasets
for n in datasets.volumes_for_split('out_of_source'):
    raw, raw_sp = datasets.load_dataset(n)
    ras, sp = datasets.load_dataset(n, canonical=True)
    print(n, raw.shape, '->', ras.shape, tuple(round(s, 3) for s in sp))
"
```

Expected: four lines; shapes are permutations of the raw shapes; spacings are the header's direction-vector lengths (e.g. `ct_chest` → `(0.762, 0.762, 2.5)`).

- [ ] **Step 6: Commit**

```bash
git add datasets.py tests/test_datasets.py
git commit -m "feat(data): canonical RAS loading for Slicer CTs from NRRD headers"
```

---

### Task 7: Orientation check tool, visual verification, README

**Files:**
- Create: `tools/check_orientation.py`
- Test: `tests/test_check_orientation.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

Create `tests/test_check_orientation.py`:

```python
import numpy as np

from tools.check_orientation import projection_panels


def test_projection_panels_put_superior_and_anterior_on_top():
    volume = np.full((10, 20, 30), -1000.0, dtype=np.float32)   # R x A x S
    volume[5, 10, 28] = 2000.0      # near superior end
    volume[5, 18, 5] = 2000.0       # near anterior end
    coronal, sagittal, axial = projection_panels(volume)
    assert coronal.shape == (30, 10) and sagittal.shape == (30, 20) and axial.shape == (20, 10)
    assert coronal[:5].max() == 255          # superior voxel lands in the top rows
    assert sagittal[:5].max() == 255
    assert axial[:5].max() == 255            # anterior voxel lands in the top rows of the axial view
    assert axial[-5:].max() < 255
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_check_orientation.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.check_orientation'`.

- [ ] **Step 3: Implement**

Create `tools/check_orientation.py`:

```python
"""Write maximum-intensity projections of canonical (RAS) volumes for a visual
orientation check: in the coronal and sagittal panels the head (superior) must
be at the top; in the axial panel the front of the body (anterior) must be at
the top.

    python -m tools.check_orientation ct_chest ts_s0011
"""
import argparse
import os

import numpy as np
from PIL import Image

from datasets import load_dataset

OUT_DIR = "out/orientation"
PANEL_HEIGHT = 320
WINDOW_HU = (-200.0, 1000.0)


def projection_panels(volume: np.ndarray):
    """(coronal, sagittal, axial) uint8 images; row 0 is the top of the image."""
    lo, hi = WINDOW_HU
    v = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    to_u8 = lambda plane: (plane * 255).round().astype(np.uint8)
    coronal = v.max(axis=1).T[::-1]      # rows: superior -> inferior, cols: R axis
    sagittal = v.max(axis=0).T[::-1]     # rows: superior -> inferior, cols: A axis
    axial = v.max(axis=2).T[::-1]        # rows: anterior -> posterior, cols: R axis
    return to_u8(coronal), to_u8(sagittal), to_u8(axial)


def _panel_image(plane, row_mm, col_mm):
    width = max(1, round(PANEL_HEIGHT * plane.shape[1] * col_mm / (plane.shape[0] * row_mm)))
    return Image.fromarray(plane).resize((width, PANEL_HEIGHT))


def write_projection(name: str, out_dir: str = OUT_DIR) -> str:
    volume, (sx, sy, sz) = load_dataset(name, canonical=True)
    coronal, sagittal, axial = projection_panels(volume)
    panels = [_panel_image(coronal, sz, sx), _panel_image(sagittal, sz, sy), _panel_image(axial, sy, sx)]
    sheet = Image.new("L", (sum(p.width for p in panels) + 20 * (len(panels) - 1), PANEL_HEIGHT))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + 20
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        print(write_projection(name))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_check_orientation.py`
Expected: 1 passed.

- [ ] **Step 5: Visual verification**

Run:

```bash
.venv/bin/python -m tools.check_orientation ct_chest ct_skull ct_cardio ct_abdomen \
  $(.venv/bin/python -c "import datasets; print(' '.join(datasets.volumes_for_split('test')[:2]))")
```

Open each written `out/orientation/<name>.png` (the Read tool displays PNGs). For every image check: coronal (left) and sagittal (middle) panels have the head/superior end at the top; the axial panel (right) has the front of the body (sternum/abdominal wall; for the skull, the face) at the top. Report each volume as OK or wrong, with what is wrong. If any Slicer CT is wrong, STOP and report it (do not patch the reorientation by hand) — the controller decides.

- [ ] **Step 6: README subsection**

In `README.md`, directly before the line `## Local 3D Viewer`, insert:

````markdown
### RL v2 volumes

RL v2 trains and evaluates on 30 CT scans from the TotalSegmentator small
subset (CC-BY-4.0, Zenodo record 10047263), split by subject into 20 train,
4 validation and 6 test volumes (stratified by body region; see
`data/totalseg_manifest.json`), plus the four Slicer CTs as an out-of-source
test set. Fetch and extract once:

```bash
curl -L -o data/Totalsegmentator_dataset_small_v201.zip \
  "https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip?download=1"
python -m tools.select_totalseg
```

The selected volumes appear as `ts_<subject>` in the dataset list. RL v2 code
loads every volume with `load_dataset(name, canonical=True)` (RAS axis order)
and gets split members from `volumes_for_split("train" | "val" | "test" |
"out_of_source")`. `python -m tools.check_orientation <name>` writes projection
images for a visual orientation check.

````

- [ ] **Step 7: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tools/check_orientation.py tests/test_check_orientation.py README.md
git commit -m "feat(data): orientation check tool and RL v2 data docs"
```
