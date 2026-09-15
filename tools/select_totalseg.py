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
