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
