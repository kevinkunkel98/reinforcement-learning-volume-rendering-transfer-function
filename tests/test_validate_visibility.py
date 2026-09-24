import numpy as np

from tools.validate_visibility import (
    CONTRIBUTION_FLOOR, _label_id, pearson, rank_agreement, summarize, sweep_params,
)
from anatomy import CANONICAL_CLASSES
from transfer import peak_internal


def test_pearson_matches_numpy():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.1, 5.9, 8.2])
    assert np.isclose(pearson(a, b), np.corrcoef(a, b)[0, 1])


def test_pearson_is_zero_for_a_constant_series():
    assert pearson(np.ones(5), np.arange(5.0)) == 0.0


def test_summarize_marks_classes_below_the_noise_floor_unvalidated():
    estimate = {"skeleton": np.array([0.1, 0.2, 0.3]), "muscle": np.array([0.1, 0.2, 0.3])}
    reference = {"skeleton": np.array([0.01, 0.02, 0.031]),
                 "muscle": np.array([0.001, 0.001 + CONTRIBUTION_FLOOR / 10, 0.001])}
    summary = summarize(estimate, reference, threshold=0.7)
    assert summary["skeleton"]["validated"] is True
    assert summary["skeleton"]["passed"] is True
    assert summary["muscle"]["validated"] is False
    assert summary["muscle"]["passed"] is None


def test_summarize_fails_an_uncorrelated_class():
    estimate = {"skeleton": np.array([0.1, 0.2, 0.3, 0.4])}
    reference = {"skeleton": np.array([0.05, 0.01, 0.06, 0.02])}
    summary = summarize(estimate, reference, threshold=0.7)
    assert summary["skeleton"]["validated"] is True
    assert summary["skeleton"]["passed"] is False


def test_sweep_params_varies_the_matching_peak():
    rng = np.random.default_rng(0)
    functions = sweep_params(rng, peak=2, n=10)
    heights = [peak_internal(p, 2)["height"] for p in functions]
    assert all(a < b for a, b in zip(heights, heights[1:]))


def test_rank_agreement_of_identical_order_is_one():
    a = np.array([1.0, 5.0, 2.0, 9.0])
    b = np.array([10.0, 50.0, 20.0, 90.0])
    assert rank_agreement(a, b) == 1.0


def test_rank_agreement_penalizes_reordering():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 1.0, 3.0, 4.0])
    assert rank_agreement(a, b) < 1.0


def test_label_ids_cover_all_eight_canonical_classes():
    assert [_label_id(name) for name in CANONICAL_CLASSES] == list(range(1, 9))


def test_visibility_cache_key_includes_eight_class_layout():
    from visibility import VisibilityModel

    model = VisibilityModel(
        np.zeros((1, 2, 2, 2), dtype=np.uint8), 1.0, np.zeros(16),
        np.zeros((1, 2, 2, 2), dtype=np.uint8), "intensity", "test",
    )
    key = model.cache_key("volume-v1")
    assert len(key) == 16


def test_visibility_cache_key_changes_with_layout_metadata(monkeypatch):
    from visibility import VisibilityModel

    model = VisibilityModel(
        np.zeros((1, 2, 2, 2), dtype=np.uint8), 1.0, np.zeros(16),
        np.zeros((1, 2, 2, 2), dtype=np.uint8), "intensity", "test",
    )
    original = model.cache_key("volume-v1")
    monkeypatch.setattr("visibility.CLASS_LAYOUT_VERSION", "anatomy-test-layout")
    assert model.cache_key("volume-v1") != original


def test_visibility_cache_key_changes_with_transfer_layout(monkeypatch):
    import transfer
    from visibility import VisibilityModel

    model = VisibilityModel(
        np.zeros((1, 2, 2, 2), dtype=np.uint8), 1.0, np.zeros(16),
        np.zeros((1, 2, 2, 2), dtype=np.uint8), "intensity", "test",
    )
    original = model.cache_key("volume-v1")
    monkeypatch.setattr(transfer, "TRANSFER_LAYOUT_VERSION", "transfer-test-layout")
    assert model.cache_key("volume-v1") != original


def test_visibility_cache_key_changes_with_transfer_peak_order(monkeypatch):
    import transfer
    from visibility import VisibilityModel

    model = VisibilityModel(
        np.zeros((1, 2, 2, 2), dtype=np.uint8), 1.0, np.zeros(16),
        np.zeros((1, 2, 2, 2), dtype=np.uint8), "intensity", "test",
    )
    original = model.cache_key("volume-v1")
    monkeypatch.setattr(transfer, "TRANSFER_LAYOUT_ORDER", tuple(reversed(transfer.TRANSFER_LAYOUT_ORDER)))
    assert model.cache_key("volume-v1") != original
