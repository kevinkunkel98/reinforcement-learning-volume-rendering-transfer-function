import numpy as np

from tools.validate_visibility import (
    CONTRIBUTION_FLOOR, pearson, rank_agreement, summarize, sweep_params,
)
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
