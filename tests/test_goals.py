import math

import numpy as np
import pytest

import goals
import visibility
from transfer import PARAMS_PER_PEAK, peak_internal


def _features(skeleton=(0.0, 0.0), lungs=(0.0, 0.0), organs=(0.0, 0.0),
              muscle=(0.0, 0.0), vessels=(0.0, 0.0), coverage=0.5):
    """A raw visibility.features()-shaped dict: each arg is (vis, bright)."""
    values = {"skeleton": skeleton, "lungs": lungs, "organs": organs,
              "muscle": muscle, "vessels": vessels}
    return {"vis": {c: v[0] for c, v in values.items()},
            "bright": {c: v[1] for c, v in values.items()},
            "coverage": coverage}


def _aggregated(skeleton=0.0, lungs=0.0, soft=0.0, vessels=0.0,
                 skeleton_b=0.0, lungs_b=0.0, soft_b=0.0, vessels_b=0.0, coverage=0.5):
    """An aggregate()-shaped dict directly, for tests that don't need aggregate() itself."""
    return {"vis": {"skeleton": skeleton, "lungs": lungs, "soft": soft, "vessels": vessels},
            "bright": {"skeleton": skeleton_b, "lungs": lungs_b, "soft": soft_b, "vessels": vessels_b},
            "coverage": coverage}


def test_starting_params_places_one_peak_per_goal_class():
    params = goals.starting_params()
    assert params.shape == (24,)
    for goal_class, index in goals.PEAK_INDEX.items():
        centre = peak_internal(params, index)["center"]
        assert centre == pytest.approx(goals.PEAK_CENTRES_HU[goal_class], abs=1.0)


def test_aggregate_sums_soft_and_weights_its_brightness():
    features = _features(skeleton=(0.3, 0.9), organs=(0.2, 0.4), muscle=(0.6, 0.8))
    aggregated = goals.aggregate(features)
    assert aggregated["vis"]["soft"] == pytest.approx(0.8)
    assert aggregated["bright"]["soft"] == pytest.approx(0.7)
    assert aggregated["vis"]["skeleton"] == pytest.approx(0.3)
    assert aggregated["bright"]["skeleton"] == pytest.approx(0.9)


def test_aggregate_soft_brightness_is_zero_when_invisible():
    features = _features(organs=(0.0, 0.0), muscle=(0.0, 0.0))
    aggregated = goals.aggregate(features)
    assert aggregated["vis"]["soft"] == pytest.approx(0.0)
    assert aggregated["bright"]["soft"] == 0.0


def test_goal_vector_layout():
    vector = goals.goal_vector({"skeleton": {"vis": 0.3}})
    assert vector.shape == (16,)
    d, m, e, n = vector[0:4], vector[4:8], vector[8:12], vector[12:16]
    skeleton = goals.GOAL_CLASSES.index("skeleton")
    assert d[skeleton] == pytest.approx(0.3)
    assert m[skeleton] == 1.0
    rest = np.delete(vector, [skeleton, 4 + skeleton])
    assert np.all(rest == 0.0)


def test_progress_is_log_change_for_visibility_and_plain_change_for_brightness():
    start = _aggregated(skeleton=0.001, skeleton_b=0.4)
    current = _aggregated(skeleton=0.011, skeleton_b=0.55)
    c, b = goals.progress(start, current)
    assert c["skeleton"] == pytest.approx(0.778, abs=1e-3)
    assert b["skeleton"] == pytest.approx(0.15)


def test_distance_is_zero_when_the_goal_is_met_exactly():
    start = _aggregated(skeleton=0.01)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_vis = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    current = _aggregated(skeleton=new_vis)
    assert goals.distance(goal, start, current) == pytest.approx(0.0, abs=1e-9)


def test_distance_penalises_unmentioned_classes_drifting():
    start = _aggregated(skeleton=0.01, lungs=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_vis = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    without_drift = _aggregated(skeleton=new_vis, lungs=0.05)
    with_drift = _aggregated(skeleton=new_vis, lungs=0.08)
    c_lungs = math.log10(0.08 + goals.EPSILON) - math.log10(0.05 + goals.EPSILON)
    extra = goals.LAMBDA_KEEP * abs(c_lungs)
    d0 = goals.distance(goal, start, without_drift)
    d1 = goals.distance(goal, start, with_drift)
    assert d1 - d0 == pytest.approx(extra)


def test_attainment_is_one_when_reached_and_negative_when_worse():
    start = _aggregated(skeleton=0.01)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    reached_vis = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    reached = _aggregated(skeleton=reached_vis)
    assert goals.attainment(goal, start, reached) == pytest.approx(1.0)

    worse_vis = 10 ** (-0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    worse = _aggregated(skeleton=worse_vis)
    assert goals.attainment(goal, start, worse) < 0.0


def test_is_useless_detects_empty_and_opaque_states():
    empty = _aggregated(skeleton=0.2, coverage=0.0)
    assert goals.is_useless(empty) is True

    opaque = _aggregated(skeleton=0.0, lungs=0.0, soft=0.0, vessels=0.0, coverage=1.0)
    assert goals.is_useless(opaque) is True

    normal = _aggregated(skeleton=0.1, lungs=0.05, soft=0.2, vessels=0.02, coverage=0.5)
    assert goals.is_useless(normal) is False


@pytest.mark.slow
def test_real_check_starting_params_on_ts_s1379():
    model = visibility.for_volume("ts_s1379")
    aggregated = goals.aggregate(model.features(goals.starting_params()))
    print(aggregated["vis"])
    assert aggregated["vis"]["skeleton"] > 0.0
    assert aggregated["vis"]["lungs"] > 0.0
    assert aggregated["vis"]["soft"] > 0.0
    assert goals.is_useless(aggregated) is False
