import numpy as np
import pytest

import goals
import transfer
from rl.keep_repair import keep_repair

P = transfer.PARAMS_PER_PEAK


class _OcclusionModel:
    """Each class's visibility is its own peak's height, except the heart,
    which the lungs peak occludes (heart vis = heart height x (1 - lungs
    height)); brightness is the peak's mean colour. Fast, deterministic, and
    enough to tell an unmentioned peak that matters to the goal from one that
    does not."""

    def __init__(self):
        self.calls = 0

    def features(self, params, active_layers=None):
        self.calls += 1
        height = {c: transfer.peak_internal(params, goals.PEAK_INDEX[c])["height"] for c in goals.GOAL_CLASSES}
        vis = dict(height)
        vis["heart"] = height["heart"] * (1.0 - height["lungs"])
        vis["other"] = 0.1
        bright = {c: float(np.mean(transfer.peak_internal(params, goals.PEAK_INDEX[c])["rgb"]))
                  for c in goals.GOAL_CLASSES}
        bright["other"] = 0.5
        return {"vis": vis, "bright": bright, "coverage": min(1.0, sum(vis.values()))}


def _peak(params, goal_class):
    i = goals.PEAK_INDEX[goal_class] * P
    return params[i:i + P]


def _set(params, goal_class, offset, value):
    params[goals.PEAK_INDEX[goal_class] * P + offset] = value


def _start():
    return goals.starting_params("v2")


def _attainment(model, goal, start, params, weights=None):
    start_agg = goals.aggregate(model.features(start))
    final = goals.aggregate(model.features(params))
    return goals.attainment(goal, start_agg, final, **({} if weights is None else {"weights": weights}))


def test_reverts_an_unmentioned_peak_that_does_not_help_the_goal():
    model = _OcclusionModel()
    start = _start()
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    params = start.copy()
    _set(params, "skeleton", 2, 0.4)       # the asked-for change
    _set(params, "spleen", 2, -0.9)        # side effect: spleen faded
    _set(params, "liver", 1, 0.5)          # side effect: liver peak widened
    _set(params, "liver", 3, 0.9)          # ... and recoloured

    model.calls = 0
    repaired, n_evals = keep_repair(model, goal, start, params)

    np.testing.assert_array_equal(_peak(repaired, "spleen"), _peak(start, "spleen"))
    np.testing.assert_array_equal(_peak(repaired, "liver"), _peak(start, "liver"))
    np.testing.assert_array_equal(_peak(repaired, "skeleton"), _peak(params, "skeleton"))
    # start + proposal + one candidate per changed unmentioned peak
    assert n_evals == model.calls == 1 + 1 + 2
    assert _attainment(model, goal, start, repaired) >= _attainment(model, goal, start, params) - 0.01


def test_keeps_an_unmentioned_peak_the_goal_needs():
    model = _OcclusionModel()
    start = _start()
    _set(start, "lungs", 2, 0.4)          # lungs at height 0.7 hide most of the heart
    goal = goals.goal_vector({"heart": {"vis": 0.3}})
    params = start.copy()
    _set(params, "lungs", 2, -0.3)        # halving the lungs reveals the heart
    _set(params, "kidneys", 2, -0.8)      # unrelated side effect

    repaired, n_evals = keep_repair(model, goal, start, params)

    np.testing.assert_array_equal(_peak(repaired, "lungs"), _peak(params, "lungs"))
    np.testing.assert_array_equal(_peak(repaired, "kidneys"), _peak(start, "kidneys"))
    assert n_evals == 1 + 1 + 2


def test_never_touches_mentioned_peaks():
    model = _OcclusionModel()
    start = _start()
    # skeleton mentioned for visibility, liver for brightness only
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}, "liver": {"bright": 0.2}})
    params = start.copy()
    _set(params, "skeleton", 2, -0.9)     # the wrong way: reverting would help, but it is mentioned
    _set(params, "liver", 3, -0.9)        # the wrong way for the liver's brightness too

    repaired, n_evals = keep_repair(model, goal, start, params)

    np.testing.assert_array_equal(repaired, params)
    assert n_evals == 1 + 1                 # nothing unmentioned changed: no candidates


def test_reuses_a_given_start_aggregate_and_leaves_inputs_alone():
    model = _OcclusionModel()
    start = _start()
    start_agg = goals.aggregate(model.features(start))
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    params = start.copy()
    _set(params, "skeleton", 2, 0.4)
    _set(params, "spleen", 2, -0.9)
    before = params.copy()

    model.calls = 0
    repaired, n_evals = keep_repair(model, goal, start, params, start_agg=start_agg)

    assert n_evals == model.calls == 1 + 1
    np.testing.assert_array_equal(params, before)
    np.testing.assert_array_equal(_peak(repaired, "spleen"), _peak(start, "spleen"))


def test_tolerance_bounds_the_total_attainment_loss():
    """A revert is kept only while attainment stays within `eps` of the
    proposal's -- not of the previous revert's, so small losses cannot add up."""
    model = _OcclusionModel()
    start = _start()
    _set(start, "lungs", 2, 0.4)
    goal = goals.goal_vector({"heart": {"vis": 0.3}})
    params = start.copy()
    _set(params, "lungs", 2, -0.3)

    strict, _ = keep_repair(model, goal, start, params, eps=0.01)
    loose, _ = keep_repair(model, goal, start, params, eps=10.0)

    np.testing.assert_array_equal(_peak(strict, "lungs"), _peak(params, "lungs"))
    np.testing.assert_array_equal(loose, start)


def test_largest_change_is_tried_first():
    """Changed unmentioned peaks are tried in order of how far they moved."""
    class _Recording(_OcclusionModel):
        def __init__(self):
            super().__init__()
            self.seen = []

        def features(self, params, active_layers=None):
            self.seen.append(np.asarray(params).copy())
            return super().features(params, active_layers)

    model = _Recording()
    start = _start()
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    params = start.copy()
    _set(params, "skeleton", 2, 0.4)
    _set(params, "spleen", 2, -0.2)     # small change
    _set(params, "kidneys", 2, -0.9)    # large change

    keep_repair(model, goal, start, params)

    first_candidate = model.seen[2]
    np.testing.assert_array_equal(_peak(first_candidate, "kidneys"), _peak(start, "kidneys"))
    np.testing.assert_array_equal(_peak(first_candidate, "spleen"), _peak(params, "spleen"))


def test_a_goal_already_met_at_the_start_returns_the_proposal():
    model = _OcclusionModel()
    start = _start()
    goal = goals.goal_vector({"skeleton": {"vis": 0.0}})
    params = start.copy()
    _set(params, "spleen", 2, -0.9)

    repaired, _ = keep_repair(model, goal, start, params)

    np.testing.assert_array_equal(repaired, params)


def test_peak_classes_follow_the_canonical_peak_order():
    from rl.keep_repair import PEAK_CLASS
    assert PEAK_CLASS == {i: c for c, i in transfer.ANATOMICAL_PEAK_INDEX.items()}
    assert sorted(PEAK_CLASS.values()) == sorted(goals.GOAL_CLASSES)
    assert len(PEAK_CLASS) == transfer.N_PEAKS


@pytest.mark.parametrize("weights", [None, {"skeleton": 2.0}])
def test_scores_with_the_instruction_weights(weights):
    model = _OcclusionModel()
    start = _start()
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    params = start.copy()
    _set(params, "skeleton", 2, 0.4)
    _set(params, "spleen", 2, -0.9)

    repaired, _ = keep_repair(model, goal, start, params, weights=weights)

    np.testing.assert_array_equal(_peak(repaired, "spleen"), _peak(start, "spleen"))


def _with_occluders(targets, occluders):
    """An instruction for `targets` with `occluders` marked the way task v2/v3
    marks them (`goals.occlusion_aware`): mentioned, weight 0."""
    instruction = {"targets": targets, "goal": goals.goal_vector(targets)}
    instruction = goals.occlusion_aware(instruction, set(occluders))
    return instruction["goal"], instruction["weights"]


def test_reverts_an_occluder_fade_the_goal_does_not_need_only_when_asked():
    """The soft peak is marked as an occluder of the skeleton but hides
    nothing in the stub, so fading it is a free side effect: reverted with
    include_occluders=True, left alone (it counts as mentioned) without."""
    model = _OcclusionModel()
    start = _start()
    goal, weights = _with_occluders({"skeleton": {"vis": 0.3}}, {"soft"})
    params = start.copy()
    _set(params, "skeleton", 2, 0.4)
    _set(params, "soft", 2, -0.9)          # occluder erased, for nothing

    default, _ = keep_repair(model, goal, start, params, weights=weights)
    model.calls = 0
    repaired, n_evals = keep_repair(model, goal, start, params, weights=weights, include_occluders=True)

    np.testing.assert_array_equal(default, params)
    np.testing.assert_array_equal(_peak(repaired, "soft"), _peak(start, "soft"))
    np.testing.assert_array_equal(_peak(repaired, "skeleton"), _peak(params, "skeleton"))
    assert n_evals == model.calls == 1 + 1 + 1


def test_keeps_an_occluder_fade_the_goal_needs():
    """Fading the lungs is what reveals the heart: reverting it costs far more
    than eps, so the attainment check keeps it with or without the flag."""
    model = _OcclusionModel()
    start = _start()
    _set(start, "lungs", 2, 0.4)           # lungs at height 0.7 hide most of the heart
    goal, weights = _with_occluders({"heart": {"vis": 0.3}}, {"lungs"})
    params = start.copy()
    _set(params, "lungs", 2, -0.3)

    for include in (False, True):
        repaired, _ = keep_repair(model, goal, start, params, weights=weights, include_occluders=include)
        np.testing.assert_array_equal(repaired, params)


def test_include_occluders_never_touches_weighted_mentioned_classes():
    model = _OcclusionModel()
    start = _start()
    goal, weights = _with_occluders({"skeleton": {"vis": 0.3}, "liver": {"bright": 0.2}}, {"soft"})
    params = start.copy()
    _set(params, "skeleton", 2, -0.9)      # the wrong way, but weighted and mentioned
    _set(params, "liver", 3, -0.9)
    _set(params, "soft", 2, -0.9)

    repaired, _ = keep_repair(model, goal, start, params, weights=weights, include_occluders=True)

    np.testing.assert_array_equal(_peak(repaired, "skeleton"), _peak(params, "skeleton"))
    np.testing.assert_array_equal(_peak(repaired, "liver"), _peak(params, "liver"))
    np.testing.assert_array_equal(_peak(repaired, "soft"), _peak(start, "soft"))


def test_include_occluders_without_weights_has_no_occluders():
    """Without `weights` every mentioned class weighs 1.0: nothing new to try."""
    model = _OcclusionModel()
    start = _start()
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}, "soft": {"vis": -0.3}})
    params = start.copy()
    _set(params, "soft", 2, -0.9)

    repaired, n_evals = keep_repair(model, goal, start, params, include_occluders=True)

    np.testing.assert_array_equal(repaired, params)
    assert n_evals == 1 + 1


def test_occluder_classes_are_the_weight_zero_mentioned_ones():
    from rl.keep_repair import occluder_classes
    goal, weights = _with_occluders({"heart": {"vis": 0.3}}, {"lungs", "soft", "other"})
    assert occluder_classes(goal, weights) == {"lungs", "soft"}
    assert occluder_classes(goal, None) == set()
