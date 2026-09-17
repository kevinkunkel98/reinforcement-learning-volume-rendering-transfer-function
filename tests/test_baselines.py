import numpy as np
import pytest

import commands
import goals
import transfer
from rl import baselines


class _StubModel:
    """A synthetic model where each goal-class peak's height/colour maps
    directly to that class's visibility/brightness (organs stands in for
    `soft`; muscle stays at zero) -- fast and deterministic, but still
    responsive to every controllable value, unlike the real renderer-backed
    VisibilityModel these baselines are meant to run against."""

    MEASURED = {"skeleton": "skeleton", "lungs": "lungs", "soft": "organs", "vessels": "vessels"}

    def features(self, params) -> dict:
        vis, bright = {}, {}
        for goal_class, measured in self.MEASURED.items():
            idx = goals.PEAK_INDEX[goal_class]
            peak = transfer.peak_internal(params, idx)
            vis[measured] = max(float(peak["height"]), 0.0)
            bright[measured] = float(sum(peak["rgb"]) / 3.0)
        vis["muscle"] = 0.0
        bright["muscle"] = 0.0
        coverage = min(1.0, sum(vis.values()))
        return {"vis": vis, "bright": bright, "coverage": coverage}

    def solo_max(self, name: str) -> float:
        return 1.0


class _CountingModel(_StubModel):
    def __init__(self):
        self.calls = 0

    def features(self, params) -> dict:
        self.calls += 1
        return super().features(params)


def _start_params():
    return goals.starting_params()


def _peak_height(params, goal_class):
    return transfer.peak_internal(params, goals.PEAK_INDEX[goal_class])["height"]


def _peak_rgb(params, goal_class):
    return transfer.peak_internal(params, goals.PEAK_INDEX[goal_class])["rgb"]


def _relative_instruction(goal_class: str, vis_delta: float) -> dict:
    targets = {goal_class: {"vis": vis_delta}}
    return {"kind": "relative", "text": "test", "targets": targets, "goal": goals.goal_vector(targets)}


def _show_only_instruction(shown: list) -> dict:
    targets = {}
    for goal_class in goals.GOAL_CLASSES:
        targets[goal_class] = {"vis": goals.HIDE_STRENGTH if goal_class in shown else -goals.HIDE_STRENGTH}
    return {"kind": "show_only", "text": "test", "targets": targets, "goal": goals.goal_vector(targets)}


def _absolute_instruction(model, start_params, goal_class: str, level: str) -> dict:
    start_vis = goals.aggregate(model.features(start_params))["vis"][goal_class]
    delta = goals._absolute_target_delta(model, goal_class, level, start_vis)
    targets = {goal_class: {"vis": delta}}
    return {"kind": "absolute", "text": "test", "targets": targets, "goal": goals.goal_vector(targets)}


def _brightness_instruction(goal_class: str, bright_delta: float) -> dict:
    targets = {goal_class: {"bright": bright_delta}}
    return {"kind": "brightness", "text": "test", "targets": targets, "goal": goals.goal_vector(targets)}


# --- B0 do_nothing -------------------------------------------------------------

def test_do_nothing_returns_start_unchanged_and_scores_zero():
    model = _StubModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", goals.VISIBILITY_STRENGTH["moderately"])

    result = baselines.do_nothing(model, start, instruction)

    assert np.array_equal(result, start)
    assert result is not start  # a copy, not the same array object

    start_agg = goals.aggregate(model.features(start))
    final_agg = goals.aggregate(model.features(result))
    assert goals.attainment(instruction["goal"], start_agg, final_agg) == pytest.approx(0.0)


# --- B1 current_executor -----------------------------------------------------

def test_current_executor_only_touches_the_mentioned_peak():
    model = _StubModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", goals.VISIBILITY_STRENGTH["moderately"])

    result = baselines.current_executor(model, start, instruction)

    assert _peak_height(result, "skeleton") > _peak_height(start, "skeleton")
    for other in ("lungs", "soft", "vessels"):
        assert _peak_height(result, other) == pytest.approx(_peak_height(start, other))


def test_current_executor_show_only_zeroes_the_others():
    model = _StubModel()
    start = _start_params()
    instruction = _show_only_instruction(["skeleton"])

    result = baselines.current_executor(model, start, instruction)

    assert _peak_height(result, "skeleton") == pytest.approx(0.7)
    for other in ("lungs", "soft", "vessels"):
        assert _peak_height(result, other) == pytest.approx(0.0)


def test_current_executor_absolute_sets_the_level():
    model = _StubModel()
    start = _start_params()
    instruction = _absolute_instruction(model, start, "vessels", "high")

    result = baselines.current_executor(model, start, instruction)

    assert _peak_height(result, "vessels") == pytest.approx(commands.LEVEL_WORDS["high"])


def test_current_executor_brightness_changes_colour_not_height():
    model = _StubModel()
    start = _start_params()
    instruction = _brightness_instruction("lungs", goals.BRIGHTNESS_STRENGTH["moderately"])

    result = baselines.current_executor(model, start, instruction)

    assert _peak_height(result, "lungs") == pytest.approx(_peak_height(start, "lungs"))
    assert _peak_rgb(result, "lungs") != _peak_rgb(start, "lungs")
    for other in ("skeleton", "soft", "vessels"):
        assert _peak_rgb(result, other) == pytest.approx(_peak_rgb(start, other))


# --- B5 occlusion_rule --------------------------------------------------------

def test_occlusion_rule_suppresses_the_other_peaks_when_increasing():
    model = _StubModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", goals.VISIBILITY_STRENGTH["moderately"])

    executor_result = baselines.current_executor(model, start, instruction)
    result = baselines.occlusion_rule(model, start, instruction)

    assert _peak_height(result, "skeleton") == pytest.approx(_peak_height(executor_result, "skeleton"))
    for other in ("lungs", "soft", "vessels"):
        assert _peak_height(result, other) == pytest.approx(0.02)


def test_occlusion_rule_matches_the_executor_when_only_decreasing():
    model = _StubModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", -goals.VISIBILITY_STRENGTH["moderately"])

    executor_result = baselines.current_executor(model, start, instruction)
    result = baselines.occlusion_rule(model, start, instruction)

    assert np.array_equal(result, executor_result)


# --- B2 random_policy ----------------------------------------------------------

def test_random_policy_is_seeded_and_stays_in_bounds():
    model = _StubModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", goals.VISIBILITY_STRENGTH["moderately"])

    a = baselines.random_policy(model, start, instruction, seed=7)
    b = baselines.random_policy(model, start, instruction, seed=7)
    c = baselines.random_policy(model, start, instruction, seed=8)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (24,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


# --- B3/B4 hill_climb ----------------------------------------------------------

def test_hill_climb_reduces_the_goal_distance():
    model = _StubModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", goals.VISIBILITY_STRENGTH["strongly"])
    start_agg = goals.aggregate(model.features(start))

    result = baselines.hill_climb(model, start, instruction, evaluations=50)

    before = goals.distance(instruction["goal"], start_agg, start_agg)
    after = goals.distance(instruction["goal"], start_agg, goals.aggregate(model.features(result)))
    assert after <= before


def test_hill_climb_respects_its_evaluation_budget():
    model = _CountingModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", goals.VISIBILITY_STRENGTH["strongly"])

    baselines.hill_climb(model, start, instruction, evaluations=17)

    assert model.calls <= 17


# --- shape/bounds for every baseline --------------------------------------------

def test_every_baseline_returns_valid_params():
    model = _StubModel()
    start = _start_params()
    instruction = _relative_instruction("skeleton", goals.VISIBILITY_STRENGTH["moderately"])

    for name, fn in baselines.BASELINES.items():
        result = fn(model, start, instruction)
        assert result.shape == (24,), name
        assert np.all(np.isfinite(result)), name
        assert np.all(result >= -1.0) and np.all(result <= 1.0), name


def test_apply_controllable_sets_each_group_to_the_action_value():
    """A single-index group (width, height) takes the action value exactly --
    the behaviour every caller relied on before colour was separated out."""
    start = goals.starting_params()
    action = np.linspace(-1.0, 1.0, len(baselines.CONTROLLABLE))
    params = baselines.apply_controllable(start, action)
    for group, value in zip(baselines.CONTROLLABLE, action):
        if len(group) == 1:
            assert params[group[0]] == pytest.approx(value)


def test_apply_controllable_keeps_each_peak_coloured():
    """The (r, g, b) group is one action value meaning the peak's *brightness*.
    Writing that scalar into all three channels makes every render grey, which
    told a rater at a glance which candidate came from the policy -- the
    collected pairs were not blind. The channel offsets around the group mean
    are the peak's hue and must survive."""
    start = goals.starting_params()
    action = np.zeros(len(baselines.CONTROLLABLE))
    params = baselines.apply_controllable(start, action)
    for peak in range(transfer.N_PEAKS):
        base = peak * transfer.PARAMS_PER_PEAK
        start_rgb = [start[base + 3], start[base + 4], start[base + 5]]
        got_rgb = [params[base + 3], params[base + 4], params[base + 5]]
        if max(start_rgb) - min(start_rgb) > 1e-9:
            assert max(got_rgb) - min(got_rgb) > 1e-9, f"peak {peak} went grey"
        offset = np.mean(start_rgb)
        for start_channel, got_channel in zip(start_rgb, got_rgb):
            assert got_channel == pytest.approx(start_channel - offset, abs=1e-9)


def test_apply_controllable_round_trips_through_the_observation():
    """What the observation reports for a group (its mean) is what an action
    of that value reproduces, so the policy's action space is unchanged."""
    start = goals.starting_params()
    action = np.linspace(-0.4, 0.4, len(baselines.CONTROLLABLE))
    params = baselines.apply_controllable(start, action)
    got = [float(np.mean([params[i] for i in group])) for group in baselines.CONTROLLABLE]
    assert got == pytest.approx(list(action), abs=1e-9)


def test_apply_controllable_clips_to_the_parameter_range():
    start = goals.starting_params()
    action = np.full(len(baselines.CONTROLLABLE), 1.0)
    params = baselines.apply_controllable(start, action)
    assert params.min() >= -1.0 and params.max() <= 1.0
