import math

import numpy as np
import pytest

import goals
import transfer
from rl import vis_eval
from rl.vis_env import ACTION_SIZE, MAX_STEPS, VisibilityTFEnv


class _StubModel:
    """Same shape as the stub in test_vis_env.py / test_vis_train.py: each
    goal-class peak's height/colour maps directly to that class's
    visibility/brightness -- fast and deterministic, but still responsive to
    every controllable value."""

    MEASURED = {"skeleton": "skeleton", "lungs": "lungs", "soft": "organs", "vessels": "vessels"}

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)

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


CLASSES_PRESENT = {
    "stub_a": ["skeleton", "lungs", "organs", "muscle", "vessels"],
    "stub_b": ["skeleton", "lungs", "organs", "muscle", "vessels"],
}


def _patch_totalseg(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: CLASSES_PRESENT[name])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)


def _model_for_volume(name):
    return _StubModel()


class _FixedActionModel:
    """Stand-in for an SB3 model: `predict` always returns the same
    deterministic action."""

    def __init__(self, action=None):
        self.action = action if action is not None else np.zeros(ACTION_SIZE, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        assert deterministic is True
        return self.action, None


# --- fixed_episodes -----------------------------------------------------------

def test_fixed_episodes_is_deterministic(monkeypatch):
    _patch_totalseg(monkeypatch)
    kwargs = dict(volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume, seed=3)

    first = vis_eval.fixed_episodes("val", 6, **kwargs)
    second = vis_eval.fixed_episodes("val", 6, **kwargs)

    assert len(first) == 6
    for a, b in zip(first, second):
        assert a["volume"] == b["volume"]
        assert np.array_equal(a["start_params"], b["start_params"])
        assert a["instruction"]["text"] == b["instruction"]["text"]
        assert np.array_equal(a["instruction"]["goal"], b["instruction"]["goal"])


def test_fixed_episodes_only_uses_volumes_of_the_requested_split(monkeypatch):
    _patch_totalseg(monkeypatch)
    monkeypatch.setattr(vis_eval.datasets, "volumes_for_split",
                         lambda split: {"val": ["stub_a", "stub_b"], "train": ["other"]}[split])

    episodes = vis_eval.fixed_episodes("val", 10, seed=0, model_for_volume=_model_for_volume)

    assert {episode["volume"] for episode in episodes} <= {"stub_a", "stub_b"}


def test_fixed_episodes_start_params_have_full_transfer_function_length(monkeypatch):
    _patch_totalseg(monkeypatch)
    [episode] = vis_eval.fixed_episodes(
        "val", 1, seed=0, volume_ids=("stub_a",), model_for_volume=_model_for_volume)
    assert episode["start_params"].shape == (transfer.N_PEAKS * transfer.PARAMS_PER_PEAK,)


# --- run_policy -----------------------------------------------------------------

def test_run_policy_replays_the_exact_episode_state(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 3, seed=1, volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume)
    action = np.full(ACTION_SIZE, 0.2, dtype=np.float32)
    model = _FixedActionModel(action=action)

    results = vis_eval.run_policy("unused.zip", episodes, model_for_volume=_model_for_volume,
                                   load_model=lambda path: model)

    assert len(results) == 3
    for row, episode in zip(results, episodes):
        # Reproduce the same episode by hand: freeze the same recorded
        # (volume, start_params, instruction), then take the same fixed
        # action for MAX_STEPS.
        env = VisibilityTFEnv([episode["volume"]], model_for_volume=_model_for_volume)
        env._volume = episode["volume"]
        env._model = _model_for_volume(episode["volume"])
        raw = env._model.features(episode["start_params"])
        start_agg = goals.aggregate(raw)
        env._params = episode["start_params"].copy()
        env._instruction = episode["instruction"]
        env._start_agg = start_agg
        start_distance = goals.distance(episode["instruction"]["goal"], start_agg, start_agg)
        env._start_distance = start_distance
        env._prev_distance = start_distance
        env._step_count = 0

        info = None
        for _ in range(MAX_STEPS):
            _, _, _, _, info = env.step(action)

        assert row["attainment"] == pytest.approx(info["attainment"])
        assert row["kind"] == episode["instruction"]["kind"]


def test_run_policy_loads_the_model_from_model_path(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 1, seed=0, volume_ids=("stub_a",), model_for_volume=_model_for_volume)
    seen_paths = []

    def fake_loader(path):
        seen_paths.append(path)
        return _FixedActionModel()

    vis_eval.run_policy("some/path.zip", episodes, model_for_volume=_model_for_volume, load_model=fake_loader)

    assert seen_paths == ["some/path.zip"]


# --- run_baseline -----------------------------------------------------------------

def test_run_baseline_do_nothing_scores_zero(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 4, seed=2, volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume)

    results = vis_eval.run_baseline("B0_do_nothing", episodes, model_for_volume=_model_for_volume)

    assert len(results) == 4
    for row in results:
        assert row["attainment"] == pytest.approx(0.0, abs=1e-9)


def test_run_baseline_records_none_when_the_baseline_raises(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 2, seed=0, volume_ids=("stub_a",), model_for_volume=_model_for_volume)

    def _broken(model, start_params, instruction):
        raise RuntimeError("boom")

    monkeypatch.setitem(vis_eval.BASELINES, "B_broken", _broken)
    results = vis_eval.run_baseline("B_broken", episodes, model_for_volume=_model_for_volume)

    assert results == [{"attainment": None, "kind": row["instruction"]["kind"]} for row in episodes]


# --- wilcoxon --------------------------------------------------------------------

def test_wilcoxon_matches_a_hand_computed_example():
    # x - y = [3, -1, 2, -4, 5]: n = 5, no zero differences, no tied |diff|.
    # Ranks of |diff| ascending are just 1..5 (all distinct):
    #   |3| -> 3   |1| -> 1   |2| -> 2   |4| -> 4   |5| -> 5
    # W+ (ranks of positive diffs 3, 2, 5) = 3 + 2 + 5 = 10
    # W- (ranks of negative diffs -1, -4)  = 1 + 4     = 5
    # statistic = min(W+, W-) = 5
    # mean = n(n+1)/4 = 5*6/4 = 7.5
    # sigma = sqrt(n(n+1)(2n+1)/24) = sqrt(5*6*11/24) = sqrt(13.75) ~= 3.708099
    # continuity-corrected z = (5 - 7.5 + 0.5) / 3.708099 ~= -0.539360
    # p = 2*(1 - Phi(0.539360)) ~= 0.589639
    x = [4.0, 2.0, 5.0, 1.0, 7.0]
    y = [1.0, 3.0, 3.0, 5.0, 2.0]   # x - y = [3, -1, 2, -4, 5]

    result = vis_eval.wilcoxon(x, y)

    assert result["n"] == 5
    assert result["statistic"] == pytest.approx(5.0)
    expected_p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(0.5393598899705937 / math.sqrt(2.0))))
    assert result["p_value"] == pytest.approx(expected_p, abs=1e-9)
    assert result["p_value"] == pytest.approx(0.589639, abs=1e-5)


def test_wilcoxon_returns_p_one_when_series_are_identical():
    x = [0.1, 0.2, 0.3, -0.4]
    y = [0.1, 0.2, 0.3, -0.4]

    result = vis_eval.wilcoxon(x, y)

    assert result["n"] == 0
    assert result["p_value"] == pytest.approx(1.0)


def test_wilcoxon_averages_ranks_for_tied_absolute_differences():
    # x - y = [2, -2, 3, 1]: two diffs share |2| -> both get rank (1+2)/2 = 1.5;
    # |1| -> rank 1... wait, |1| < |2| < |3|, sorted |diff| = [1, 2, 2, 3],
    # so |1| -> rank 1, the two |2|s share ranks 2 and 3 -> average 2.5, |3| -> rank 4.
    x = [3.0, 1.0, 4.0, 2.0]
    y = [1.0, 3.0, 1.0, 1.0]   # x - y = [2, -2, 3, 1]

    result = vis_eval.wilcoxon(x, y)

    # W+ (diffs 2, 3, 1 -> ranks 2.5, 4, 1) = 2.5 + 4 + 1 = 7.5
    # W- (diff -2 -> rank 2.5)              = 2.5
    assert result["n"] == 4
    assert result["statistic"] == pytest.approx(2.5)


def test_wilcoxon_requires_paired_series_of_equal_length():
    with pytest.raises(ValueError):
        vis_eval.wilcoxon([1.0, 2.0], [1.0])


# --- compare ------------------------------------------------------------------

def test_compare_aggregates_per_kind_and_handles_a_failed_baseline_episode():
    results = {
        "policy": [
            {"attainment": 0.8, "kind": "relative"},
            {"attainment": 0.6, "kind": "relative"},
            {"attainment": 0.4, "kind": "compound"},
        ],
        "B0_do_nothing": [
            {"attainment": 0.0, "kind": "relative"},
            {"attainment": 0.0, "kind": "relative"},
            {"attainment": None, "kind": "compound"},   # e.g. the baseline raised
        ],
    }

    comparison = vis_eval.compare(results)

    summary = comparison["summary"]
    # Robust stats (median, mean_clipped, mean_raw, share_positive, n), from
    # goals.summarise_attainment -- see test_goals.py for the formulas.
    assert summary["policy"]["median"] == pytest.approx(0.6)
    assert summary["policy"]["mean_raw"] == pytest.approx((0.8 + 0.6 + 0.4) / 3.0)
    assert summary["policy"]["mean_clipped"] == pytest.approx((0.8 + 0.6 + 0.4) / 3.0)
    assert summary["policy"]["share_positive"] == pytest.approx(1.0)
    assert summary["policy"]["n"] == 3
    assert summary["policy"]["by_kind"]["relative"] == pytest.approx(0.7)
    assert summary["policy"]["by_kind"]["compound"] == pytest.approx(0.4)
    assert summary["B0_do_nothing"]["median"] == pytest.approx(0.0)
    assert summary["B0_do_nothing"]["mean_raw"] == pytest.approx(0.0)
    assert summary["B0_do_nothing"]["n"] == 2   # the None episode is dropped
    assert summary["B0_do_nothing"]["by_kind"]["compound"] is None

    comparison_row = comparison["comparisons"]["B0_do_nothing"]
    assert comparison_row["n"] == 2   # only the two episodes with both sides present
    assert 0.0 <= comparison_row["p_value"] <= 1.0


def test_compare_uses_the_given_policy_name():
    results = {
        "my_policy": [{"attainment": 1.0, "kind": "relative"}],
        "B0_do_nothing": [{"attainment": 0.0, "kind": "relative"}],
    }
    comparison = vis_eval.compare(results, policy_name="my_policy")
    assert "B0_do_nothing" in comparison["comparisons"]
    assert "my_policy" not in comparison["comparisons"]
