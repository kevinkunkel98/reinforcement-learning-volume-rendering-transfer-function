import json
import math

import numpy as np
import pytest

import goals
import provenance
import transfer
from rl import vis_eval
from rl.baselines import CONTROLLABLE
from rl.oneshot_env import OneShotEnv
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


class _CountingActionModel(_FixedActionModel):
    """Same as `_FixedActionModel`, but counts how many times `predict` was
    called -- so a test can assert exactly one env step per one-shot
    episode."""

    def __init__(self, action=None):
        super().__init__(action=action)
        self.calls = 0

    def predict(self, obs, deterministic=True):
        self.calls += 1
        return super().predict(obs, deterministic=deterministic)


class _CountingFeaturesModel:
    """Wraps a `_StubModel`, counting calls to `features()` -- so a test can
    isolate how many visibility evaluations a refinement search spends,
    separate from the one-shot proposal's own setup/scoring calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def features(self, params) -> dict:
        self.calls += 1
        return self._inner.features(params)

    def solo_max(self, name: str) -> float:
        return self._inner.solo_max(name)

    @property
    def histogram(self):
        return self._inner.histogram


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


def test_fixed_episodes_one_shot_is_deterministic(monkeypatch):
    _patch_totalseg(monkeypatch)
    kwargs = dict(volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume, seed=3,
                  formulation="one_shot")

    first = vis_eval.fixed_episodes("val", 6, **kwargs)
    second = vis_eval.fixed_episodes("val", 6, **kwargs)

    assert len(first) == 6
    for a, b in zip(first, second):
        assert a["volume"] == b["volume"]
        assert np.array_equal(a["start_params"], b["start_params"])
        assert a["instruction"]["text"] == b["instruction"]["text"]
        assert np.array_equal(a["instruction"]["goal"], b["instruction"]["goal"])


def test_fixed_episodes_one_shot_only_uses_volumes_of_the_requested_split(monkeypatch):
    _patch_totalseg(monkeypatch)
    monkeypatch.setattr(vis_eval.datasets, "volumes_for_split",
                         lambda split: {"val": ["stub_a", "stub_b"], "train": ["other"]}[split])

    episodes = vis_eval.fixed_episodes("val", 10, seed=0, model_for_volume=_model_for_volume,
                                        formulation="one_shot")

    assert {episode["volume"] for episode in episodes} <= {"stub_a", "stub_b"}


def test_fixed_episodes_one_shot_start_params_differ_from_multi_step_convention(monkeypatch):
    # Same seed and single volume, so the only remaining difference is each
    # env's own reset-noise convention (OneShotEnv.START_NOISE applied
    # uniformly per controllable group vs. VisibilityTFEnv's separate
    # width/height/brightness noise draws) -- the two conventions must not
    # coincidentally produce the same start parameters.
    _patch_totalseg(monkeypatch)
    kwargs = dict(volume_ids=("stub_a",), model_for_volume=_model_for_volume, seed=5)

    [multi_step_episode] = vis_eval.fixed_episodes("val", 1, formulation="multi_step", **kwargs)
    [one_shot_episode] = vis_eval.fixed_episodes("val", 1, formulation="one_shot", **kwargs)

    assert not np.array_equal(multi_step_episode["start_params"], one_shot_episode["start_params"])


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


def test_run_policy_one_shot_replays_the_exact_episode_state(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 3, seed=1, volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume,
        formulation="one_shot")
    action = np.full(ACTION_SIZE, 0.2, dtype=np.float32)
    model = _FixedActionModel(action=action)

    results = vis_eval.run_policy("unused.zip", episodes, model_for_volume=_model_for_volume,
                                   load_model=lambda path: model, formulation="one_shot")

    assert len(results) == 3
    for row, episode in zip(results, episodes):
        # Reproduce the same episode by hand: freeze the same recorded
        # (volume, start_params, instruction) into a `OneShotEnv`, then take
        # the same fixed action for its single step.
        env = OneShotEnv([episode["volume"]], model_for_volume=_model_for_volume)
        env._volume = episode["volume"]
        env._model = _model_for_volume(episode["volume"])
        raw = env._model.features(episode["start_params"])
        start_agg = goals.aggregate(raw)
        env._start_params = episode["start_params"].copy()
        env._params = env._start_params
        env._instruction = episode["instruction"]
        env._start_agg = start_agg
        env._start_distance = goals.distance(episode["instruction"]["goal"], start_agg, start_agg)

        _, _, _, _, info = env.step(action)

        assert row["attainment"] == pytest.approx(info["attainment"])
        assert row["kind"] == episode["instruction"]["kind"]


def test_run_policy_one_shot_takes_exactly_one_step_per_episode(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 3, seed=0, volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume,
        formulation="one_shot")
    model = _CountingActionModel(action=np.full(ACTION_SIZE, 0.2, dtype=np.float32))

    vis_eval.run_policy("unused.zip", episodes, model_for_volume=_model_for_volume,
                         load_model=lambda path: model, formulation="one_shot")

    assert model.calls == 3


def test_run_policy_defaults_to_multi_step_formulation(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 2, seed=0, volume_ids=("stub_a",), model_for_volume=_model_for_volume)
    model = _CountingActionModel(action=np.full(ACTION_SIZE, 0.2, dtype=np.float32))

    vis_eval.run_policy("unused.zip", episodes, model_for_volume=_model_for_volume,
                         load_model=lambda path: model)

    assert model.calls == 2 * MAX_STEPS


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


# --- run_policy_with_refinement ----------------------------------------------------

def test_run_policy_with_refinement_zero_evaluations_matches_plain_policy(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 3, seed=0, volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume,
        formulation="one_shot")
    action = np.full(ACTION_SIZE, 0.2, dtype=np.float32)
    policy = _FixedActionModel(action=action)

    plain = vis_eval.run_policy("unused.zip", episodes, model_for_volume=_model_for_volume,
                                 load_model=lambda path: policy, formulation="one_shot")
    refined = vis_eval.run_policy_with_refinement(
        "unused.zip", episodes, evaluations=0, model_for_volume=_model_for_volume,
        load_model=lambda path: policy)

    assert refined == plain


def test_run_policy_with_refinement_never_scores_worse_than_the_proposal(monkeypatch):
    # A hand-built episode (bypassing fixed_episodes' randomness) whose
    # instruction unambiguously wants more skeleton visibility, and a fixed
    # action that keeps things visible -- so the plain one-shot proposal
    # scores clearly better than a "search" that hides everything.
    _patch_totalseg(monkeypatch)
    start_params = goals.starting_params()
    instruction = {"kind": "relative", "text": "more bone",
                   "targets": {"skeleton": {"vis": 0.3}},
                   "goal": goals.goal_vector({"skeleton": {"vis": 0.3}})}
    episodes = [{"volume": "stub_a", "start_params": start_params, "instruction": instruction}]
    action = np.full(ACTION_SIZE, 0.6, dtype=np.float32)
    policy = _FixedActionModel(action=action)

    plain = vis_eval.run_policy("unused.zip", episodes, model_for_volume=_model_for_volume,
                                 load_model=lambda path: policy, formulation="one_shot")
    proposal_attainment = plain[0]["attainment"]

    def _hides_everything(model, start_params, instruction, evaluations=200, initial_step=0.2):
        bad = start_params.copy()
        for group in CONTROLLABLE:
            for index in group:
                bad[index] = -1.0
        return bad

    monkeypatch.setattr(vis_eval, "hill_climb", _hides_everything)

    refined = vis_eval.run_policy_with_refinement(
        "unused.zip", episodes, evaluations=5, model_for_volume=_model_for_volume,
        load_model=lambda path: policy)

    assert refined[0]["attainment"] == pytest.approx(proposal_attainment)


def test_run_policy_with_refinement_search_uses_exactly_the_budgeted_evaluations(monkeypatch):
    # `evaluations` bounds rl.baselines.hill_climb's own model.features
    # calls (its documented contract); run_policy_with_refinement spends one
    # further call scoring the search's result against the original start,
    # so the refinement's total is evaluations + 1 beyond the plain
    # proposal's own setup/scoring calls.
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 1, seed=0, volume_ids=("stub_a",), model_for_volume=_model_for_volume,
        formulation="one_shot")
    action = np.full(ACTION_SIZE, 0.2, dtype=np.float32)
    policy = _FixedActionModel(action=action)

    zero = _CountingFeaturesModel(_StubModel())
    vis_eval.run_policy_with_refinement(
        "unused.zip", episodes, evaluations=0, model_for_volume=lambda name: zero,
        load_model=lambda path: policy)

    five = _CountingFeaturesModel(_StubModel())
    vis_eval.run_policy_with_refinement(
        "unused.zip", episodes, evaluations=5, model_for_volume=lambda name: five,
        load_model=lambda path: policy)

    assert five.calls - zero.calls == 5 + 1


# --- run_baseline -----------------------------------------------------------------

def test_run_baseline_do_nothing_scores_zero(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 4, seed=2, volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume)

    results = vis_eval.run_baseline("B0_do_nothing", episodes, model_for_volume=_model_for_volume)

    assert len(results) == 4
    for row in results:
        assert row["attainment"] == pytest.approx(0.0, abs=1e-9)


def test_run_baseline_do_nothing_scores_zero_on_one_shot_episodes(monkeypatch):
    # Baselines are scored identically regardless of which formulation
    # generated the episodes -- the episode record's shape is the same.
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 4, seed=2, volume_ids=("stub_a", "stub_b"), model_for_volume=_model_for_volume,
        formulation="one_shot")

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
    relative = summary["policy"]["by_kind"]["relative"]
    assert relative["median"] == pytest.approx(0.7)
    assert relative["mean"] == pytest.approx(0.7)
    assert relative["share_positive"] == pytest.approx(1.0)
    assert relative["n"] == 2
    assert summary["policy"]["by_kind"]["compound"]["median"] == pytest.approx(0.4)
    assert summary["B0_do_nothing"]["median"] == pytest.approx(0.0)
    assert summary["B0_do_nothing"]["mean_raw"] == pytest.approx(0.0)
    assert summary["B0_do_nothing"]["n"] == 2   # the None episode is dropped
    empty = summary["B0_do_nothing"]["by_kind"]["compound"]
    assert empty["n"] == 0 and empty["median"] is None and empty["mean"] is None

    comparison_row = comparison["comparisons"]["B0_do_nothing"]
    assert comparison_row["n"] == 2   # only the two episodes with both sides present
    assert 0.0 <= comparison_row["p_value"] <= 1.0


def test_summarize_reports_the_median_per_kind_not_just_the_mean():
    # The incident this guards: attainment is unbounded below, so one
    # pathological episode dragged a kind's mean to -23.5 while its median was
    # comfortably positive, and the kind read as broken for an hour.
    rows = [{"attainment": value, "kind": "relative"}
            for value in (0.5, 0.6, 0.4, 0.55, -100.0)]

    by_kind = vis_eval._summarize(rows)["by_kind"]["relative"]

    assert by_kind["median"] == pytest.approx(0.5)
    assert by_kind["mean"] == pytest.approx(sum((0.5, 0.6, 0.4, 0.55, -100.0)) / 5.0)
    assert by_kind["share_positive"] == pytest.approx(0.8)
    assert by_kind["n"] == 5


# --- reading by_kind out of old and new result files -----------------------------

def test_kind_stats_normalises_the_new_dict_shape():
    stats = vis_eval.kind_stats({"median": 0.5, "mean": -2.0, "share_positive": 0.6, "n": 10})
    assert stats == {"median": 0.5, "mean": -2.0, "share_positive": 0.6, "n": 10}


def test_kind_stats_reads_an_old_float_valued_entry_as_a_mean_only():
    # out/rl_v2/*.json written before this change store a bare mean per kind.
    # A median cannot be recovered from one, so it must stay None rather than
    # be quietly filled in with the mean.
    stats = vis_eval.kind_stats(-23.5)
    assert stats["mean"] == pytest.approx(-23.5)
    assert stats["median"] is None
    assert stats["share_positive"] is None
    assert stats["n"] is None


def test_kind_stats_reads_an_old_empty_kind():
    assert vis_eval.kind_stats(None) == {"median": None, "mean": None,
                                          "share_positive": None, "n": None}


def test_print_table_handles_a_file_mixing_old_float_and_new_dict_kinds(capsys):
    comparison = {
        "summary": {
            "policy": {"median": 0.5, "mean_clipped": 0.5, "mean_raw": 0.5,
                        "share_positive": 1.0, "n": 2,
                        "by_kind": {"relative": {"median": 0.5, "mean": 0.4,
                                                  "share_positive": 1.0, "n": 2},
                                    "compound": -23.5,          # old float shape
                                    "show_only": None}},        # old empty kind
        },
        "comparisons": {},
    }

    vis_eval._print_table(comparison)

    out = capsys.readouterr().out
    assert "-23.500" in out       # the old mean is still shown, not dropped
    assert "0.500" in out


def test_compare_uses_the_given_policy_name():
    results = {
        "my_policy": [{"attainment": 1.0, "kind": "relative"}],
        "B0_do_nothing": [{"attainment": 0.0, "kind": "relative"}],
    }
    comparison = vis_eval.compare(results, policy_name="my_policy")
    assert "B0_do_nothing" in comparison["comparisons"]
    assert "my_policy" not in comparison["comparisons"]


# --- CLI -------------------------------------------------------------------------

def test_cli_default_formulation_is_one_shot():
    args = vis_eval.parse_args(["--policy", "some/path.zip"])
    assert args.formulation == "one_shot"


def test_cli_formulation_can_be_set_to_multi_step():
    args = vis_eval.parse_args(["--policy", "some/path.zip", "--formulation", "multi_step"])
    assert args.formulation == "multi_step"


def test_cli_refine_defaults_to_off():
    args = vis_eval.parse_args(["--policy", "some/path.zip"])
    assert args.refine == 0


def test_cli_refine_can_be_set():
    args = vis_eval.parse_args(["--policy", "some/path.zip", "--refine", "10"])
    assert args.refine == 10


def test_compare_result_metadata_reports_one_shot_contract():
    comparison = vis_eval.compare({"policy": [{"attainment": 0.5, "kind": "relative"}]})

    assert comparison["metadata"]["observation_size"] == 97
    assert comparison["metadata"]["action_size"] == 24
    assert comparison["metadata"]["anatomy_layout"] == "anatomy-v2"


def test_per_class_summary_counts_every_class_and_scores_requested_reachable_only(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 1, seed=0, volume_ids=("stub_a",), model_for_volume=_model_for_volume,
        formulation="one_shot")
    results = {"policy": [{"attainment": 0.4, "kind": episodes[0]["instruction"]["kind"]}]}

    report = vis_eval.per_class_summary(results, episodes, model_for_volume=_model_for_volume)

    assert set(report["policy"]) == set(goals.GOAL_CLASSES)
    assert sum(entry["n"] for entry in report["policy"].values()) == 1
    assert sum(entry["unsupported"] + entry["unreachable"] + entry["reachable"]
               for entry in report["policy"].values()) == len(goals.GOAL_CLASSES)


def test_compare_accepts_and_returns_per_class_summary(monkeypatch):
    _patch_totalseg(monkeypatch)
    episodes = vis_eval.fixed_episodes(
        "val", 1, seed=0, volume_ids=("stub_a",), model_for_volume=_model_for_volume,
        formulation="one_shot")
    results = {"policy": [{"attainment": 0.4, "kind": episodes[0]["instruction"]["kind"]}]}

    comparison = vis_eval.compare(results, episodes=episodes, model_for_volume=_model_for_volume)

    assert "per_class" in comparison
    assert set(comparison["per_class"]["policy"]) == set(goals.GOAL_CLASSES)


def test_per_class_summary_rejects_row_episode_length_mismatch():
    with pytest.raises(ValueError, match="rows.*episodes"):
        vis_eval.per_class_summary(
            {"policy": []}, [{"volume": "stub_a"}])


# --- provenance ------------------------------------------------------------------

def _stub_run(monkeypatch):
    """Enough of a run for `main` to reach the writing path without touching a
    checkpoint, a volume or a baseline."""
    episode = {"volume": "stub_a", "start_params": None,
               "instruction": {"kind": "relative", "goal": {}}}
    monkeypatch.setattr(vis_eval, "fixed_episodes", lambda *a, **k: [episode])
    monkeypatch.setattr(vis_eval, "run_policy",
                        lambda *a, **k: [{"attainment": 0.5, "kind": "relative"}])
    monkeypatch.setattr(vis_eval, "BASELINES", {})


def test_main_records_code_provenance_and_explicit_layer_absence(tmp_path, monkeypatch):
    # The incident: a job started before a scoring fix wrote its file hours
    # after the fix landed. The file must name the code it actually ran.
    _stub_run(monkeypatch)
    out = tmp_path / "eval.json"

    vis_eval.main(["--policy", "p.zip", "--out", str(out)])

    written = json.loads(out.read_text())["provenance"]
    assert written["anatomy_layers"] is None
    assert written["scoring_fingerprint"] == provenance.IMPORT_TIME_PROVENANCE["scoring_fingerprint"]
    assert written["anatomy_layer_layout"] == provenance.IMPORT_TIME_PROVENANCE["anatomy_layer_layout"]


def test_parse_args_accepts_inline_active_layers_json():
    args = vis_eval.parse_args(["--policy", "p.zip", "--layers",
                                '{"liver": {"opacity": 0.0}}'])

    assert args.layers == '{"liver": {"opacity": 0.0}}'


def test_main_loads_cli_layers_into_provenance(tmp_path, monkeypatch):
    _stub_run(monkeypatch)
    out = tmp_path / "eval-cli-layers.json"

    vis_eval.main(["--policy", "p.zip", "--layers",
                   '{"liver": {"opacity": 0.0}}', "--out", str(out)])

    assert json.loads(out.read_text())["provenance"]["anatomy_layers"]["liver"]["opacity"] == 0.0


def test_main_records_non_default_evaluation_layers(tmp_path, monkeypatch):
    _stub_run(monkeypatch)
    layers = {"liver": {"opacity": 0.0}}
    received = {}
    real_result_provenance = provenance.result_provenance
    monkeypatch.setattr(vis_eval.provenance, "result_provenance",
                        lambda active_layers=None: (received.setdefault("layers", active_layers),
                                                    real_result_provenance(active_layers))[1])
    out = tmp_path / "eval-layers.json"

    vis_eval.main(["--policy", "p.zip", "--out", str(out)], active_layers=layers)

    assert json.loads(out.read_text())["provenance"]["anatomy_layers"]["liver"]["opacity"] == 0.0
    assert received["layers"] == layers


def test_check_provenance_passes_active_layers_as_expected_layers(monkeypatch):
    layers = {"liver": {"opacity": 0.0}}
    received = {}

    def compare(recorded, current=None, expected_layers=None):
        received["layers"] = expected_layers
        return {"stale": False, "reasons": [], "recorded_dirty": False}

    monkeypatch.setattr(vis_eval.provenance, "compare", compare)

    assert vis_eval.check_provenance({"provenance": {}}, active_layers=layers)["stale"] is False
    assert received["layers"] == layers


def _comparison_with(provenance_record) -> dict:
    return {
        "summary": {"policy": {"median": 0.5, "mean_clipped": 0.5, "mean_raw": 0.5,
                                "share_positive": 1.0, "n": 1,
                                "by_kind": {"relative": {"median": 0.5, "mean": 0.5,
                                                          "share_positive": 1.0, "n": 1}}}},
        "comparisons": {},
        "provenance": provenance_record,
    }


def test_print_table_warns_loudly_when_the_recorded_scoring_code_is_stale(capsys):
    stale = {**provenance.IMPORT_TIME_PROVENANCE, "scoring_fingerprint": "000000000000"}

    vis_eval._print_table(_comparison_with(stale))

    out = capsys.readouterr().out
    assert "STALE" in out
    assert "000000000000" in out


def test_print_table_is_quiet_when_the_provenance_matches(capsys):
    vis_eval._print_table(_comparison_with(dict(provenance.IMPORT_TIME_PROVENANCE)))
    assert "STALE" not in capsys.readouterr().out


def test_print_table_warns_when_a_result_carries_no_provenance_at_all(capsys):
    comparison = _comparison_with(None)
    del comparison["provenance"]

    vis_eval._print_table(comparison)

    assert "STALE" in capsys.readouterr().out


def test_show_loads_a_result_file_and_warns_that_it_is_stale(tmp_path, capsys):
    path = tmp_path / "eval_rerun.json"
    stale = {**provenance.IMPORT_TIME_PROVENANCE, "scoring_fingerprint": "deadbeefcafe"}
    path.write_text(json.dumps(_comparison_with(stale)))

    vis_eval.main(["--show", str(path)])

    out = capsys.readouterr().out
    assert "STALE" in out
    assert "deadbeefcafe" in out


def test_show_does_not_need_a_policy(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_comparison_with(dict(provenance.IMPORT_TIME_PROVENANCE))))
    args = vis_eval.parse_args(["--show", str(path)])
    assert args.show == str(path)


def test_cli_still_requires_a_policy_when_not_showing_a_file():
    with pytest.raises(SystemExit):
        vis_eval.parse_args([])


# --- episodes_detail ------------------------------------------------------------

def test_episodes_detail_tags_every_row_with_the_episode_it_came_from():
    episodes = [{"volume": "ts_s0454"}, {"volume": "ts_s0477"}]
    results = {"policy": [{"attainment": 0.4, "kind": "absolute"},
                          {"attainment": -0.1, "kind": "compound"}],
               "B0_do_nothing": [{"attainment": 0.0, "kind": "absolute"},
                                 {"attainment": 0.0, "kind": "compound"}]}

    detail = vis_eval.episodes_detail(results, episodes)

    assert detail["policy"] == [
        {"attainment": 0.4, "kind": "absolute", "volume": "ts_s0454"},
        {"attainment": -0.1, "kind": "compound", "volume": "ts_s0477"},
    ]
    assert [row["volume"] for row in detail["B0_do_nothing"]] == ["ts_s0454", "ts_s0477"]


def test_episodes_detail_refuses_rows_that_do_not_line_up_with_the_episodes():
    episodes = [{"volume": "ts_s0454"}, {"volume": "ts_s0477"}]
    results = {"policy": [{"attainment": 0.4, "kind": "absolute"}]}

    with pytest.raises(ValueError, match="policy"):
        vis_eval.episodes_detail(results, episodes)


def test_main_writes_one_row_per_episode_for_every_method(tmp_path, monkeypatch):
    episodes = [{"volume": "ts_s0454"}, {"volume": "ts_s0477"}]
    rows = [{"attainment": 0.5, "kind": "absolute"}, {"attainment": 0.1, "kind": "relative"}]
    monkeypatch.setattr(vis_eval, "fixed_episodes", lambda *a, **k: episodes)
    monkeypatch.setattr(vis_eval, "run_policy", lambda *a, **k: rows)
    monkeypatch.setattr(vis_eval, "run_baseline", lambda name, eps, **k: rows)
    monkeypatch.setattr(vis_eval, "BASELINES", {"B0_do_nothing": None})
    out = tmp_path / "eval.json"

    vis_eval.main(["--policy", "stub.zip", "--split", "test", "--episodes", "2",
                   "--out", str(out)])

    written = json.loads(out.read_text())
    assert written["episodes_detail"]["policy"][0]["volume"] == "ts_s0454"
    assert len(written["episodes_detail"]["B0_do_nothing"]) == 2
