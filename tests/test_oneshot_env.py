import math

import numpy as np
import pytest

import goals
import transfer
from rl.baselines import CONTROLLABLE
from rl.oneshot_env import ACTION_SIZE, OBSERVATION_SIZE, USELESS_PENALTY, OneShotEnv


class _StubModel:
    """Like the baselines' stub: each goal-class peak's height/colour maps
    directly to that class's visibility/brightness -- fast and deterministic,
    but still responsive to every controllable value."""

    MEASURED = {"skeleton": "skeleton", "lungs": "lungs", "soft": "organs", "vessels": "vessels"}
    # Distinct per underlying (non-goal) class, so a test can tell the
    # appended observation values apart instead of all matching one constant.
    SOLO_MAX = {"skeleton": 0.6, "lungs": 0.3, "organs": 0.4, "muscle": 0.1, "vessels": 0.05}

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)
        self.solo_max_calls = 0

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
        self.solo_max_calls += 1
        return self.SOLO_MAX[name]


CLASSES_PRESENT = {
    "fake_a": ["skeleton", "lungs", "organs", "muscle", "vessels"],
    "fake_b": ["skeleton", "lungs", "organs"],
}
CONTRAST = {"fake_a": True, "fake_b": False}


def _patch_totalseg(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: CLASSES_PRESENT[name])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: CONTRAST[name])


def _make_env(monkeypatch, volume_ids=("fake_a", "fake_b")):
    _patch_totalseg(monkeypatch)
    return OneShotEnv(list(volume_ids), model_for_volume=lambda name: _StubModel())


# --- shapes and spaces -----------------------------------------------------

def test_observation_and_action_space_shapes(monkeypatch):
    env = _make_env(monkeypatch)
    assert env.observation_space.shape == (OBSERVATION_SIZE,)
    assert env.action_space.shape == (ACTION_SIZE,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_observation_appends_log_solo_max_per_goal_class(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    obs, info = env.reset(seed=0)

    model = env._model
    expected = [math.log10(sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[c]) + goals.EPSILON)
                for c in goals.GOAL_CLASSES]

    # Layout: goal(16) + histogram(16) + log_vis(4) + bright(4) + solo_max(4)
    # + controllable(12) + coverage(1) == 57, appended right after brightness.
    start = 16 + 16 + 4 + 4
    appended = obs[start:start + 4]
    assert appended == pytest.approx(expected, abs=1e-5)


def test_solo_max_is_computed_once_per_volume_across_resets(monkeypatch):
    # sample_instruction() may itself call model.solo_max() (e.g. for
    # "absolute" instructions), so the *total* call count after reset()
    # depends on which instruction kind was sampled -- isolate the env's own
    # per-volume caching (_solo_max_log) instead of asserting an exact total.
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)
    model = env._model

    assert "fake_a" in env._solo_max_log_cache
    calls_before = model.solo_max_calls
    first = env._solo_max_log("fake_a", model)
    second = env._solo_max_log("fake_a", model)

    assert model.solo_max_calls == calls_before   # already cached during reset(), no new calls
    assert second is first                        # the exact cached list, not a recomputation


def test_observation_is_finite_and_inside_space_after_reset_and_step(monkeypatch):
    env = _make_env(monkeypatch)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBSERVATION_SIZE,)
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
    assert obs.shape == (OBSERVATION_SIZE,)
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)


# --- the action IS the new transfer function, not a delta -------------------

def test_action_writes_parameters_directly_not_a_delta(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)
    # starting_params() seeds non-zero heights/widths/colours per peak, so the
    # start value at every controllable index is not the mid-range 0.0.
    assert not np.allclose([env._start_params[i] for group in CONTROLLABLE for i in group], 0.0)

    env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    # A group's value is its mean (what the observation reports for it), so an
    # action of 0 lands the group at 0 regardless of where it started.
    for group in CONTROLLABLE:
        assert float(np.mean([env._params[i] for i in group])) == pytest.approx(0.0)


def test_nonzero_action_sets_each_group_mean_and_keeps_the_peak_coloured(monkeypatch):
    """The action sets each group's mean; within the (r, g, b) group the
    channel offsets around that mean -- the peak's hue -- survive. Writing the
    scalar into all three channels instead made every policy render grey, which
    identified the policy's candidate on sight during preference collection
    (see `rl.baselines.apply_controllable`)."""
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)
    # Clear of ±1 so no colour channel saturates -- clipping one would shift
    # its group's mean off the action value (`test_large_actions_are_clipped`).
    action = np.linspace(-0.4, 0.4, ACTION_SIZE, dtype=np.float32)

    env.step(action)

    for value, group in zip(action, CONTROLLABLE):
        got = [env._params[i] for i in group]
        assert float(np.mean(got)) == pytest.approx(float(value), abs=1e-6)
        start = [env._start_params[i] for i in group]
        if max(start) - min(start) > 1e-9:
            assert max(got) - min(got) > 1e-9, "peak went grey"


def test_large_actions_are_clipped_to_unit_range(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)

    env.step(np.full(ACTION_SIZE, 5.0, dtype=np.float32))

    assert np.all(env._params >= -1.0) and np.all(env._params <= 1.0)


# --- termination -------------------------------------------------------------

def test_episode_always_terminates_after_one_step(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert terminated is True
    assert truncated is False


# --- reward = clipped attainment, minus the useless penalty -----------------

def test_reward_equals_clipped_attainment_when_not_useless(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert not info["useless"]
    assert reward == pytest.approx(float(np.clip(info["attainment"], -1.0, 1.0)))


def test_useless_penalty_lowers_reward(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)

    # Every controllable value at its floor: every peak's real height is 0,
    # so nothing is drawn.
    obs, reward, terminated, truncated, info = env.step(np.full(ACTION_SIZE, -1.0, dtype=np.float32))

    assert info["useless"]
    assert reward == pytest.approx(float(np.clip(info["attainment"], -1.0, 1.0)) - USELESS_PENALTY)


def test_info_attainment_is_unclipped(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    final_agg = goals.aggregate(env._model.features(env._params))
    expected = goals.attainment(env._instruction["goal"], env._start_agg, final_agg)
    assert info["attainment"] == pytest.approx(expected)
    # Not necessarily inside [-1, 1] -- the clip only applies to the reward.


def test_info_has_the_expected_keys(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert set(info.keys()) == {"attainment", "kind", "volume", "text", "useless", "goal_source"}
    assert info["volume"] == "fake_a"
    assert info["goal_source"] == "instruction"   # hindsight_ratio defaults to 0
    assert info["kind"] == env._instruction["kind"]
    assert info["text"] == env._instruction["text"]


# --- determinism ---------------------------------------------------------------

def test_reset_with_seed_is_deterministic(monkeypatch):
    env_a = _make_env(monkeypatch)
    env_b = _make_env(monkeypatch)

    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)

    assert info_a["volume"] == info_b["volume"]
    assert info_a["text"] == info_b["text"]
    assert np.array_equal(env_a._start_params, env_b._start_params)
    assert np.array_equal(obs_a, obs_b)


# --- instruction respects the volume's supported classes ------------------------

def test_sampled_instruction_only_targets_supported_classes(monkeypatch):
    env = _make_env(monkeypatch)

    for seed in range(30):
        obs, info = env.reset(seed=seed)
        supported = set(goals.goal_classes_for_volume(info["volume"]))
        assert set(env._instruction["targets"].keys()) <= supported


# --- hindsight goals: derived from a reachable target, not invented ------------

def test_hindsight_episodes_are_solvable_by_the_action_that_made_them():
    # The goal is derived from a target the action space can reach, so the
    # action that produced the target must score near-perfect attainment.
    # Anything less means the goal encoding and the reward disagree.
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    env.reset(seed=0)
    oracle = env.hindsight_action()
    _obs, _reward, _done, _truncated, info = env.step(oracle)
    assert info["attainment"] > 0.8


def test_hindsight_ratio_zero_keeps_sampling_instructions():
    env = OneShotEnv(["synthetic"], hindsight_ratio=0.0)
    _obs, info = env.reset(seed=0)
    assert info["goal_source"] == "instruction"
    assert env.hindsight_action() is None


def test_hindsight_episodes_report_their_source():
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    _obs, info = env.reset(seed=0)
    assert info["goal_source"] == "hindsight"


def test_hindsight_goal_mentions_at_least_one_class():
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    env.reset(seed=1)
    goal = env._instruction["goal"]
    mentioned = goal[4:8]  # the m[4] block of goals.goal_vector
    assert mentioned.sum() >= 1.0
