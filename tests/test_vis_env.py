import math

import numpy as np
import pytest

import goals
import transfer
from rl.vis_env import ACTION_SIZE, MAX_STEPS, OBSERVATION_SIZE, VisibilityTFEnv


class _StubModel:
    """Like the baselines' stub: each goal-class peak's height/colour maps
    directly to that class's visibility/brightness -- fast and deterministic,
    but still responsive to every controllable value."""

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
    "fake_a": ["skeleton", "lungs", "organs", "muscle", "vessels"],
    "fake_b": ["skeleton", "lungs", "organs"],
}
CONTRAST = {"fake_a": True, "fake_b": False}


def _patch_totalseg(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: CLASSES_PRESENT[name])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: CONTRAST[name])


def _make_env(monkeypatch, volume_ids=("fake_a", "fake_b")):
    _patch_totalseg(monkeypatch)
    return VisibilityTFEnv(list(volume_ids), model_for_volume=lambda name: _StubModel())


# --- shapes and spaces -----------------------------------------------------

def test_observation_and_action_space_shapes(monkeypatch):
    env = _make_env(monkeypatch)
    assert env.observation_space.shape == (OBSERVATION_SIZE,)
    assert env.action_space.shape == (ACTION_SIZE,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


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


# --- reward = drop in goals.distance ----------------------------------------

def test_step_reward_equals_drop_in_goal_distance(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    obs, info = env.reset(seed=1)

    start_agg = goals.aggregate(env._model.features(env._params))
    start_distance = goals.distance(env._instruction["goal"], env._start_agg, start_agg)

    action = np.full(ACTION_SIZE, 0.5, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    final_agg = goals.aggregate(env._model.features(env._params))
    final_distance = goals.distance(env._instruction["goal"], env._start_agg, final_agg)

    assert not info["useless"]
    assert reward == pytest.approx(start_distance - final_distance)


# --- useless penalty ---------------------------------------------------------

def test_useless_penalty_applied_for_an_all_transparent_state(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=2)
    # Force an all-transparent transfer function directly: every controllable
    # value at its floor means every peak's real height is 0.
    env._params = np.full(24, -1.0, dtype=np.float64)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert info["useless"]


def test_useless_penalty_lowers_reward_relative_to_no_penalty(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=2)
    env._params = np.full(24, -1.0, dtype=np.float64)
    prev_distance = env._prev_distance

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    final_agg = goals.aggregate(env._model.features(env._params))
    distance = goals.distance(env._instruction["goal"], env._start_agg, final_agg)
    bare_drop = prev_distance - distance

    assert info["useless"]
    assert reward == pytest.approx(bare_drop - 1.0)


# --- truncation --------------------------------------------------------------

def test_episode_truncates_after_exactly_max_steps(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=3)

    for step in range(1, MAX_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
        assert terminated is False
        if step < MAX_STEPS:
            assert not truncated
        else:
            assert truncated


# --- determinism ---------------------------------------------------------------

def test_reset_with_seed_is_deterministic(monkeypatch):
    env_a = _make_env(monkeypatch)
    env_b = _make_env(monkeypatch)

    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)

    assert info_a["volume"] == info_b["volume"]
    assert info_a["text"] == info_b["text"]
    assert np.array_equal(env_a._params, env_b._params)
    assert np.array_equal(obs_a, obs_b)


# --- action clipping -----------------------------------------------------------

def test_large_actions_are_clipped_to_unit_range(monkeypatch):
    env_a = _make_env(monkeypatch, volume_ids=("fake_a",))
    env_b = _make_env(monkeypatch, volume_ids=("fake_a",))
    env_a.reset(seed=5)
    env_b.reset(seed=5)

    obs_a, *_ = env_a.step(np.full(ACTION_SIZE, 5.0, dtype=np.float32))
    obs_b, *_ = env_b.step(np.full(ACTION_SIZE, 1.0, dtype=np.float32))

    assert np.allclose(obs_a[:12], obs_b[:12])


def test_params_stay_within_bounds_after_many_large_actions(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=6)

    for _ in range(MAX_STEPS):
        env.step(np.full(ACTION_SIZE, 5.0, dtype=np.float32))

    assert np.all(env._params >= -1.0) and np.all(env._params <= 1.0)


# --- attainment matches goals.attainment ----------------------------------------

def test_final_attainment_matches_goals_attainment(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=7)

    info = None
    rng = np.random.default_rng(0)
    for _ in range(MAX_STEPS):
        action = rng.uniform(-1.0, 1.0, size=ACTION_SIZE).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    final_agg = goals.aggregate(env._model.features(env._params))
    expected = goals.attainment(env._instruction["goal"], env._start_agg, final_agg)
    assert info["attainment"] == pytest.approx(expected)


# --- instruction respects the volume's supported classes ------------------------

def test_sampled_instruction_only_targets_supported_classes(monkeypatch):
    env = _make_env(monkeypatch)

    for seed in range(30):
        obs, info = env.reset(seed=seed)
        supported = set(goals.goal_classes_for_volume(info["volume"]))
        assert set(env._instruction["targets"].keys()) <= supported
