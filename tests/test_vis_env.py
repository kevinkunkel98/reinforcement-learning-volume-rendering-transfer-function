
import numpy as np
import pytest

import goals
import transfer
from anatomy import CANONICAL_CLASSES
from rl import vis_env
from rl.vis_env import ACTION_SIZE, DISTANCE_FLOOR, MAX_STEPS, OBSERVATION_SIZE, REWARD_CLIP, VisibilityTFEnv


class _StubModel:
    """Like the baselines' stub: each goal-class peak's height/colour maps
    directly to that class's visibility/brightness -- fast and deterministic,
    but still responsive to every controllable value."""

    MEASURED = {name: name for name in CANONICAL_CLASSES}

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)

    def features(self, params) -> dict:
        vis, bright = {}, {}
        for goal_class, measured in self.MEASURED.items():
            idx = goals.PEAK_INDEX[goal_class]
            peak = transfer.peak_internal(params, idx)
            vis[measured] = max(float(peak["height"]), 0.0)
            bright[measured] = float(sum(peak["rgb"]) / 3.0)
        coverage = min(1.0, sum(vis.values()))
        return {"vis": vis, "bright": bright, "coverage": coverage}

    def solo_max(self, name: str) -> float:
        return 1.0


CLASSES_PRESENT = {
    "fake_a": list(CANONICAL_CLASSES),
    "fake_b": ["skeleton", "lungs", "soft"],
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
    assert OBSERVATION_SIZE == 106
    assert ACTION_SIZE == transfer.TOTAL_PARAMS // transfer.PARAMS_PER_PEAK * 3
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


# --- reward = drop in goals.distance, normalised by episode difficulty ------

def test_step_reward_equals_distance_drop_normalised_by_start_distance(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    obs, info = env.reset(seed=1)

    start_agg = goals.aggregate(env._model.features(env._params))
    start_distance = goals.distance(env._instruction["goal"], env._start_agg, start_agg)

    action = np.full(ACTION_SIZE, 0.5, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    final_agg = goals.aggregate(env._model.features(env._params))
    final_distance = goals.distance(env._instruction["goal"], env._start_agg, final_agg)

    denom = max(env._start_distance, DISTANCE_FLOOR)
    expected = float(np.clip((start_distance - final_distance) / denom, -REWARD_CLIP, REWARD_CLIP))

    assert not info["useless"]
    assert reward == pytest.approx(expected)


def test_normalised_reward_matches_a_worked_example(monkeypatch):
    # An episode with D_start=0.4 whose distance halves to 0.2 in one step
    # should earn reward (0.4 - 0.2) / 0.4 = 0.5 -- the same as the episode's
    # attainment at that point, since D_start is the fixed denominator for
    # every step (see the module docstring's telescoping-sum argument).
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=1)
    env._start_distance = 0.4
    env._prev_distance = 0.4
    monkeypatch.setattr(vis_env.goals, "distance", lambda goal, start, current: 0.2)
    monkeypatch.setattr(vis_env.goals, "is_useless", lambda features: False)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert reward == pytest.approx(0.5)


def test_undiscounted_return_telescopes_to_attainment_when_unclipped(monkeypatch):
    # Reward is (prev - current) / D_start every step, so summing it over an
    # episode telescopes to (D_start - D_final) / D_start == attainment,
    # as long as no per-step reward hits the clip.
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=1)
    env._start_distance = 1.0
    env._prev_distance = 1.0

    # goals.distance is called more than once per step (step() itself, then
    # again inside goals.attainment() while building info["attainment"]), so
    # the stub must be idempotent: memoize by the identity of `current` and
    # special-case the start-vs-start baseline call attainment() also makes.
    step_distances = iter([0.8, 0.6, 0.5])
    cache = {}

    def fake_distance(goal, start, current):
        if current is start:
            return 1.0
        key = id(current)
        if key not in cache:
            cache[key] = next(step_distances)
        return cache[key]

    monkeypatch.setattr(vis_env.goals, "distance", fake_distance)
    monkeypatch.setattr(vis_env.goals, "is_useless", lambda features: False)

    total_reward = 0.0
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
        total_reward += reward

    assert total_reward == pytest.approx((1.0 - 0.5) / 1.0)


def test_reward_is_clipped_for_a_catastrophic_step(monkeypatch):
    # A start distance near the floor combined with a huge jump in distance
    # would otherwise produce an unbounded reward; it must be clipped to
    # [-REWARD_CLIP, REWARD_CLIP].
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=1)
    env._start_distance = DISTANCE_FLOOR
    env._prev_distance = 0.0
    monkeypatch.setattr(vis_env.goals, "distance", lambda goal, start, current: 10.0)
    monkeypatch.setattr(vis_env.goals, "is_useless", lambda features: False)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert reward == pytest.approx(-REWARD_CLIP)


def test_reward_clip_is_applied_before_the_useless_penalty(monkeypatch):
    # The clip bounds the raw distance-drop term to [-1, 1]; USELESS_PENALTY
    # is subtracted afterward, so a useless catastrophic step can still read
    # below -REWARD_CLIP.
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=1)
    env._start_distance = DISTANCE_FLOOR
    env._prev_distance = 0.0
    monkeypatch.setattr(vis_env.goals, "distance", lambda goal, start, current: 10.0)
    monkeypatch.setattr(vis_env.goals, "is_useless", lambda features: True)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert reward == pytest.approx(-REWARD_CLIP - 1.0)


# --- useless penalty ---------------------------------------------------------

def test_useless_penalty_applied_for_an_all_transparent_state(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=2)
    # Force an all-transparent transfer function directly: every controllable
    # value at its floor means every peak's real height is 0.
    env._params = np.full(transfer.TOTAL_PARAMS, -1.0, dtype=np.float64)

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    assert info["useless"]


def test_useless_penalty_lowers_reward_relative_to_no_penalty(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=2)
    env._params = np.full(transfer.TOTAL_PARAMS, -1.0, dtype=np.float64)
    prev_distance = env._prev_distance
    start_distance = env._start_distance

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))

    final_agg = goals.aggregate(env._model.features(env._params))
    distance = goals.distance(env._instruction["goal"], env._start_agg, final_agg)
    denom = max(start_distance, DISTANCE_FLOOR)
    normalised_drop = float(np.clip((prev_distance - distance) / denom, -REWARD_CLIP, REWARD_CLIP))

    assert info["useless"]
    assert reward == pytest.approx(normalised_drop - 1.0)


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
