"""Tests for the TFEnv Gymnasium environment."""
import numpy as np
import pytest

from rl.env import DIRECTIONS, MAX_STEPS, PARAMS_PER_PEAK, TISSUES, TFEnv


def test_reset_observation_shape_and_bounds():
    env = TFEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (31,)
    onehot = obs[24:29]
    assert onehot.sum() == pytest.approx(1.0)
    assert set(np.unique(onehot)) <= {0.0, 1.0}
    assert obs[29] in (-1.0, 1.0)
    assert 0.0 <= obs[30] <= 1.0


def test_reset_picks_valid_tissue_and_direction():
    env = TFEnv(seed=1)
    env.reset()
    assert env.target_tissue in TISSUES
    assert env.direction in DIRECTIONS


def test_step_moves_height_in_action_direction():
    env = TFEnv(seed=2)
    env.reset()
    base = env.peak_idx * PARAMS_PER_PEAK
    height_before = env.params[base + 2]
    env.step(np.array([1.0], dtype=np.float32))
    height_after = env.params[base + 2]
    assert height_after > height_before or height_before >= 1.0 - 1e-6


def test_reward_sign_matches_goal_direction():
    env = TFEnv(seed=3)
    env.reset()
    env.direction = "increase"
    _, reward, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert reward >= 0.0

    env.reset()
    env.direction = "decrease"
    _, reward, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert reward <= 0.0


def test_height_stays_clipped_to_unit_range():
    env = TFEnv(seed=4)
    env.reset()
    base = env.peak_idx * PARAMS_PER_PEAK
    for _ in range(50):
        env.step(np.array([1.0], dtype=np.float32))
    internal_height = (env.params[base + 2] + 1.0) / 2.0
    assert -1e-6 <= internal_height <= 1.0 + 1e-6


def test_episode_truncates_at_max_steps_and_never_terminates():
    env = TFEnv(seed=5)
    env.reset()
    truncated = False
    for i in range(MAX_STEPS):
        _, _, terminated, truncated, _ = env.step(np.array([0.1], dtype=np.float32))
        assert terminated is False
        if i < MAX_STEPS - 1:
            assert truncated is False
    assert truncated is True
