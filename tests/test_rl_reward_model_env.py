"""Tests for rl/reward_model_env.py."""
import numpy as np
import pytest

import rl.reward_model_env as reward_model_env
from rl.reward_model_env import RewardModelTFEnv


class _StubRewardModel:
    pass  # predict_reward itself is monkeypatched in the fast test below, so
          # this never needs real behavior


def test_step_caches_previous_after_as_next_before(monkeypatch):
    render_calls = []

    def fake_render(volume, params, spacing, camera):
        render_calls.append(params.copy())
        return object()

    def fake_grab(win):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    feature_sequence = iter([
        {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0},  # reset() seed
        {"mean": 1.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0},  # step 1 after
        {"mean": 2.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0},  # step 2 after
    ])

    def fake_features(rgb):
        return next(feature_sequence)

    predict_calls = []

    def fake_predict_reward(model, before_features, after_features, target_tissue, direction):
        predict_calls.append((before_features, after_features))
        return 0.5

    monkeypatch.setattr(reward_model_env.render, "render", fake_render)
    monkeypatch.setattr(reward_model_env.render, "grab", fake_grab)
    monkeypatch.setattr(reward_model_env.render, "features", fake_features)
    monkeypatch.setattr(reward_model_env, "predict_reward", fake_predict_reward)

    env = RewardModelTFEnv(volume=np.zeros((4, 4, 4)), spacing=(1.0, 1.0, 1.0),
                            reward_model=_StubRewardModel(), seed=0)
    env.reset()
    env.step([0.3])
    env.step([0.3])

    assert len(render_calls) == 3  # 1 at reset (seeds cache) + 1 per step
    # step 2's "before" must equal step 1's "after" -- the cached value, not a fresh render
    assert predict_calls[1][0] == predict_calls[0][1]


@pytest.mark.slow
def test_real_render_and_reward_model_end_to_end(tmp_path):
    import json

    from datasets import load_dataset
    from rl.reward_model import train_reward_model

    prefs_path = tmp_path / "preferences.jsonl"
    with open(prefs_path, "w") as f:
        for i in range(10):
            before = {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            after = {"mean": 60.0 if i % 2 == 0 else 40.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            f.write(json.dumps({
                "timestamp": "2026-01-01T00:00:00", "rater_id": "default",
                "target_tissue": "bone", "direction": "increase", "delta": 0.1,
                "before_features": before, "after_features": after,
                "label": 1 if i % 2 == 0 else -1,
            }) + "\n")
    model = train_reward_model(str(prefs_path), str(tmp_path / "reward_model.pt"), epochs=20)

    volume, spacing = load_dataset("synthetic")
    env = RewardModelTFEnv(volume=volume, spacing=spacing, reward_model=model, seed=0)
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step([0.3])
    assert -1.0 <= reward <= 1.0
    assert "automatic_mass_fraction_reward" in info
