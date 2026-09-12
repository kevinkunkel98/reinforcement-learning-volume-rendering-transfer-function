"""Tests for rl/reward_model_env.py."""
import numpy as np
import pytest

import rl.reward_model_env as reward_model_env
from rl.reward_model_env import RewardModelTFEnv


class _StubRewardModel:
    pass  # predict_reward itself is monkeypatched in the fast test below, so
          # this never needs real behavior


def _patch_render(monkeypatch, feature_sequence):
    monkeypatch.setattr(reward_model_env.render, "render", lambda *args: object())
    monkeypatch.setattr(reward_model_env.render, "grab", lambda win: object())
    monkeypatch.setattr(
        reward_model_env.render, "features", lambda rgb: next(feature_sequence)
    )


def _env(monkeypatch, features, **kwargs):
    _patch_render(monkeypatch, iter(features))
    return RewardModelTFEnv(
        volume=np.zeros((4, 4, 4)),
        spacing=(1.0, 1.0, 1.0),
        reward_model=kwargs.pop("reward_model", _StubRewardModel()),
        seed=0,
        **kwargs,
    )


def _features(mean=0.0, coverage=0.5):
    return {"mean": mean, "std": 0.0, "coverage": coverage, "entropy": 0.0}


def test_alpha_one_uses_ensemble_mean_minus_std(monkeypatch):
    predictions = iter([0.8, 0.4])
    monkeypatch.setattr(reward_model_env, "predict_reward", lambda *args: next(predictions))
    env = _env(monkeypatch, [_features(), _features()], reward_model=[object(), object()], alpha=1.0)

    env.reset()
    _, reward, _, _, _ = env.step([0.3])

    assert reward == pytest.approx(0.4)


def test_alpha_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match="alpha"):
        RewardModelTFEnv(np.zeros((4, 4, 4)), (1.0, 1.0, 1.0), _StubRewardModel(), alpha=-0.1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"alpha": np.nan}, "alpha"),
        ({"coverage_threshold": -0.1}, "coverage_threshold"),
        ({"coverage_threshold": 1.1}, "coverage_threshold"),
        ({"coverage_threshold": np.inf}, "coverage_threshold"),
        ({"mean_opacity_threshold": -1.0}, "mean_opacity_threshold"),
        ({"mean_opacity_threshold": np.inf}, "mean_opacity_threshold"),
        ({"hard_penalty": 0.0}, "hard_penalty"),
        ({"hard_penalty": np.inf}, "hard_penalty"),
    ],
)
def test_reward_parameters_must_be_finite_and_sensible(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RewardModelTFEnv(
            np.zeros((4, 4, 4)), (1.0, 1.0, 1.0), _StubRewardModel(), **kwargs
        )


def test_alpha_zero_uses_signed_objective_anchor(monkeypatch):
    class _UnusableModel:
        def __getattr__(self, name):
            raise AssertionError("model inference must be skipped")

    env = _env(monkeypatch, [_features(), _features()], reward_model=_UnusableModel(), alpha=0.0)
    env.direction = "increase"
    monkeypatch.setattr(
        reward_model_env.TFEnv,
        "step",
        lambda self, action: (np.zeros(1), 0.5, False, False, {}),
    )

    env.reset()
    _, reward, _, _, _ = env.step([0.3])

    assert reward == pytest.approx(0.5)


@pytest.mark.parametrize(
    "features",
    [[_features(mean=0.0, coverage=0.01), _features(mean=0.0, coverage=0.01)],
     [_features(mean=1.0, coverage=0.5), _features(mean=1.0, coverage=0.5)]],
)
def test_degenerate_render_receives_independent_hard_penalty(monkeypatch, features):
    monkeypatch.setattr(reward_model_env, "predict_reward", lambda *args: 0.0)
    env = _env(
        monkeypatch,
        features,
        alpha=1.0,
        coverage_threshold=0.05,
        mean_opacity_threshold=0.95,
        hard_penalty=-2.0,
    )

    env.reset()
    _, reward, _, _, _ = env.step([0.3])

    assert reward == pytest.approx(-2.0)


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
