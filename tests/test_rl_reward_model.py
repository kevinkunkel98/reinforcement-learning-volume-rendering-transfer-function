"""Fast tests for rl/reward_model.py -- no rendering, no real preference data."""
import json

import numpy as np
import pytest

from rl.reward_model import INPUT_DIM, featurize, load_reward_model, predict_reward, train_reward_model


def test_featurize_shape_and_values():
    before = {"mean": 10.0, "std": 5.0, "coverage": 0.2, "entropy": 1.0}
    after = {"mean": 15.0, "std": 5.0, "coverage": 0.3, "entropy": 1.2}
    x = featurize(before, after, "bone", "increase")
    assert x.shape == (INPUT_DIM,)
    assert x.dtype == np.float32
    assert x[0] == pytest.approx(5.0)
    assert x[1] == pytest.approx(0.0)
    assert x[2] == pytest.approx(0.1)
    assert x[3] == pytest.approx(0.2)
    # tissue one-hot order is rl.env.TISSUES = ["air", "fat", "soft", "spongy", "bone"]
    assert list(x[4:9]) == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert x[9] == 1.0


def test_featurize_decrease_direction_flag():
    before = {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0}
    after = {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0}
    x = featurize(before, after, "air", "decrease")
    assert x[9] == -1.0
    assert list(x[4:9]) == [1.0, 0.0, 0.0, 0.0, 0.0]


def _make_separable_preferences(path, n=40):
    """Trivially separable synthetic data: label=1 whenever mean went up,
    label=-1 whenever it went down. The reward model should learn this easily
    -- this is a correctness check on the training loop, not a claim about
    real preference data."""
    rng = np.random.default_rng(0)
    with open(path, "w") as f:
        for i in range(n):
            delta_mean = float(rng.uniform(1.0, 10.0)) * (1 if i % 2 == 0 else -1)
            before = {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            after = {"mean": 50.0 + delta_mean, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            record = {
                "timestamp": "2026-01-01T00:00:00", "rater_id": "default",
                "target_tissue": "bone", "direction": "increase", "delta": 0.1,
                "before_features": before, "after_features": after,
                "label": 1 if delta_mean > 0 else -1,
            }
            f.write(json.dumps(record) + "\n")


def test_train_reward_model_learns_separable_data(tmp_path):
    prefs_path = tmp_path / "preferences.jsonl"
    _make_separable_preferences(prefs_path)
    model_path = tmp_path / "reward_model.pt"

    model = train_reward_model(str(prefs_path), str(model_path), epochs=300, seed=0)

    assert model_path.exists()
    better = predict_reward(model, {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                             {"mean": 60.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                             "bone", "increase")
    worse = predict_reward(model, {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                            {"mean": 40.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                            "bone", "increase")
    assert better > worse


def test_load_reward_model_roundtrip(tmp_path):
    prefs_path = tmp_path / "preferences.jsonl"
    _make_separable_preferences(prefs_path)
    model_path = tmp_path / "reward_model.pt"
    train_reward_model(str(prefs_path), str(model_path), epochs=50, seed=0)

    loaded = load_reward_model(str(model_path))
    r = predict_reward(loaded, {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                        {"mean": 60.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                        "bone", "increase")
    assert -1.0 <= r <= 1.0
