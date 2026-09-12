"""Fast tests for rl/reward_model.py -- no rendering, no real preference data."""
import json

import numpy as np
import pytest

import torch
import torch.nn as nn

from rl.reward_model import (
    INPUT_DIM,
    RewardModel,
    bradley_terry_loss,
    encode_target,
    ensemble_reward,
    featurize,
    load_reward_model,
    predict_reward,
    save_reward_model,
    train_reward_model,
    _pair_arrays,
)


def test_featurize_contains_after_before_delta_goal_and_direction():
    before = {"mean": 10.0, "std": 5.0, "coverage": 0.2, "entropy": 1.0}
    after = {"mean": 15.0, "std": 5.0, "coverage": 0.3, "entropy": 1.2}
    x = featurize(before, after, "bone", "increase")
    assert x.shape == (18,)
    assert x.dtype == np.float32
    np.testing.assert_allclose(x[:4], [15.0, 5.0, 0.3, 1.2])
    np.testing.assert_allclose(x[4:8], [10.0, 5.0, 0.2, 1.0])
    np.testing.assert_allclose(x[8:12], [5.0, 0.0, 0.1, 0.2])
    np.testing.assert_allclose(x[12:17], [0.0, 0.0, 0.0, 0.0, 1.0])
    assert x[17] == 1.0


def test_featurize_decrease_direction_flag():
    before = {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0}
    after = {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0}
    x = featurize(before, after, "air", "decrease")
    assert x[17] == -1.0
    assert list(x[12:17]) == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_encode_target_accepts_command_dict():
    np.testing.assert_allclose(encode_target({"target_tissue": "bone", "direction": "increase"}),
                               [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize("cmd", [
    {"target_tissue": "unknown", "direction": "increase"},
    {"target_tissue": "bone", "direction": "sideways"},
])
def test_encode_target_rejects_invalid_goal(cmd):
    with pytest.raises(ValueError, match="target|direction"):
        encode_target(cmd)


def test_reward_model_has_two_hidden_layers():
    layers = list(RewardModel().net)
    assert [layer.out_features for layer in layers if isinstance(layer, nn.Linear)] == [64, 32, 1]


def test_weighted_bradley_terry_loss_scales_examples():
    preferred = torch.tensor([2.0, 0.0])
    other = torch.tensor([0.0, 0.0])
    low = bradley_terry_loss(preferred, other, torch.tensor([1.0, 1.0]))
    high = bradley_terry_loss(preferred, other, torch.tensor([1.0, 3.0]))
    assert high > low


def test_pair_arrays_parses_canonical_observations_and_swaps_negative_label():
    before = {"mean": 10.0, "std": 1.0, "coverage": 0.1, "entropy": 0.2}
    after_a = {"mean": 20.0, "std": 2.0, "coverage": 0.2, "entropy": 0.3}
    after_b = {"mean": 30.0, "std": 3.0, "coverage": 0.3, "entropy": 0.4}
    record = {
        "target_tissue": "bone",
        "direction": "increase",
        "observation_a": {"before_features": before, "after_features": after_a},
        "observation_b": {"before_features": before, "after_features": after_b},
        "label": -1,
    }

    preferred, other, weights = _pair_arrays([record])

    np.testing.assert_allclose(preferred[0, :4].numpy(), [30.0, 3.0, 0.3, 0.4])
    np.testing.assert_allclose(other[0, :4].numpy(), [20.0, 2.0, 0.2, 0.3])
    np.testing.assert_allclose(weights.numpy(), [1.0])
    assert not torch.equal(preferred, other)
    assert torch.count_nonzero(preferred - other).item() > 0


def test_pair_arrays_rejects_canonical_labels_other_than_plus_or_minus_one():
    features = {"mean": 1.0, "std": 1.0, "coverage": 1.0, "entropy": 1.0}
    record = {
        "target_tissue": "bone", "direction": "increase", "label": 0,
        "observation_a": {"before_features": features, "after_features": features},
        "observation_b": {"before_features": features, "after_features": features},
    }
    with pytest.raises(ValueError, match="label"):
        _pair_arrays([record])


def test_featurize_rejects_non_finite_features():
    before = {"mean": 1.0, "std": 1.0, "coverage": 1.0, "entropy": 1.0}
    after = {**before, "mean": float("inf")}
    with pytest.raises(ValueError, match="finite"):
        featurize(before, after, "bone", "increase")


@pytest.mark.parametrize("weight", [0.0, -1.0, np.nan, np.inf])
def test_pair_arrays_rejects_non_positive_or_non_finite_weight(weight):
    record = {
        "target_tissue": "bone",
        "direction": "increase",
        "before_features": {"mean": 1.0, "std": 1.0, "coverage": 1.0, "entropy": 1.0},
        "after_features": {"mean": 2.0, "std": 1.0, "coverage": 1.0, "entropy": 1.0},
        "label": 1,
        "weight": weight,
    }
    with pytest.raises(ValueError, match="weight"):
        _pair_arrays([record])


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


def test_train_reward_model_rejects_single_record(tmp_path):
    prefs_path = tmp_path / "preferences.jsonl"
    _make_separable_preferences(prefs_path, n=1)
    with pytest.raises(ValueError, match="at least two"):
        train_reward_model(str(prefs_path), str(tmp_path / "reward_model.pt"), epochs=1)


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


def test_checkpoint_round_trip_preserves_prediction(tmp_path):
    path = tmp_path / "model.pt"
    model = RewardModel()
    save_reward_model(model, path)
    loaded = load_reward_model(str(path))
    x = torch.zeros(1, INPUT_DIM)
    assert loaded(x).item() == pytest.approx(model(x).item())


def test_load_reward_model_rejects_unversioned_legacy_checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(RewardModel().state_dict(), path)
    with pytest.raises(ValueError, match="legacy|version"):
        load_reward_model(str(path))


def test_load_reward_model_rejects_unknown_checkpoint_version(tmp_path):
    path = tmp_path / "future.pt"
    torch.save({
        "format": "goal-conditioned-reward",
        "version": 999,
        "state_dict": RewardModel().state_dict(),
    }, path)
    with pytest.raises(ValueError, match="version"):
        load_reward_model(str(path))


def test_ensemble_reward_is_mean_minus_std():
    values = np.array([[.2, .4], [.6, .2]])
    assert ensemble_reward(values) == pytest.approx(values.mean() - values.std())


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-np.inf]])
def test_ensemble_reward_rejects_empty_or_nonfinite_values(values):
    with pytest.raises(ValueError, match="non-empty|finite"):
        ensemble_reward(values)


def test_predict_reward_stays_finite_for_extreme_logit():
    class ExtremeModel:
        def __call__(self, x):
            return torch.tensor([1000.0])

    features = {"mean": 1.0, "std": 1.0, "coverage": 1.0, "entropy": 1.0}
    reward = predict_reward(ExtremeModel(), features, features, "bone", "increase")
    assert reward == pytest.approx(1.0)
    assert np.isfinite(reward)
