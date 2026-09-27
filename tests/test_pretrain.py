import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from rl import pretrain
from rl.oneshot_env import ACTION_SIZE, OBSERVATION_SIZE, OneShotEnv

V9 = {"policy_version": "oneshot-v9", "action_mode": "residual", "reward_mode": "target"}


def _env(**kwargs):
    return OneShotEnv(["synthetic"], **{**V9, **kwargs})


def test_collect_returns_aligned_hindsight_samples():
    data = pretrain.collect_hindsight(["synthetic"], V9, count=6, seed=0)

    assert data["obs"].shape == (6, OBSERVATION_SIZE)
    assert data["actions"].shape == (6, ACTION_SIZE)
    assert data["rewards"].shape == (6,)
    assert np.all(np.abs(data["actions"]) <= 1.0)


def test_collect_is_deterministic_and_independent_of_workers():
    one = pretrain.collect_hindsight(["synthetic"], V9, count=4, seed=3, workers=1)
    again = pretrain.collect_hindsight(["synthetic"], V9, count=4, seed=3, workers=2)

    for key in ("obs", "actions", "rewards"):
        np.testing.assert_allclose(one[key], again[key])


def test_collected_reward_is_the_oracle_reward():
    data = pretrain.collect_hindsight(["synthetic"], V9, count=3, seed=5)
    env = _env(hindsight_ratio=1.0)
    for index in range(3):
        env.reset(seed=pretrain.episode_seed(5, index))
        _obs, reward, *_ = env.step(env.hindsight_policy_action())
        assert data["rewards"][index] == pytest.approx(reward)


def test_save_and_load_round_trip(tmp_path):
    data = pretrain.collect_hindsight(["synthetic"], V9, count=2, seed=0)
    path = tmp_path / "hindsight.npz"
    pretrain.save_dataset(data, str(path), {"seed": 0})
    loaded, meta = pretrain.load_dataset(str(path))

    np.testing.assert_allclose(loaded["actions"], data["actions"])
    assert meta["seed"] == 0


def _toy_dataset(n=512, seed=0):
    """Targets a fixed linear function of the observation, so a working fit
    must drive the error well below predicting zero."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, OBSERVATION_SIZE)).astype(np.float32)
    weights = rng.normal(scale=0.05, size=(OBSERVATION_SIZE, ACTION_SIZE))
    actions = np.tanh(obs @ weights).astype(np.float32) * 0.5
    return {"obs": obs, "actions": actions, "rewards": np.zeros(n, dtype=np.float32)}


def test_pretrain_actor_fits_the_demonstrations():
    torch.manual_seed(0)
    model = SAC("MlpPolicy", _env(), seed=0)
    data = _toy_dataset()
    baseline = float(np.mean(data["actions"] ** 2))

    report = pretrain.pretrain_actor(model, data, epochs=60, seed=0)

    assert report["val_mse"] < 0.5 * baseline
    predicted, _ = model.predict(data["obs"][:8], deterministic=True)
    assert predicted.shape == (8, ACTION_SIZE)


def test_prefill_replay_buffer_adds_every_transition():
    model = SAC("MlpPolicy", _env(), seed=0)
    data = _toy_dataset(n=10)

    pretrain.prefill_replay_buffer(model, data)

    assert model.replay_buffer.size() == 10
    np.testing.assert_allclose(model.replay_buffer.actions[:10, 0], data["actions"])
    assert np.all(model.replay_buffer.dones[:10] == 1.0)
