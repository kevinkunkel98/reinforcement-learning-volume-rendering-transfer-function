import json
import math

import numpy as np
import pytest
import torch

import goals
import policy as policy_module
import transfer
from anatomy import CANONICAL_CLASSES
from rl import model_based as mb
from rl import vis_eval
from rl.oneshot_env import (ACTION_SIZE, OBSERVATION_SIZE, OneShotEnv, build_observation,
                            load_policy_metadata)


class _StubModel:
    """Each goal-class peak's height/colour is that class's visibility/brightness."""

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)

    def features(self, params) -> dict:
        vis, bright = {}, {}
        for goal_class in CANONICAL_CLASSES:
            peak = transfer.peak_internal(params, goals.PEAK_INDEX[goal_class])
            vis[goal_class] = max(float(peak["height"]), 0.0)
            bright[goal_class] = float(sum(peak["rgb"]) / 3.0)
        return {"vis": vis, "bright": bright, "coverage": min(1.0, sum(vis.values()))}

    def solo_max(self, name: str) -> float:
        return 0.5


def _stub_env(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: list(CANONICAL_CLASSES))
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    return OneShotEnv(["fake_a"], model_for_volume=lambda name: _StubModel(), **mb.ENV_KWARGS)


def _random_agg(rng):
    classes = list(goals.GOAL_CLASSES)
    return {"vis": {c: float(rng.uniform(0, 0.3)) for c in classes + ["other"]},
            "bright": {c: float(rng.uniform(0, 1)) for c in classes},
            "coverage": float(rng.uniform(0, 1))}


def test_goal_distance_matches_goals_distance():
    rng = np.random.default_rng(0)
    for _ in range(50):
        start, current = _random_agg(rng), _random_agg(rng)
        targets = {}
        for c in rng.choice(goals.GOAL_CLASSES, rng.integers(1, 4), replace=False):
            targets[str(c)] = ({"vis": float(rng.uniform(-1, 1))} if rng.random() < 0.6
                               else {"bright": float(rng.uniform(-0.4, 0.4))})
        goal = goals.goal_vector(targets)
        weights = {c: float(rng.uniform(0.5, 2)) for c in goals.GOAL_CLASSES} if rng.random() < 0.5 else None
        expected = goals.distance(goal, start, current, **({"weights": weights} if weights else {}))
        w = torch.tensor([[(weights or {}).get(c, 1.0) for c in goals.GOAL_CLASSES]])
        delta = torch.as_tensor(mb.state_vector(current) - mb.state_vector(start))[None]
        got = mb.goal_distance(torch.as_tensor(goal, dtype=torch.float32)[None], w, delta)
        assert float(got) == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_policy_observation_matches_build_observation():
    rng = np.random.default_rng(1)
    agg = _random_agg(rng)
    goal = goals.goal_vector({"heart": {"vis": 0.3}})
    histogram = rng.uniform(0, 1, 16)
    solo = rng.uniform(-3, 0, len(goals.GOAL_CLASSES))
    controllable = rng.uniform(-1, 1, ACTION_SIZE)
    expected = build_observation(goal, histogram, agg, solo, controllable)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]
    got = mb.policy_observation(T(goal), T(histogram), T(mb.state_vector(agg)), T(solo), T(controllable))
    np.testing.assert_allclose(got[0].numpy(), expected, rtol=1e-5, atol=1e-5)


def test_policy_predicts_bounded_actions_and_round_trips(tmp_path):
    torch.manual_seed(0)
    policy = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net(), {"seed": 3})
    obs = np.random.default_rng(2).normal(size=OBSERVATION_SIZE).astype(np.float32)
    action, state = policy.predict(obs, deterministic=True)
    assert state is None and action.shape == (ACTION_SIZE,)
    assert np.all(np.abs(action) <= 1.0)
    path = policy.save(str(tmp_path / "run"), mb.DynamicsEnsemble())
    loaded = mb.ModelBasedPolicy.load(path)
    np.testing.assert_allclose(loaded.predict(obs)[0], action, rtol=1e-6)
    assert loaded.metadata == {"seed": 3}
    assert isinstance(mb.load_dynamics(path), mb.DynamicsEnsemble)


def test_checkpoint_metadata_declares_the_v9_residual_target_contract(tmp_path):
    path = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net()).save(str(tmp_path / "run"))
    metadata = load_policy_metadata(path)
    assert metadata == {"policy_version": "oneshot-v9", "action_mode": "residual", "reward_mode": "target"}
    assert json.loads((tmp_path / "run" / mb.METADATA_NAME).read_text())["algorithm"] == "model_based"


def test_viewer_and_vis_eval_load_a_model_based_checkpoint(tmp_path, monkeypatch):
    path = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net()).save(str(tmp_path / "run"))
    monkeypatch.setattr(policy_module, "POLICY_PATH", path)
    policy_module.reset_cache()
    try:
        assert isinstance(policy_module.load_policy(), mb.ModelBasedPolicy)
    finally:
        policy_module.reset_cache()
    assert isinstance(vis_eval._load_sac(path), mb.ModelBasedPolicy)


def test_collect_records_the_rendered_change_of_each_action(monkeypatch):
    env = _stub_env(monkeypatch)
    data = mb.collect_starts(env, seeds=[0, 1], actions_per_start=4)
    assert data["act"].shape == (2, 4, ACTION_SIZE)
    assert data["s1"].shape == (2, 4, mb.STATE_SIZE)
    assert data["goal"].shape == (2, mb.INSTRUCTIONS_PER_START, 4 * len(goals.GOAL_CLASSES))
    # replaying a recorded action reproduces its recorded state
    env.reset(seed=1)
    ctrl = np.asarray(env._controllable_values(env._start_params))
    params = mb.apply_controllable(env._start_params, ctrl + data["act"][1, 2])
    replay = mb.state_vector(goals.aggregate(goals.features(env._model, params)))
    np.testing.assert_allclose(replay, data["s1"][1, 2], rtol=1e-5, atol=1e-6)


def test_policy_training_improves_model_predicted_attainment(monkeypatch):
    env = _stub_env(monkeypatch)
    data = mb.collect_starts(env, seeds=range(40), actions_per_start=10)
    train, val = mb.split_starts(len(data["s0"]))
    ensemble, _ = mb.train_dynamics(data, train, val, epochs=15, seed=0, log=lambda *_: None)
    policy, report = mb.train_policy(ensemble, data, train, steps=1000, seed=0, log=lambda *_: None)
    assert report["model_predicted_median"] > 0.0


def test_evaluate_scores_real_renders(monkeypatch):
    _stub_env(monkeypatch)
    policy = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net())
    result = mb.evaluate(policy, episodes=3, volume_ids=["fake_a"], model_for_volume=lambda name: _StubModel())
    assert len(result["rows"]) == 3
    assert all(math.isfinite(r["attainment"]) for r in result["rows"])


def test_policy_architecture_round_trips(tmp_path):
    policy = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net(hidden=64, depth=3))
    path = policy.save(str(tmp_path / "run"))
    loaded = mb.ModelBasedPolicy.load(path)
    assert loaded.net[0].out_features == 64
    assert (len(loaded.net) - 1) // 2 == 3


def test_training_can_reuse_a_saved_dynamics_ensemble(monkeypatch, tmp_path):
    env = _stub_env(monkeypatch)
    data = mb.collect_starts(env, seeds=range(12), actions_per_start=6)
    data_path = tmp_path / "data.npz"
    np.savez_compressed(data_path, **data)
    first = mb.run_training([str(data_path)], str(tmp_path / "first"), model_epochs=2, policy_steps=10,
                            eval_episodes=0, log=lambda *_: None)
    second = mb.run_training([str(data_path)], str(tmp_path / "second"), policy_steps=10, eval_episodes=0,
                             dynamics_path=first["path"], log=lambda *_: None)
    assert second["report"]["dynamics"]["reused"] == first["path"]
    for a, b in zip(first["dynamics"].state_dict().values(), mb.load_dynamics(second["path"]).state_dict().values()):
        assert torch.equal(a, b)
