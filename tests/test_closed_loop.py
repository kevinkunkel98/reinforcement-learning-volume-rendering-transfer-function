import json
import math

import numpy as np
import pytest
import torch

import goals
from rl import closed_loop as cl
from rl import model_based as mb
from tests.test_model_based import _StubModel, _stub_env

N = len(goals.GOAL_CLASSES)


def _goal(targets):
    return torch.as_tensor(goals.goal_vector(targets), dtype=torch.float32)[None]


def _delta(**changes):
    delta = torch.zeros(1, mb.STATE_SIZE)
    for name, value in changes.items():
        delta[0, goals.GOAL_CLASSES.index(name)] = value
    return delta


def test_remaining_goal_subtracts_progress_and_adds_undo_targets():
    goal = _goal({"heart": {"vis": 0.6}})
    np.testing.assert_allclose(cl.remaining_goal(goal, torch.zeros(1, mb.STATE_SIZE)), goal)
    heart, liver = goals.GOAL_CLASSES.index("heart"), goals.GOAL_CLASSES.index("liver")
    left = cl.remaining_goal(goal, _delta(heart=0.4, liver=0.5))[0]
    assert float(left[heart]) == pytest.approx(0.2)
    assert float(left[liver]) == pytest.approx(-0.5) and float(left[N + liver]) == 1.0
    small = cl.remaining_goal(goal, _delta(liver=goals.KEEP_TOLERANCE / 2))[0]
    assert float(small[liver]) == 0.0 and float(small[N + liver]) == 0.0


def test_remaining_goal_leaves_v2_occluders_free():
    goal = _goal({"heart": {"vis": 0.6}})
    soft = goals.GOAL_CLASSES.index("soft")
    weights = torch.ones(1, N + 1)
    weights[0, soft] = 0.0
    left = cl.remaining_goal(goal, _delta(soft=-1.0), weights)[0]
    assert float(left[soft]) == 0.0


def test_remaining_goal_for_one_episode_matches_the_batched_version():
    rng = np.random.default_rng(0)
    classes = list(goals.GOAL_CLASSES)
    agg = lambda: {"vis": {c: float(rng.uniform(0, 0.3)) for c in classes + ["other"]},
                   "bright": {c: float(rng.uniform(0, 1)) for c in classes}, "coverage": 0.5}
    start, current = agg(), agg()
    goal = goals.goal_vector({"kidneys": {"vis": 0.3}, "liver": {"bright": 0.2}})
    delta = torch.as_tensor(mb.state_vector(current) - mb.state_vector(start))[None]
    expected = cl.remaining_goal(torch.as_tensor(goal, dtype=torch.float32)[None], delta)[0].numpy()
    np.testing.assert_allclose(cl.remaining_goal_for(goal, start, current), expected, rtol=1e-6)


def test_finetune_trains_through_a_multi_step_rollout(monkeypatch):
    env = _stub_env(monkeypatch)
    data = mb.collect_starts(env, seeds=range(20), actions_per_start=6)
    train, val = mb.split_starts(len(data["s0"]))
    ensemble, _ = mb.train_dynamics(data, train, val, epochs=3, seed=0, log=lambda *_: None)
    policy = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net())
    before = [p.detach().clone() for p in policy.net.parameters()]
    _, report = cl.finetune(policy, ensemble, data, train, steps=1000, horizon=2, batch=16, log=lambda *_: None)
    assert math.isfinite(report["model_predicted_median"])
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.net.parameters()))
    assert not policy.net.training


def test_first_closed_loop_step_is_the_one_shot_score(monkeypatch):
    _stub_env(monkeypatch)
    torch.manual_seed(0)
    policy = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net())
    kw = {"episodes": 4, "volume_ids": ["fake_a"], "model_for_volume": lambda name: _StubModel()}
    looped = cl.evaluate(policy, horizon=2, **kw)
    one_shot = mb.evaluate(policy, **kw)
    assert [r["t1"] for r in looped["rows"]] == pytest.approx([r["attainment"] for r in one_shot["rows"]])
    assert all(math.isfinite(r["t2"]) for r in looped["rows"])
    assert set(looped["medians"]["ALL"]) == {"t1", "t2"}


def test_cli_writes_a_checkpoint_the_viewer_can_load(monkeypatch, tmp_path):
    env = _stub_env(monkeypatch)
    data = mb.collect_starts(env, seeds=range(12), actions_per_start=4)
    np.savez_compressed(tmp_path / "data.npz", **data)
    init = mb.ModelBasedPolicy(mb.ModelBasedPolicy.new_net()).save(str(tmp_path / "init"), mb.DynamicsEnsemble())
    cl.main(["--init", init, "--data", str(tmp_path / "data.npz"), "--out", str(tmp_path / "out"),
             "--steps", "3", "--eval-episodes", "0", "--threads", "1", "--task-version", "v2"])
    loaded = mb.ModelBasedPolicy.load(str(tmp_path / "out" / mb.CHECKPOINT_NAME))
    assert loaded.metadata["closed_loop_horizon"] == 2 and loaded.metadata["task_version"] == "v2"
    assert json.loads((tmp_path / "out" / mb.METADATA_NAME).read_text())["task_version"] == "v2"
