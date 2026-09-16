import csv
import os

import numpy as np
import pytest

import goals
import transfer
from rl import vis_train
from rl.vis_env import ACTION_SIZE, MAX_STEPS, VisibilityTFEnv


class _StubModel:
    """Same shape as the stub in test_vis_env.py: deterministic and cheap,
    but responsive to every controllable value."""

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


CLASSES_PRESENT = {"stub_a": ["skeleton", "lungs", "organs", "muscle", "vessels"],
                    "stub_b": ["skeleton", "lungs", "organs", "muscle", "vessels"]}


def _patch_totalseg(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: CLASSES_PRESENT[name])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)


def _stub_env(monkeypatch, volume_ids=("stub_a", "stub_b")):
    _patch_totalseg(monkeypatch)
    return VisibilityTFEnv(list(volume_ids), model_for_volume=lambda name: _StubModel())


class _FixedActionModel:
    """A stand-in for an SB3 model: `predict` always returns the same
    deterministic action, so `evaluate_policy_on` can be tested without
    training anything."""

    def __init__(self, action=None):
        self.action = action if action is not None else np.zeros(ACTION_SIZE, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        assert deterministic is True
        return self.action, None


# --- build_env / build_eval_env -------------------------------------------------

def test_build_env_uses_the_training_split(monkeypatch):
    fake_train = ["train_a", "train_b", "train_c"]
    monkeypatch.setattr(vis_train.datasets, "volumes_for_split", lambda split: {
        "train": fake_train, "val": ["val_a"]}[split])

    env = vis_train.build_env()

    assert env.volume_ids == fake_train


def test_build_eval_env_uses_the_validation_split(monkeypatch):
    fake_val = ["val_a", "val_b"]
    monkeypatch.setattr(vis_train.datasets, "volumes_for_split", lambda split: {
        "train": ["train_a"], "val": fake_val}[split])

    env = vis_train.build_eval_env()

    assert env.volume_ids == fake_val


def test_build_env_accepts_an_injected_volume_list(monkeypatch):
    env = vis_train.build_env(volume_ids=["only_this_one"])
    assert env.volume_ids == ["only_this_one"]


# --- evaluate_policy_on -----------------------------------------------------------

def test_evaluate_policy_on_returns_per_episode_attainment(monkeypatch):
    env = _stub_env(monkeypatch, volume_ids=("stub_a",))
    episodes = vis_train.validation_episodes(env, count=5, seed_base=0)
    model = _FixedActionModel()

    results = vis_train.evaluate_policy_on(episodes, model)

    assert len(results) == 5
    for row in results:
        assert "attainment" in row and "kind" in row
        assert isinstance(row["attainment"], float)


def test_evaluate_policy_on_matches_a_manual_rollout(monkeypatch):
    env = _stub_env(monkeypatch, volume_ids=("stub_a",))
    action = np.full(ACTION_SIZE, 0.3, dtype=np.float32)
    model = _FixedActionModel(action=action)

    [result] = vis_train.evaluate_policy_on([(env, 123)], model)

    # Reproduce the same episode by hand.
    env.reset(seed=123)
    info = None
    for _ in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(action)
    assert result["attainment"] == pytest.approx(info["attainment"])
    assert result["kind"] == info["kind"]


def test_summarize_eval_aggregates_overall_and_per_kind():
    results = [
        {"attainment": 0.5, "kind": "relative"},
        {"attainment": 1.0, "kind": "relative"},
        {"attainment": 0.0, "kind": "compound"},
    ]
    summary = vis_train.summarize_eval(results)
    assert summary["mean_attainment"] == pytest.approx(0.5)
    assert summary["by_kind"]["relative"] == pytest.approx(0.75)
    assert summary["by_kind"]["compound"] == pytest.approx(0.0)
    assert summary["by_kind"]["show_only"] is None


# --- run directory guard -----------------------------------------------------------

def test_ensure_run_dir_raises_on_an_existing_directory(tmp_path):
    existing = tmp_path / "run"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        vis_train.ensure_run_dir(str(existing))


def test_ensure_run_dir_creates_a_new_directory(tmp_path):
    fresh = tmp_path / "run"
    vis_train.ensure_run_dir(str(fresh))
    assert fresh.is_dir()


# --- end-to-end smoke run with a stub env -----------------------------------------

def test_smoke_run_writes_eval_progress_csv_with_expected_columns(monkeypatch, tmp_path):
    _patch_totalseg(monkeypatch)
    train_env = VisibilityTFEnv(["stub_a", "stub_b"], model_for_volume=lambda name: _StubModel())
    eval_env = VisibilityTFEnv(["stub_a", "stub_b"], model_for_volume=lambda name: _StubModel())
    out = str(tmp_path / "smoke_run")

    result = vis_train.run_training(
        out=out, timesteps=200, seed=0, eval_interval=100,
        train_env=train_env, eval_env=eval_env, eval_episode_count=5)

    eval_path = result["eval_progress_path"]
    assert os.path.exists(eval_path)
    with open(eval_path, newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == vis_train.CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 2  # 200 timesteps / 100 eval-interval
    assert os.path.exists(result["best_path"])
