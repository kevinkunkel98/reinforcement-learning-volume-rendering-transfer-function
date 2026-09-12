"""Marked slow: exercises a real (tiny) reward-model-driven SAC training
run, including real VTK rendering, to verify eval logging and the gallery
output."""
import json
import os

import pytest
import torch

from rl.reward_model import RewardModel, save_reward_model, train_reward_model
from rl.reward_model_train import reward_model_train


@pytest.mark.parametrize("argument", ["total_timesteps", "eval_interval"])
def test_reward_model_train_rejects_nonpositive_intervals(tmp_path, argument):
    kwargs = {
        "total_timesteps": 1,
        "eval_interval": 1,
        "reward_model_path": str(tmp_path / "missing-model.pt"),
    }
    kwargs[argument] = 0

    with pytest.raises(ValueError, match="greater than zero"):
        reward_model_train(**kwargs)


def _make_tiny_reward_model(tmp_path):
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
    model_path = tmp_path / "reward_model.pt"
    train_reward_model(str(prefs_path), str(model_path), epochs=20)
    return str(model_path)


@pytest.mark.slow
def test_reward_model_train_writes_eval_log_and_gallery(tmp_path, monkeypatch):
    reward_model_path = _make_tiny_reward_model(tmp_path)
    monkeypatch.chdir(tmp_path)

    reward_model_train(total_timesteps=100, eval_interval=50, seed=42,
                        reward_model_path=reward_model_path,
                        model_path="out/rl_models/test_reward_model_agent.zip")

    run_dir = os.path.join("out", "rl_logs", "reward_model_run_seed42")
    eval_csv_path = os.path.join(run_dir, "eval_progress.csv")
    assert os.path.exists(eval_csv_path)
    with open(eval_csv_path, newline="") as f:
        rows = f.readlines()
    assert rows[0].strip() == "timesteps,mean_reward_model_score,mean_automatic_mass_fraction_reward"
    assert len(rows) == 3  # header + 2 checkpoints (100 / 50)

    gallery_dir = os.path.join(run_dir, "gallery")
    assert os.path.exists(os.path.join(gallery_dir, "0_before.png"))
    assert os.path.exists(os.path.join(gallery_dir, "0_after.png"))

    assert os.path.exists("out/rl_models/test_reward_model_agent.zip")


def test_model_parent_dir_is_optional(tmp_path):
    from rl.reward_model_train import _ensure_parent_dir

    _ensure_parent_dir("agent.zip")
    _ensure_parent_dir(str(tmp_path / "nested" / "agent.zip"))


def test_load_reward_models_accepts_aggregate_and_single_checkpoints(tmp_path):
    from rl.reward_model_train import _load_reward_models

    member = tmp_path / "member.pt"
    save_reward_model(RewardModel(), member)
    aggregate = tmp_path / "aggregate.pt"
    torch.save({"member_paths": [str(member)], "seeds": [0]}, aggregate)

    assert len(_load_reward_models(aggregate)) == 1
    assert len(_load_reward_models(member)) == 1
