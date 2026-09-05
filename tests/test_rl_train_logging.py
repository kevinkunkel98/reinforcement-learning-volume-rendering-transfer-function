"""Marked slow: exercises a real (tiny) SAC training run to verify the
custom CSV+tensorboard logger actually gets attached in rl.train.train()."""
import os

import pytest

from rl.train import train


@pytest.mark.slow
def test_train_writes_progress_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train(total_timesteps=200, n_envs=1, seed=42, model_path="out/rl_models/test_agent.zip")
    csv_path = os.path.join("out", "rl_logs", "run_seed42", "progress.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        header = f.readline()
    assert "rollout/ep_rew_mean" in header
    assert "time/total_timesteps" in header
