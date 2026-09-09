"""Marked slow: exercises a real (tiny) online SAC training run to verify
the held-out eval log and SB3 progress log both get written correctly."""
import csv
import os

import pytest

from rl.online_train import online_train


@pytest.mark.slow
def test_online_train_writes_eval_and_progress_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    online_train(total_timesteps=200, eval_interval=100, seed=42,
                  model_path="out/rl_models/test_online_agent.zip")

    run_dir = os.path.join("out", "rl_logs", "online_run_seed42")

    eval_csv_path = os.path.join(run_dir, "eval_progress.csv")
    assert os.path.exists(eval_csv_path)
    with open(eval_csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timesteps", "mean_final_mass_fraction", "mean_steps_to_90pct"]
    assert len(rows) == 3  # header + 2 checkpoints (200 / 100)
    assert rows[1][0] == "100"
    assert rows[2][0] == "200"

    progress_csv_path = os.path.join(run_dir, "progress.csv")
    assert os.path.exists(progress_csv_path)

    model_path = os.path.join("out", "rl_models", "test_online_agent.zip")
    assert os.path.exists(model_path)
