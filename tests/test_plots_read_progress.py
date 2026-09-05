"""Tests for plots/read_progress.py -- CSV to numpy arrays, no pandas."""
import numpy as np

from plots.read_progress import load_progress


def test_load_progress_parses_columns(tmp_path):
    csv_path = tmp_path / "progress.csv"
    csv_path.write_text(
        "time/total_timesteps,rollout/ep_rew_mean,train/actor_loss\n"
        "100,0.05,\n"
        "200,0.08,-0.91\n"
        "300,0.12,-1.02\n"
    )
    cols = load_progress(str(csv_path))
    assert list(cols["time/total_timesteps"]) == [100.0, 200.0, 300.0]
    assert list(cols["rollout/ep_rew_mean"]) == [0.05, 0.08, 0.12]
    assert np.isnan(cols["train/actor_loss"][0])
    assert cols["train/actor_loss"][1] == -0.91
    assert cols["train/actor_loss"][2] == -1.02


def test_load_progress_all_columns_same_length(tmp_path):
    csv_path = tmp_path / "progress.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")
    cols = load_progress(str(csv_path))
    assert len(cols["a"]) == 3
    assert len(cols["b"]) == 3
