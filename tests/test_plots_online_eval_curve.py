"""Tests for plots/online_eval_curve.py."""
import csv
import os

import pytest

from plots.online_eval_curve import _find_latest_run, plot_online_eval_curve


def test_find_latest_run_raises_when_no_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="online_run_seed"):
        _find_latest_run(str(tmp_path))


def test_plot_online_eval_curve_raises_when_eval_csv_missing(tmp_path):
    run_dir = tmp_path / "online_run_seed0"
    run_dir.mkdir()  # no eval_progress.csv inside
    with pytest.raises(FileNotFoundError, match="eval_progress.csv"):
        plot_online_eval_curve(str(run_dir))


def test_plot_online_eval_curve_writes_pdf_and_png(tmp_path):
    run_dir = tmp_path / "online_run_seed0"
    run_dir.mkdir()
    csv_path = run_dir / "eval_progress.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timesteps", "mean_final_mass_fraction", "mean_steps_to_90pct"])
        writer.writerow([1000, 0.10, 8.0])
        writer.writerow([2000, 0.20, 6.0])
        writer.writerow([3000, 0.30, 4.0])

    output_dir = tmp_path / "plots_output"
    base = plot_online_eval_curve(str(run_dir), output_dir=str(output_dir))

    assert os.path.exists(f"{base}.pdf")
    assert os.path.getsize(f"{base}.pdf") > 0
    assert os.path.exists(f"{base}.png")
    assert os.path.getsize(f"{base}.png") > 0
