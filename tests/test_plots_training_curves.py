"""Tests for plots/training_curves.py -- renders real figure files without
inspecting pixel content (a bad column name or bad matplotlib call raises;
this test only needs to catch that)."""
import os

from plots.training_curves import plot_training_curves


def _write_fake_progress_csv(path):
    path.write_text(
        "time/total_timesteps,rollout/ep_rew_mean,train/actor_loss,train/critic_loss,train/ent_coef\n"
        "1000,0.02,-0.5,1e-3,0.5\n"
        "2000,0.05,-0.6,5e-4,0.3\n"
        "3000,0.09,-0.8,1e-4,0.1\n"
        "4000,0.12,-1.0,5e-5,0.05\n"
    )


def test_plot_training_curves_writes_pdf_and_png(tmp_path):
    run_dir = tmp_path / "run_seed0"
    run_dir.mkdir()
    _write_fake_progress_csv(run_dir / "progress.csv")
    output_dir = tmp_path / "output"

    base = plot_training_curves(str(run_dir), output_dir=str(output_dir))

    pdf_path = base + ".pdf"
    png_path = base + ".png"
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
    assert os.path.exists(png_path)
    assert os.path.getsize(png_path) > 0
