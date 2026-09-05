"""Tests for plots/training_curves.py -- renders real figure files without
inspecting pixel content (a bad column name or bad matplotlib call raises;
this test only needs to catch that)."""
import os
import time

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


def test_find_latest_run_uses_modification_time_not_lexicographic_order(tmp_path):
    from plots.training_curves import _find_latest_run

    log_dir = tmp_path / "rl_logs"
    log_dir.mkdir()
    older = log_dir / "run_seed10"
    older.mkdir()
    newer = log_dir / "run_seed2"
    newer.mkdir()

    # Note: as strings, "run_seed10" < "run_seed2" (since "1" < "2" at the
    # first differing character), so a lexicographic sort happens to place
    # "run_seed2" last here too -- this particular pair doesn't demonstrate
    # the old bug on its own. The point of this test is simply that
    # _find_latest_run must pick by actual mtime, not by name, so we set
    # the mtimes explicitly and assert the newer one wins regardless of
    # what the names look like.
    os.utime(older, (time.time() - 100, time.time() - 100))
    os.utime(newer, (time.time(), time.time()))

    result = _find_latest_run(str(log_dir))
    assert result == str(newer)
