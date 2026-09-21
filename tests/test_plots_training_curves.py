"""Tests for plots/training_curves.py -- renders real figure files without
inspecting pixel content (a bad column name or bad matplotlib call raises;
this test only needs to catch that)."""
import os
import time

import pytest

from plots.training_curves import plot_training_curves


def _write_fake_progress_csv(path):
    path.write_text(
        "time/total_timesteps,rollout/ep_rew_mean,train/actor_loss,train/critic_loss,train/ent_coef\n"
        "1000,0.02,-0.5,1e-3,0.5\n"
        "2000,0.05,-0.6,5e-4,0.3\n"
        "3000,0.09,-0.8,1e-4,0.1\n"
        "4000,0.12,-1.0,5e-5,0.05\n"
    )


def _write_fake_eval_progress_csv(path, values):
    path.write_text(
        "timesteps,median_attainment,share_positive,"
        "attainment_relative,attainment_compound,attainment_show_only,"
        "attainment_absolute,attainment_brightness\n"
        + "\n".join(
            f"{step},{median},{share},0.1,0.2,0.3,0.4,0.5"
            for step, median, share in values
        )
        + "\n"
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
    older = log_dir / "run_seed9"
    older.mkdir()
    newer = log_dir / "run_seed10"
    newer.mkdir()

    # Verified: sorted(["run_seed9", "run_seed10"]) == ["run_seed10", "run_seed9"]
    # (string comparison hits the differing character "1" vs "9" right after
    # the shared "run_seed" prefix, and "1" < "9"), so "run_seed9" sorts LAST
    # lexicographically. We give the lexicographically-last directory
    # ("run_seed9") the OLDER mtime and the other one ("run_seed10") the
    # NEWER mtime. That means the old buggy implementation
    # (`sorted(candidates)[-1]`) would return "run_seed9", while the correct
    # mtime-based implementation (`max(candidates, key=os.path.getmtime)`)
    # returns "run_seed10" -- the two implementations disagree on this input,
    # so this test actually catches a regression to lexicographic sorting.
    os.utime(older, (time.time() - 100, time.time() - 100))
    os.utime(newer, (time.time(), time.time()))

    result = _find_latest_run(str(log_dir), run_glob="run_seed*")
    assert result == str(newer)


def test_find_latest_run_defaults_to_the_one_shot_runs_not_the_retired_ten_step_logs(tmp_path):
    """The reported pipeline lives in out/rl_v2/oneshot_v3_seed*; out/rl_logs
    holds the retired ten-step runs. Defaulting to the latter once put a figure
    of the wrong experiment in front of a reader."""
    from plots import training_curves

    log_dir = tmp_path / "rl_v2"
    log_dir.mkdir()
    (log_dir / "oneshot_v3_seed0").mkdir()
    (log_dir / "run_seed1").mkdir()      # a retired ten-step run sitting alongside

    assert training_curves.LOG_DIR == "out/rl_v2"
    assert _basename(training_curves._find_latest_run(str(log_dir))) == "oneshot_v3_seed0"


def test_load_validation_run_selects_best_median_and_final_row(tmp_path):
    from plots.training_curves import load_validation_run

    run_dir = tmp_path / "oneshot_v3_seed0"
    run_dir.mkdir()
    _write_fake_eval_progress_csv(
        run_dir / "eval_progress.csv",
        [(10000, 0.1, 0.5), (20000, 0.4, 0.8), (30000, 0.3, 0.7)],
    )

    result = load_validation_run(str(run_dir))

    assert result["seed"] == 0
    assert result["best_median"] == 0.4
    assert result["best_timestep"] == 20000
    assert result["final_median"] == 0.3
    assert result["final_share_positive"] == 0.7


def test_load_validation_runs_are_sorted_by_seed(tmp_path):
    from plots.training_curves import load_validation_runs

    for seed in (2, 10, 1):
        run_dir = tmp_path / f"oneshot_v3_seed{seed}"
        run_dir.mkdir()
        _write_fake_eval_progress_csv(
            run_dir / "eval_progress.csv", [(10000, seed / 10, 0.5)]
        )

    runs = load_validation_runs(str(tmp_path), "oneshot_v3_seed*")

    assert [run["seed"] for run in runs] == [1, 2, 10]


def test_load_validation_runs_rejects_non_numeric_seed_directory(tmp_path):
    from plots.training_curves import load_validation_runs

    run_dir = tmp_path / "oneshot_v3_seedunknown"
    run_dir.mkdir()
    _write_fake_eval_progress_csv(run_dir / "eval_progress.csv", [(10000, 0.4, 0.5)])

    with pytest.raises(ValueError) as exc_info:
        load_validation_runs(str(tmp_path), "oneshot_v3_seed*")

    message = str(exc_info.value)
    assert str(run_dir) in message
    assert "numeric seed" in message


@pytest.mark.parametrize(
    "missing_column", ["timesteps", "median_attainment", "share_positive"]
)
def test_load_validation_run_reports_each_missing_required_column(tmp_path, missing_column):
    from plots.training_curves import load_validation_run

    run_dir = tmp_path / "oneshot_v3_seed0"
    run_dir.mkdir()
    columns = ["timesteps", "median_attainment", "share_positive"]
    columns.remove(missing_column)
    (run_dir / "eval_progress.csv").write_text(
        ",".join(columns) + "\n" + ",".join("0.2" for _ in columns) + "\n"
    )

    with pytest.raises(ValueError) as exc_info:
        load_validation_run(str(run_dir))
    message = str(exc_info.value)
    assert str(run_dir) in message
    assert missing_column in message


def test_load_validation_run_rejects_nonfinite_required_column_data(tmp_path):
    from plots.training_curves import load_validation_run

    run_dir = tmp_path / "oneshot_v3_seed0"
    run_dir.mkdir()
    (run_dir / "eval_progress.csv").write_text(
        "timesteps,median_attainment,share_positive\n"
        "10000,nan,0.5\n"
    )

    with pytest.raises(ValueError) as exc_info:
        load_validation_run(str(run_dir))

    message = str(exc_info.value)
    assert str(run_dir) in message
    assert "median_attainment" in message


def test_load_validation_run_uses_last_finite_summary_row(tmp_path):
    from plots.training_curves import load_validation_run

    run_dir = tmp_path / "oneshot_v3_seed0"
    run_dir.mkdir()
    (run_dir / "eval_progress.csv").write_text(
        "timesteps,median_attainment,share_positive\n"
        "10000,0.2,0.5\n20000,0.4,0.8\n30000,nan,nan\n"
    )

    result = load_validation_run(str(run_dir))

    assert result["final_median"] == 0.4
    assert result["final_share_positive"] == 0.8


def test_load_validation_run_selects_final_metrics_independently(tmp_path):
    from plots.training_curves import load_validation_run

    run_dir = tmp_path / "oneshot_v3_seed0"
    run_dir.mkdir()
    (run_dir / "eval_progress.csv").write_text(
        "timesteps,median_attainment,share_positive\n"
        "10000,0.2,0.5\n20000,0.4,nan\n30000,nan,0.8\n"
    )

    result = load_validation_run(str(run_dir))

    assert result["final_median"] == 0.4
    assert result["final_share_positive"] == 0.8


def test_load_validation_runs_exposes_seed_spread(tmp_path):
    from plots.training_curves import load_validation_runs

    for seed, best_median in ((0, 0.2), (1, 0.7), (2, 0.4)):
        run_dir = tmp_path / f"oneshot_v3_seed{seed}"
        run_dir.mkdir()
        _write_fake_eval_progress_csv(
            run_dir / "eval_progress.csv", [(10000, best_median, 0.5)]
        )

    runs = load_validation_runs(str(tmp_path), "oneshot_v3_seed*")

    assert [run["seed_spread"] for run in runs] == pytest.approx([0.5, 0.5, 0.5])


def test_plot_validation_runs_writes_thesis_exports(tmp_path):
    from plots.training_curves import plot_validation_runs

    for seed in range(3):
        run_dir = tmp_path / f"oneshot_v3_seed{seed}"
        run_dir.mkdir()
        _write_fake_eval_progress_csv(
            run_dir / "eval_progress.csv",
            [(10000, 0.1 + seed / 100, 0.5), (20000, 0.3 + seed / 100, 0.8)],
        )

    base = plot_validation_runs(str(tmp_path), output_dir=str(tmp_path / "output"))

    assert os.path.exists(base + ".pdf")
    assert os.path.getsize(base + ".pdf") > 0
    assert os.path.exists(base + ".png")
    assert os.path.getsize(base + ".png") > 0


def test_plot_validation_runs_accepts_required_columns_without_optional_data(tmp_path):
    from plots.training_curves import plot_validation_runs

    for seed in range(3):
        run_dir = tmp_path / f"oneshot_v3_seed{seed}"
        run_dir.mkdir()
        (run_dir / "eval_progress.csv").write_text(
            "timesteps,median_attainment,share_positive\n"
            f"10000,{0.2 + seed / 100},0.5\n"
            f"20000,{0.4 + seed / 100},0.8\n"
        )

    base = plot_validation_runs(str(tmp_path), output_dir=str(tmp_path / "output"))

    assert os.path.getsize(base + ".pdf") > 0
    assert os.path.getsize(base + ".png") > 0


def test_load_validation_run_wraps_missing_eval_progress_error(tmp_path):
    from plots.training_curves import load_validation_run

    run_dir = tmp_path / "oneshot_v3_seed0"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_validation_run(str(run_dir))

    message = str(exc_info.value)
    assert str(run_dir) in message
    assert "eval_progress.csv" in message
    assert "timesteps" in message
    assert "median_attainment" in message
    assert "share_positive" in message


def test_load_validation_runs_reports_search_when_no_runs_match(tmp_path):
    from plots.training_curves import load_validation_runs

    with pytest.raises(FileNotFoundError) as exc_info:
        load_validation_runs(str(tmp_path), "oneshot_v3_seed*")

    message = str(exc_info.value)
    assert str(tmp_path) in message
    assert "oneshot_v3_seed*" in message


def test_parse_args_defaults_to_thesis_mode():
    from plots.training_curves import LOG_DIR, OUTPUT_DIR, RUN_GLOB, parse_args

    args = parse_args([])

    assert args.run_dir is None
    assert args.log_dir == LOG_DIR
    assert args.run_glob == RUN_GLOB
    assert args.out == OUTPUT_DIR


def test_parse_args_run_dir_selects_single_run_mode_and_preserves_out():
    from plots.training_curves import parse_args

    args = parse_args(["--run-dir", "/tmp/run", "--out", "/tmp/output"])

    assert args.run_dir == "/tmp/run"
    assert args.out == "/tmp/output"


def test_main_defaults_to_validation_runs_and_forwards_discovery_options(monkeypatch, capsys):
    from plots import training_curves

    calls = []

    def fake_plot_validation_runs(log_dir, run_glob, output_dir):
        calls.append((log_dir, run_glob, output_dir))
        return "/tmp/output/training_curves_thesis"

    monkeypatch.setattr(training_curves, "plot_validation_runs", fake_plot_validation_runs)

    training_curves.main(
        [
            "--log-dir", "/tmp/logs",
            "--run-glob", "seed*",
            "--out", "/tmp/output",
        ]
    )

    assert calls == [("/tmp/logs", "seed*", "/tmp/output")]
    assert capsys.readouterr().out == (
        "Wrote /tmp/output/training_curves_thesis.pdf and "
        "/tmp/output/training_curves_thesis.png\n"
    )


def test_main_with_run_dir_preserves_single_run_renderer(monkeypatch, capsys):
    from plots import training_curves

    calls = []

    def fake_plot_training_curves(run_dir, output_dir):
        calls.append((run_dir, output_dir))
        return "/tmp/output/training_curves"

    monkeypatch.setattr(training_curves, "plot_training_curves", fake_plot_training_curves)

    training_curves.main(["--run-dir", "/tmp/run", "--out", "/tmp/output"])

    assert calls == [("/tmp/run", "/tmp/output")]
    assert capsys.readouterr().out == (
        "Wrote /tmp/output/training_curves.pdf and "
        "/tmp/output/training_curves.png\n"
    )


def _basename(path):
    return os.path.basename(path.rstrip("/"))
