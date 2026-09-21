# Thesis-Clean Training Curves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thesis-clean multi-seed validation figure with selected-checkpoint summaries while preserving the existing single-run SAC diagnostics plot.

**Architecture:** Keep CSV parsing in `plots.read_progress`, add small validation-log helpers in `plots.training_curves`, and separate multi-seed rendering from existing single-run rendering. The CLI will default to the approved three-seed thesis figure, while `--run-dir` continues to request the existing single-run diagnostics.

**Tech Stack:** Python, NumPy, Matplotlib, pytest, SB3 `eval_progress.csv` files.

---

### Task 1: Add validation-log test fixtures and selection tests

**Files:**
- Modify: `tests/test_plots_training_curves.py`
- Test: `tests/test_plots_training_curves.py`

- [ ] **Step 1: Add synthetic validation CSV helper**

Add a helper that writes `eval_progress.csv` with `timesteps`, `median_attainment`, `share_positive`, and all current `attainment_*` columns. Use three rows where the middle row has the best median, so selecting the last row would fail.

```python
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
```

- [ ] **Step 2: Write failing helper tests**

Add tests for the planned public helpers:

```python
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

    for seed in (2, 0, 1):
        run_dir = tmp_path / f"oneshot_v3_seed{seed}"
        run_dir.mkdir()
        _write_fake_eval_progress_csv(
            run_dir / "eval_progress.csv", [(10000, seed / 10, 0.5)]
        )

    runs = load_validation_runs(str(tmp_path), "oneshot_v3_seed*")

    assert [run["seed"] for run in runs] == [0, 1, 2]


def test_load_validation_run_reports_missing_columns(tmp_path):
    from plots.training_curves import load_validation_run

    run_dir = tmp_path / "oneshot_v3_seed0"
    run_dir.mkdir()
    (run_dir / "eval_progress.csv").write_text("timesteps,median_attainment\n10000,0.2\n")

    with pytest.raises(ValueError, match="share_positive"):
        load_validation_run(str(run_dir))
```

Import `pytest` at the top of the test module. The tests should fail because the helpers do not exist yet.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_plots_training_curves.py::test_load_validation_run_selects_best_median_and_final_row tests/test_plots_training_curves.py::test_load_validation_runs_are_sorted_by_seed tests/test_plots_training_curves.py::test_load_validation_run_reports_missing_columns
```

Expected: FAIL with import errors for the new helpers.

### Task 2: Implement validation-log loading and summary helpers

**Files:**
- Modify: `plots/training_curves.py`
- Test: `tests/test_plots_training_curves.py`

- [ ] **Step 1: Implement required-column validation and seed parsing**

Add constants and helpers near the existing run-discovery code:

```python
EVAL_PROGRESS_NAME = "eval_progress.csv"
VALIDATION_COLUMNS = ("timesteps", "median_attainment", "share_positive")


def _seed_from_run_dir(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    match = re.search(r"seed(\d+)$", name)
    return int(match.group(1)) if match else name
```

`load_validation_run(run_dir)` must load `eval_progress.csv` with `load_progress`, require the three validation columns, reject empty data, select the finite maximum `median_attainment`, and return arrays plus summary fields. If a required column is absent or contains no finite rows, raise `ValueError` naming the run and column.

- [ ] **Step 2: Implement deterministic multi-run discovery**

Implement:

```python
def load_validation_runs(log_dir=LOG_DIR, run_glob=RUN_GLOB):
    candidates = glob.glob(os.path.join(log_dir, run_glob))
    if not candidates:
        raise FileNotFoundError(...)
    runs = [load_validation_run(path) for path in candidates]
    return sorted(runs, key=lambda run: run["seed"])
```

The error must tell the user which glob and directory were searched. Sort numerically by parsed seed, not lexical path order.

- [ ] **Step 3: Add summary-helper tests and run GREEN**

Add a test asserting `seed_spread` or the equivalent aggregate equals the range of best medians across three synthetic runs. Run the focused test file. Expected: all new tests and existing tests pass.

### Task 3: Implement the thesis-clean multi-seed figure

**Files:**
- Modify: `plots/training_curves.py`
- Test: `tests/test_plots_training_curves.py`

- [ ] **Step 1: Write the figure-output test first**

Add a test creating three synthetic run directories, calling the new renderer, and asserting both output files exist and are non-empty:

```python
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
```

- [ ] **Step 2: Run the new output test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_plots_training_curves.py::test_plot_validation_runs_writes_thesis_exports
```

Expected: FAIL because `plot_validation_runs` does not exist.

- [ ] **Step 3: Implement the two-column figure**

Implement `plot_validation_runs(log_dir=LOG_DIR, run_glob=RUN_GLOB, output_dir=OUTPUT_DIR)`:

- Call `apply_style()`.
- Load and sort runs with `load_validation_runs`.
- Create a two-column figure, with the validation axis wider than the summary axis.
- Plot each run's raw `timesteps` against `median_attainment` with blue, sage, and amber seed colours.
- Mark each run's best point with a rust-edged marker and label the selected timestep.
- Use a `k` timestep formatter and labels that say `Validation median attainment` and `Training timesteps`.
- Add a right-side summary box listing best validation median, selected step, final median, and seed spread.
- Use subtitle/caption text stating fixed validation episodes and best-median checkpoint selection.
- Save `training_curves_thesis.pdf` and `.png` and return the base path.
- Close the figure after saving.

Do not smooth the curves and do not call the validation values test performance.

- [ ] **Step 4: Run focused tests and inspect generated figure**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_plots_training_curves.py
.venv/bin/python -m plots.training_curves --out /tmp/tf-training-curves
```

Expected: all plot tests pass and the command writes `training_curves_thesis.pdf` and `.png`.

### Task 4: Preserve single-run diagnostics and update CLI

**Files:**
- Modify: `plots/training_curves.py`
- Modify: `tests/test_plots_training_curves.py`

- [ ] **Step 1: Add CLI behavior tests**

Test `parse_args([])` selects thesis mode and `parse_args(["--run-dir", "..."])` selects single-run mode. Preserve existing `--out` behavior.

- [ ] **Step 2: Implement explicit mode selection**

Keep `plot_training_curves(run_dir, output_dir)` unchanged for the existing detailed single-run diagnostics. Update `main()` so:

- No `--run-dir`: call `plot_validation_runs` and write the thesis exports.
- With `--run-dir`: call `plot_training_curves` and write the existing `training_curves.pdf/.png` diagnostics.
- Add `--log-dir` and `--run-glob` for controlling default multi-seed discovery without changing the established `--run-dir` path.

Print the exact files written in both modes.

- [ ] **Step 3: Run all plot tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_plots_read_progress.py tests/test_plots_training_curves.py
```

Expected: all tests pass.

### Task 5: Generate current artifacts and verify repository state

**Files:**
- Generated: `plots/output/training_curves_thesis.pdf`
- Generated: `plots/output/training_curves_thesis.png`

- [ ] **Step 1: Generate the real thesis figure**

Run:

```bash
.venv/bin/python -m plots.training_curves
```

Expected: reads the three `out/rl_v2/oneshot_v3_seed*` validation logs and writes the two thesis exports.

- [ ] **Step 2: Run the full fast suite**

Run:

```bash
.venv/bin/python -m pytest -q -m "not slow"
```

Expected: zero failures.

- [ ] **Step 3: Inspect diff and status**

Run:

```bash
```

Confirm only intended source, test, spec/plan, and generated plot files are present. Do not add browser brainstorming artifacts or unrelated optimization specs to this feature commit.

- [ ] **Step 4: Commit implementation**

```bash
git commit -m "feat: add thesis training curves"
```
