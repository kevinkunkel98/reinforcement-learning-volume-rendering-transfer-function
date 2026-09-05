# Training Curve Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CSV-logging path to `rl/train.py` and a small standalone `plots/` tool that renders thesis-quality matplotlib figures (reward, actor/critic loss, entropy coefficient) from it.

**Architecture:** `rl/train.py` gets a per-seed CSV+tensorboard logger attached before `.learn()`. A new `plots/` package (independent of `rl/`) reads that CSV with plain `csv`+`numpy` (no pandas) and renders one composite figure via matplotlib, styled with the thesis slide palette.

**Tech Stack:** Python, stable-baselines3 (already installed), matplotlib (already installed), numpy, pytest.

Spec: `docs/superpowers/specs/2026-09-05-training-curve-plots-design.md`

---

## File Structure

- Modify: `rl/train.py` — attach a custom CSV+tensorboard logger before `.learn()`.
- Create: `plots/__init__.py` — empty package marker.
- Create: `plots/style.py` — shared matplotlib style constants/setup.
- Create: `plots/read_progress.py` — CSV → numpy arrays, no pandas.
- Create: `plots/training_curves.py` — the plotting CLI.
- Create: `tests/test_rl_train_logging.py` — verifies the CSV logger actually gets attached and produces output.
- Create: `tests/test_plots_read_progress.py`
- Create: `tests/test_plots_training_curves.py`

---

### Task 1: Attach a CSV logger to `rl/train.py`

**Files:**
- Modify: `rl/train.py`
- Create: `tests/test_rl_train_logging.py`

Current `rl/train.py` (for reference, do not paste as final — just to locate the exact insertion point):
```python
def train(total_timesteps: int, n_envs: int = 4, seed: int = 0, model_path: str = MODEL_PATH) -> SAC:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    vec_env = make_vec_env(TFEnv, n_envs=n_envs, seed=seed)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_rl_train_logging.py`:

```python
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
```

The `monkeypatch.chdir(tmp_path)` isolates this run's `out/` artifacts from
the real project `out/` directory, matching the existing convention in
`tests/test_commands.py`/`tests/test_server.py`.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rl_train_logging.py -v -m slow`
Expected: FAIL — the `run_seed42/progress.csv` file does not exist yet
(current code only writes tensorboard event files directly under `LOG_DIR`,
no per-seed subfolder, no CSV format).

- [ ] **Step 3: Implement the logger change**

Edit `rl/train.py`'s `train()` function to insert the custom logger between
constructing `model` and calling `.learn()`:

```python
"""Train a SAC agent on TFEnv against the opacity_mass-derived reward."""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as configure_logger

from rl.env import TFEnv

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
LOG_DIR = "out/rl_logs"


def train(total_timesteps: int, n_envs: int = 4, seed: int = 0, model_path: str = MODEL_PATH) -> SAC:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    run_dir = os.path.join(LOG_DIR, f"run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    vec_env = make_vec_env(TFEnv, n_envs=n_envs, seed=seed)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, seed=seed)
    model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the TF SAC agent.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=MODEL_PATH, help="Path to save the trained model (default: %(default)s)")
    args = parser.parse_args()
    train(args.timesteps, n_envs=args.n_envs, seed=args.seed, model_path=args.out)


if __name__ == "__main__":
    main()
```

Note: `SAC(..., tensorboard_log=LOG_DIR, ...)` is still passed
`tensorboard_log=LOG_DIR` (unchanged) — this is harmless even though
`set_logger()` immediately overrides the logger SB3 would have built from
it, and keeps the constructor call minimally different from before.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rl_train_logging.py -v -m slow`
Expected: PASS.

- [ ] **Step 5: Run the full suite to confirm nothing broke**

Run: `.venv/bin/python -m pytest -q`
Expected: all previous tests still pass, plus this one new slow test (86
total, up from 85).

- [ ] **Step 6: Commit**

```bash
git add rl/train.py tests/test_rl_train_logging.py
git commit -m "Add CSV logging to SAC training runs, per-seed run directories"
```

---

### Task 2: `plots` package core — style and CSV reading

**Files:**
- Create: `plots/__init__.py`
- Create: `plots/style.py`
- Create: `plots/read_progress.py`
- Create: `tests/test_plots_read_progress.py`

- [ ] **Step 1: Create the package marker**

Create `plots/__init__.py` (empty file).

- [ ] **Step 2: Create the style module**

Create `plots/style.py`:

```python
"""Shared matplotlib styling matching the thesis slide palette
(slides/helpers.typ)."""
import matplotlib.pyplot as plt

NAVY = "#1c3a5e"
BLUE = "#1a4f8a"
SAGE = "#1e6b3c"
AMBER = "#b7770d"


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
```

No test needed for this file — it only sets global matplotlib rcParams with
no branching logic to verify; it's exercised indirectly by Task 3's test.

- [ ] **Step 3: Write the failing test for CSV reading**

Create `tests/test_plots_read_progress.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_plots_read_progress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plots.read_progress'`

- [ ] **Step 5: Implement `plots/read_progress.py`**

```python
"""Load an SB3 progress.csv into plain numpy arrays, by column name."""
import csv

import numpy as np


def load_progress(csv_path: str) -> dict:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    columns = {}
    for key in rows[0].keys():
        values = []
        for row in rows:
            raw = row.get(key, "")
            values.append(float(raw) if raw not in ("", None) else np.nan)
        columns[key] = np.array(values, dtype=np.float64)
    return columns
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_plots_read_progress.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add plots/__init__.py plots/style.py plots/read_progress.py tests/test_plots_read_progress.py
git commit -m "Add plots package: style constants and CSV-to-numpy loader"
```

---

### Task 3: `plots/training_curves.py` — the plotting CLI

**Files:**
- Create: `plots/training_curves.py`
- Create: `tests/test_plots_training_curves.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_plots_training_curves.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plots_training_curves.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plots.training_curves'`

- [ ] **Step 3: Implement `plots/training_curves.py`**

```python
"""Render training-curve figures (reward, losses, entropy coefficient)
from an SB3 progress.csv, styled to match the thesis slide palette."""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plots.read_progress import load_progress
from plots.style import AMBER, BLUE, NAVY, SAGE, apply_style

LOG_DIR = "out/rl_logs"
OUTPUT_DIR = "plots/output"


def _find_latest_run(log_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(log_dir, "run_seed*")))
    if not candidates:
        raise FileNotFoundError(
            f"No run_seed* directories found under {log_dir!r} -- run `python -m rl.train` first."
        )
    return candidates[-1]


def plot_training_curves(run_dir: str, output_dir: str = OUTPUT_DIR) -> str:
    apply_style()
    csv_path = os.path.join(run_dir, "progress.csv")
    cols = load_progress(csv_path)
    x = cols["time/total_timesteps"]

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)

    ax = axes[0]
    ax.plot(x, cols["rollout/ep_rew_mean"], color=NAVY, linewidth=1.6)
    ax.set_ylabel("Mean episode reward")
    ax.set_title("SAC training diagnostics")

    ax = axes[1]
    ax.plot(x, cols["train/actor_loss"], color=BLUE, linewidth=1.3, label="Actor loss")
    ax2 = ax.twinx()
    ax2.plot(x, np.abs(cols["train/critic_loss"]), color=SAGE, linewidth=1.3, label="Critic loss (abs, log scale)")
    ax2.set_yscale("log")
    ax.set_ylabel("Actor loss", color=BLUE)
    ax2.set_ylabel("Critic loss", color=SAGE)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(x, cols["train/ent_coef"], color=AMBER, linewidth=1.6)
    ax.set_ylabel("Entropy coefficient")
    ax.set_xlabel("Training timesteps")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "training_curves")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    plt.close(fig)
    return base


def main():
    parser = argparse.ArgumentParser(description="Plot SAC training curves for the thesis.")
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Path to a run_seed* directory (default: most recent under out/rl_logs)")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir or _find_latest_run(LOG_DIR)
    base = plot_training_curves(run_dir, output_dir=args.out)
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
```

Note the added `matplotlib.use("Agg")` before importing `pyplot` — this is
not in the original spec snippet but is required for the test suite (and
any headless/CI run) to render figures without a display backend; it must
be set before the first `import matplotlib.pyplot`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_plots_training_curves.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (87 total: 84 previous fast tests + 2 from Task 2 + 1
from this task; Task 1's test is `slow`-marked so not counted here).

- [ ] **Step 6: Manual end-to-end smoke test against a real training run**

Run a short real training run and plot it:
```bash
.venv/bin/python -m rl.train --timesteps 2000 --n-envs 2 --seed 99
.venv/bin/python -m plots.training_curves --run-dir out/rl_logs/run_seed99
```
Expected: second command prints `Wrote plots/output/training_curves.pdf and
plots/output/training_curves.png`, and both files exist with non-trivial
size. Open the PNG (e.g. via the Read tool) to visually confirm three
stacked subplots render sensibly (not blank, not obviously broken/garbled
axes) before considering this task done.

- [ ] **Step 7: Add `plots/output/` to `.gitignore`**

Edit `.gitignore`, adding `plots/output/` alongside the existing `out/`
entry (this is generated output, same treatment as everything under `out/`
already gets).

- [ ] **Step 8: Commit**

```bash
git add plots/training_curves.py tests/test_plots_training_curves.py .gitignore
git commit -m "Add training-curve plotting CLI"
```

Note: the real training run's artifacts from Step 6
(`out/rl_logs/run_seed99/`, `plots/output/training_curves.{pdf,png}`) are
gitignored (`out/` and the newly-added `plots/output/`) and will not be
staged — verify with `git status` before committing that only the three
named files/changes are staged.

---

## Notes for the implementer

- Do not modify `rl/env.py`, `rl/eval.py`, or `evaluate.py` — untouched by
  this plan.
- `pandas` is deliberately not a dependency here — `plots/read_progress.py`
  uses only the standard library `csv` module plus `numpy` (already a
  dependency). Do not introduce `pandas` as a shortcut.
- `matplotlib.use("Agg")` must be called before the first `import
  matplotlib.pyplot` anywhere in the process — if pytest ever fails with a
  backend/display error, check that this line still precedes the pyplot
  import in `plots/training_curves.py`.
