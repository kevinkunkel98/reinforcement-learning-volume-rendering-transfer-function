# Online Learning for the Transfer-Function RL Policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone script that trains a fresh SAC agent online against `TFEnv`, periodically re-evaluated on the same held-out episodes `rl/eval.py` uses, so its learning is directly observable via a new CSV log and plot — then actually run it to produce real results.

**Architecture:** `rl/online_train.py` wraps SB3's own `model.learn()` in a chunked loop (no hand-rolled replay-buffer/train calls), re-using `rl.eval`'s private helpers (`_build_episodes`, `_run_policy`, `_run_hill_climb`, `_steps_to_90pct`) unmodified so the online run's numbers are computed identically to the documented offline table. `plots/online_eval_curve.py` mirrors `plots/training_curves.py`'s structure to render the new eval log, reusing `plots/style.py` and `plots/read_progress.py` unmodified.

**Tech Stack:** Python, `stable-baselines3` (SAC), `gymnasium` (`TFEnv`), `matplotlib`, `pytest`.

**Full design:** `docs/superpowers/specs/2026-09-09-rl-online-learning-design.md`

---

## Task 1: `rl/online_train.py`

**Files:**
- Create: `rl/online_train.py`
- Test: `tests/test_rl_online_train.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

Save this to `tests/test_rl_online_train.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rl_online_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.online_train'`

- [ ] **Step 3: Write the implementation**

```python
"""Online-learning counterpart to rl/train.py -- trains a fresh SAC agent on
a single TFEnv, periodically re-evaluating against the same held-out
episodes rl/eval.py uses so the task metric's improvement over training is
directly observable, not just inferred from SB3's own reward curve.

Unlike rl/train.py (4 parallel envs, one 200k-step batch, evaluated once at
the end by rl/eval.py), this trains one env continuously and checkpoints
its held-out performance every `eval_interval` steps -- same underlying SAC
mechanics (SB3's own model.learn() drives the actual online interaction and
replay-buffer/gradient-update bookkeeping), just interleaved with
visibility. Writes to a separate model path and log directory from the
offline run, so neither one clobbers the other.
"""
import argparse
import csv
import math
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_logger

from rl.env import TFEnv
from rl.eval import EVAL_SEED, N_EPISODES, _build_episodes, _run_hill_climb, _run_policy, _steps_to_90pct

MODEL_PATH = "out/rl_models/sac_tf_agent_online.zip"
LOG_DIR = "out/rl_logs"


def online_train(total_timesteps: int = 50_000, eval_interval: int = 5_000,
                  seed: int = 0, model_path: str = MODEL_PATH) -> SAC:
    run_dir = os.path.join(LOG_DIR, f"online_run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = TFEnv(seed=seed)
    model = SAC("MlpPolicy", env, seed=seed)
    model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))

    # Same held-out set rl/eval.py checks the offline policy against, so
    # this run's numbers are directly comparable to the documented offline
    # table. Hill-climb trajectories are deterministic per episode, so
    # they're computed once here rather than once per checkpoint.
    episodes = _build_episodes(N_EPISODES, EVAL_SEED)
    hill_climb_fractions = [_run_hill_climb(ep) for ep in episodes]

    eval_log_path = os.path.join(run_dir, "eval_progress.csv")
    with open(eval_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["timesteps", "mean_final_mass_fraction", "mean_steps_to_90pct"])

    n_chunks = math.ceil(total_timesteps / eval_interval)
    timesteps_done = 0
    for _ in range(n_chunks):
        chunk = min(eval_interval, total_timesteps - timesteps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        timesteps_done += chunk

        finals, steps_to_90 = [], []
        for ep, hf in zip(episodes, hill_climb_fractions):
            pf = _run_policy(model, ep)
            combined = pf + hf
            best = max(combined) if ep["direction"] == "increase" else min(combined)
            finals.append(pf[-1])
            steps_to_90.append(_steps_to_90pct(pf, pf[0], best))

        row = [timesteps_done, float(np.mean(finals)), float(np.mean(steps_to_90))]
        with open(eval_log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[online_train] timesteps={timesteps_done} "
              f"mean_final_mass_fraction={row[1]:.4f} mean_steps_to_90pct={row[2]:.2f}")

    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train the TF SAC agent online, with periodic held-out evaluation.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    online_train(args.timesteps, eval_interval=args.eval_interval,
                 seed=args.seed, model_path=args.out)


if __name__ == "__main__":
    main()
```

Save this to `rl/online_train.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rl_online_train.py -v -m slow`
Expected: PASS (takes roughly 10-30 seconds — a real 200-step SAC run)

- [ ] **Step 5: Commit**

```bash
git add rl/online_train.py tests/test_rl_online_train.py
git commit -m "Add rl/online_train.py: online SAC training with held-out eval logging"
```

---

## Task 2: `plots/online_eval_curve.py`

**Files:**
- Create: `plots/online_eval_curve.py`
- Test: `tests/test_plots_online_eval_curve.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

Save this to `tests/test_plots_online_eval_curve.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plots_online_eval_curve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plots.online_eval_curve'`

- [ ] **Step 3: Write the implementation**

```python
"""Plot the held-out task metric (mean final mass_fraction, mean steps to
90% of best) recorded by rl/online_train.py's eval_progress.csv over
training progress, against the documented offline SAC/hill-climb baseline
from docs/architecture.typ's "RL sub-project 1" results table."""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.read_progress import load_progress
from plots.style import AMBER, NAVY, apply_style

LOG_DIR = "out/rl_logs"
OUTPUT_DIR = "plots/output"

# docs/architecture.typ, "RL sub-project 1" results table (200k timesteps,
# 20 held-out episodes). Hardcoded so this plot doesn't depend on that run's
# results file existing on disk.
OFFLINE_POLICY_MASS_FRACTION = 0.344
OFFLINE_HILL_CLIMB_MASS_FRACTION = 0.347


def _find_latest_run(log_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(log_dir, "online_run_seed*")))
    if not candidates:
        raise FileNotFoundError(
            f"No online_run_seed* directories found under {log_dir!r} -- "
            "run `python -m rl.online_train` first."
        )
    return candidates[-1]


def plot_online_eval_curve(run_dir: str, output_dir: str = OUTPUT_DIR) -> str:
    apply_style()
    csv_path = os.path.join(run_dir, "eval_progress.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No eval_progress.csv found in {run_dir!r} -- pass a directory "
            "produced by `python -m rl.online_train`, not an offline run_seed* dir."
        )
    cols = load_progress(csv_path)
    x = cols["timesteps"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, cols["mean_final_mass_fraction"], color=NAVY, linewidth=1.8,
            marker="o", markersize=3, label="Online policy (held-out mean)")
    ax.axhline(OFFLINE_POLICY_MASS_FRACTION, color=NAVY, linestyle="--", linewidth=1.0,
               label=f"Offline SAC policy ({OFFLINE_POLICY_MASS_FRACTION:.3f})")
    ax.axhline(OFFLINE_HILL_CLIMB_MASS_FRACTION, color=AMBER, linestyle="--", linewidth=1.0,
               label=f"Hill-climb baseline ({OFFLINE_HILL_CLIMB_MASS_FRACTION:.3f})")
    ax.set_xlabel("Online training timesteps")
    ax.set_ylabel("Mean final mass_fraction (held-out)")
    ax.set_title("Online-learning progress vs. offline baselines")

    ax2 = ax.twinx()
    ax2.plot(x, cols["mean_steps_to_90pct"], color=AMBER, linewidth=1.3, linestyle=":",
             label="Mean steps to 90% of best")
    ax2.set_ylabel("Mean steps to 90% of best", color=AMBER)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=7)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "online_eval_curve")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    plt.close(fig)
    return base


def main():
    parser = argparse.ArgumentParser(description="Plot online-learning held-out eval curve.")
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Path to an online_run_seed* directory (default: most recent under out/rl_logs)")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir or _find_latest_run(LOG_DIR)
    base = plot_online_eval_curve(run_dir, output_dir=args.out)
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
```

Save this to `plots/online_eval_curve.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_plots_online_eval_curve.py -v`
Expected: PASS (3 tests, all fast — no real training involved)

- [ ] **Step 5: Commit**

```bash
git add plots/online_eval_curve.py tests/test_plots_online_eval_curve.py
git commit -m "Add plots/online_eval_curve.py: plot online-learning held-out eval curve"
```

---

## Task 3: Produce real results

This task has no new code — it runs what Tasks 1-2 built, for real, to get the actual results/logs/plots.

**Files:** none created or modified — this task only runs existing scripts and inspects their output.

- [ ] **Step 1: Run the full pytest suite once (fast tests only) to confirm nothing else broke**

Run: `pytest -m "not slow"`
Expected: PASS, no new failures

- [ ] **Step 2: Run the real online-training run**

Run: `python -m rl.online_train --timesteps 50000 --eval-interval 5000 --seed 0`
Expected: prints 10 `[online_train] timesteps=... mean_final_mass_fraction=... mean_steps_to_90pct=...` lines as it goes (one per 5000-step checkpoint), takes several minutes. When it finishes:
- `out/rl_models/sac_tf_agent_online.zip` exists
- `out/rl_logs/online_run_seed0/eval_progress.csv` has 11 lines (header + 10 checkpoints)
- `out/rl_logs/online_run_seed0/progress.csv` exists (SB3's own log)

- [ ] **Step 3: Render both plots**

Run:
```bash
python -m plots.training_curves --run-dir out/rl_logs/online_run_seed0
python -m plots.online_eval_curve --run-dir out/rl_logs/online_run_seed0
```
Expected: both print `Wrote plots/output/<name>.pdf and .png`; `plots/output/training_curves.{pdf,png}` and `plots/output/online_eval_curve.{pdf,png}` all exist and are non-empty.

- [ ] **Step 4: Read back the final row of `eval_progress.csv` and report it**

Run: `tail -1 out/rl_logs/online_run_seed0/eval_progress.csv`
Compare the printed `mean_final_mass_fraction` against the documented offline numbers (0.344 policy / 0.347 hill-climb, from `docs/architecture.typ`) to see how close the online-only run got in 50k steps vs. the offline run's 200k-across-4-envs budget.

- [ ] **Step 5: Nothing to commit — report the results instead**

Both `out/` and `plots/output/` are gitignored (confirmed: `.gitignore` lines 1
and 2), matching how the offline run's own logs/plots are already handled —
generated artifacts stay local, never tracked in git. So this step is not a
commit: report the final `eval_progress.csv` row's numbers back to the user
in the conversation, and send the two PNGs (`plots/output/online_eval_curve.png`,
`plots/output/training_curves.png`) to the user directly (e.g. via
`SendUserFile` if working in Claude Code) so they can see the curves without
needing filesystem access.

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the design's "New module: rl/online_train.py" and "Testing" sections. Task 2 covers "New plotting: plots/online_eval_curve.py" and its testing section, including both `FileNotFoundError` cases from "Error handling". Task 3 covers "Data flow" end-to-end and delivers the actual results/logs/plots the user asked for.
- **Deviation from spec, called out explicitly:** the design's constants section showed `EVAL_N_EPISODES`/`EVAL_SEED` as new constants redefined in `rl/online_train.py`; the implementation instead imports `N_EPISODES`/`EVAL_SEED` directly from `rl.eval` to guarantee they can never drift apart from the offline eval's own values (DRY, still satisfies the spec's intent: "match rl.eval's defaults exactly").
- **Type/signature consistency:** `online_train(total_timesteps, eval_interval, seed, model_path)` in Task 1 matches the call in Task 3 Step 2's CLI invocation and the test in Task 1. `plot_online_eval_curve(run_dir, output_dir)` in Task 2 matches its test calls and the CLI in the same file.
- **No placeholders:** all steps show complete, runnable code; no TBD/TODO.
