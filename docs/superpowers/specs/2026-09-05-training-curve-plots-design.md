# Training Curve Plots — Design

## Purpose

The RL training run (`rl/train.py`) currently only exposes its metrics through
TensorBoard's event-file format, which is fine for live monitoring but not
for producing saved, publication-quality figures for the thesis. This adds
a small, separate `plots/` tool that reads the same training metrics from a
plain CSV and renders a fixed set of matplotlib figures, styled to match the
pitch deck's existing color palette, saved as PDF (for inclusion in the
thesis) and PNG (for quick preview).

Scope: **training curves only** (episode reward, actor/critic loss, entropy
coefficient) — not the policy-vs-hill-climbing comparison, not per-episode
trajectories, not transfer-function curves. Those were considered and
explicitly deferred; this tool can be extended to them later if wanted.

## Grounding in existing code

- `rl/train.py`'s `train()` builds `SAC(..., tensorboard_log=LOG_DIR, ...)`.
  SB3 only writes tensorboard event files by default; there is no CSV output
  unless a custom logger is attached.
- Verified directly against the installed `stable-baselines3` (2.9.0) in this
  venv: `stable_baselines3.common.logger.configure(folder, format_strings)`
  returns a `Logger` that can write multiple formats at once (confirmed
  signature: `configure(folder: str | None = None, format_strings: list[str]
  | None = None) -> Logger`).
- Also verified: `BaseAlgorithm.set_logger(logger)` sets `self._custom_logger
  = True`, and `_setup_learn()` only calls the default
  `utils.configure_logger(...)` when `not self._custom_logger` — so calling
  `model.set_logger(...)` *before* `.learn()` is the correct, non-hacky way
  to add CSV output without losing the existing tensorboard output.
- `pandas` is not installed in this venv; `matplotlib` (3.11.1) already is.
  The CSV has a handful of numeric columns — read with the standard `csv`
  module + `numpy` rather than adding a new dependency.
- Color palette source: `slides/helpers.typ` — `navy = #1c3a5e`, `blue =
  #1a4f8a`, `sage = #1e6b3c`, `amber = #b7770d`. Reusing these (rather than a
  generic style pack like `scienceplots`) keeps thesis figures and slides
  visually consistent.

## Changes to `rl/train.py`

In `train()`, immediately after constructing `model = SAC(...)` and before
`model.learn(...)`:

```python
from stable_baselines3.common.logger import configure as configure_logger

run_dir = os.path.join(LOG_DIR, f"run_seed{seed}")
os.makedirs(run_dir, exist_ok=True)
model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))
```

This writes `out/rl_logs/run_seed{seed}/progress.csv` and the tensorboard
event file into the same per-seed folder — a second run with the same seed
overwrites that seed's data (acceptable; a different seed gets a distinct
folder, avoiding the cross-run clobbering problem the model-save path
already had to fix). `LOG_DIR`/`MODEL_PATH` constants and the `--out` flag
are unaffected — this only changes *where the logger writes*, not the
model's own save path.

`progress.csv`'s columns (SB3's standard `CSVOutputFormat` header) include
(among others): `time/total_timesteps`, `rollout/ep_rew_mean`,
`train/actor_loss`, `train/critic_loss`, `train/ent_coef`. These are the
same keys already visible in the existing stdout table dumps (see the
training log excerpt from the completed 200k run).

## New package: `plots/`

- `plots/__init__.py` — empty package marker.
- `plots/style.py` — shared matplotlib style setup, reusable by any future
  plot script:
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
- `plots/read_progress.py` — CSV loading, no pandas:
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
  SB3's CSV format can have differing columns present per row (e.g. episode
  stats only appear once an episode ends), leaving some cells blank — `nan`
  for missing values lets matplotlib skip them naturally when plotting a
  line (NaNs create gaps, not crashes).
- `plots/training_curves.py` — the CLI:
  ```python
  """Render training-curve figures (reward, losses, entropy coefficient)
  from an SB3 progress.csv, styled to match the thesis slide palette."""
  import argparse
  import glob
  import os

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

## Testing

- `tests/test_plots_read_progress.py` — writes a small synthetic CSV (a
  handful of rows with the real SB3 column names, including one row with an
  intentionally blank cell to exercise the NaN path) to a temp file, and
  asserts `load_progress()` returns arrays of the right length with the
  blank cell as `nan` and everything else parsed as the correct float.
- `tests/test_plots_training_curves.py` — writes a small synthetic
  `progress.csv` (≥3 rows, all required columns populated) into a temp
  `run_seed0` directory, calls `plot_training_curves()` on it, and asserts
  both `training_curves.pdf` and `training_curves.png` are created and
  non-empty. Does not attempt to inspect pixel content — a real end-to-end
  render either raises (bad column name, bad matplotlib call) or succeeds;
  this test only needs to catch the former.

No test exercises `rl/train.py`'s new logger line beyond the existing SAC
smoke test still passing (it now also produces a `progress.csv` as a side
effect, which the smoke test doesn't need to inspect).

## New dependencies

None. `matplotlib` is already installed; CSV parsing uses the standard
library plus `numpy` (already a dependency).

## Explicitly out of scope

- Policy-vs-hill-climbing comparison plots, per-episode trajectory plots,
  and transfer-function curve plots — deferred, not part of this delivery.
- Any change to what `rl/eval.py` reports — untouched by this work.
- Auto-regenerating plots as part of `rl/train.py`'s own run (the plotting
  step is a separate, manually-invoked script, consistent with treating it
  as thesis tooling rather than part of the training pipeline itself).
