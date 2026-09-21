"""Render training-curve figures (reward, losses, entropy coefficient)
from an SB3 progress.csv, styled to match the thesis slide palette."""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from plots.read_progress import load_progress
from plots.style import AMBER, BLUE, NAVY, SAGE, apply_style

# The one-shot pipeline the thesis reports. `out/rl_logs` holds the retired
# ten-step runs, and defaulting there once put a figure of the wrong experiment
# in front of a reader.
LOG_DIR = "out/rl_v2"
RUN_GLOB = "oneshot_v3_seed*"
OUTPUT_DIR = "plots/output"
EVAL_PROGRESS_NAME = "eval_progress.csv"
VALIDATION_COLUMNS = ("timesteps", "median_attainment", "share_positive")


def _find_latest_run(log_dir: str, run_glob: str = RUN_GLOB) -> str:
    candidates = glob.glob(os.path.join(log_dir, run_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No {run_glob} directories found under {log_dir!r} -- run an SB3 training script first."
        )
    return max(candidates, key=os.path.getmtime)


def _seed_from_run_dir(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    match = re.search(r"seed(\d+)$", name)
    if not match:
        raise ValueError(
            f"Validation run {run_dir!r} must end with a numeric seed (for example, seed0)"
        )
    return int(match.group(1))


def load_validation_run(run_dir: str) -> dict:
    csv_path = os.path.join(run_dir, EVAL_PROGRESS_NAME)
    try:
        columns = load_progress(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Validation run {run_dir!r} is missing {EVAL_PROGRESS_NAME!r}; "
            f"expected columns: {', '.join(VALIDATION_COLUMNS)}"
        ) from exc
    missing = [column for column in VALIDATION_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Validation run {run_dir!r} missing column(s): {', '.join(missing)}")
    if not columns["timesteps"].size:
        raise ValueError(f"Validation run {run_dir!r} has no data")

    for column in VALIDATION_COLUMNS:
        values = columns[column]
        if not np.isfinite(values).any():
            raise ValueError(
                f"Validation run {run_dir!r} has no finite data in column {column!r}"
            )
    final_median_index = np.flatnonzero(np.isfinite(columns["median_attainment"]))[-1]
    final_share_index = np.flatnonzero(np.isfinite(columns["share_positive"]))[-1]
    valid_best = np.isfinite(columns["timesteps"]) & np.isfinite(
        columns["median_attainment"]
    )
    if not valid_best.any():
        raise ValueError(
            f"Validation run {run_dir!r} has no finite data in columns "
            "'timesteps' and 'median_attainment'"
        )

    medians = columns["median_attainment"]
    valid_medians = np.where(valid_best, medians, -np.inf)
    best_index = int(np.argmax(valid_medians))
    return {
        "seed": _seed_from_run_dir(run_dir),
        "timesteps": columns["timesteps"],
        "median_attainment": medians,
        "share_positive": columns["share_positive"],
        "best_median": float(medians[best_index]),
        "best_timestep": int(columns["timesteps"][best_index]),
        "final_median": float(medians[final_median_index]),
        "final_share_positive": float(columns["share_positive"][final_share_index]),
    }


def load_validation_runs(log_dir: str = LOG_DIR, run_glob: str = RUN_GLOB) -> list:
    candidates = glob.glob(os.path.join(log_dir, run_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No validation runs matching {run_glob!r} found under {log_dir!r}"
        )
    runs = sorted(
        (load_validation_run(path) for path in candidates),
        key=lambda run: run["seed"],
    )
    seed_spread = max(run["best_median"] for run in runs) - min(
        run["best_median"] for run in runs
    )
    for run in runs:
        run["seed_spread"] = float(seed_spread)
    return runs


def plot_validation_runs(
    log_dir: str = LOG_DIR,
    run_glob: str = RUN_GLOB,
    output_dir: str = OUTPUT_DIR,
) -> str:
    apply_style()
    runs = load_validation_runs(log_dir, run_glob)
    colors = (BLUE, SAGE, AMBER)
    rust = "#9a4d2f"

    fig, (ax, summary_ax) = plt.subplots(
        1,
        2,
        figsize=(9.2, 5.2),
        gridspec_kw={"width_ratios": (2.8, 1.2)},
    )
    for index, run in enumerate(runs):
        color = colors[index % len(colors)]
        timesteps = run["timesteps"]
        medians = run["median_attainment"]
        finite = np.isfinite(timesteps) & np.isfinite(medians)
        ax.plot(
            timesteps[finite],
            medians[finite],
            color=color,
            linewidth=1.8,
            label=f"Seed {run['seed']}",
        )
        ax.scatter(
            run["best_timestep"],
            run["best_median"],
            color="white",
            edgecolor=rust,
            linewidth=1.6,
            s=58,
            zorder=4,
        )
        ax.annotate(
            f"{run['best_timestep'] / 1000:.0f}k",
            (run["best_timestep"], run["best_median"]),
            xytext=(5, 6),
            textcoords="offset points",
            color=rust,
            fontsize=8,
        )

    ax.set_title("Validation performance across seeds", color=NAVY)
    ax.set_ylabel("Validation median attainment")
    ax.set_xlabel("Training timesteps")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}k"))
    ax.legend(loc="best", frameon=False)

    summary_ax.axis("off")
    best = "\n".join(
        f"Seed {run['seed']}: {run['best_median']:.2f} at "
        f"{run['best_timestep'] / 1000:.0f}k steps"
        for run in runs
    )
    final = "\n".join(
        f"Seed {run['seed']}: {run['final_median']:.2f}"
        for run in runs
    )
    summary_ax.text(
        0.02,
        0.97,
        "Selected checkpoints",
        transform=summary_ax.transAxes,
        va="top",
        color=NAVY,
        fontsize=12,
        fontweight="bold",
    )
    summary_ax.text(
        0.02,
        0.86,
        f"Best median\n{best}\n\nFinal median\n{final}\n\n"
        f"Seed spread\n{runs[0]['seed_spread']:.2f}",
        transform=summary_ax.transAxes,
        va="top",
        linespacing=1.45,
    )
    fig.suptitle("One-shot validation training curves", color=NAVY, y=0.99)
    fig.text(
        0.5,
        0.01,
        "Fixed validation episodes; checkpoints selected by best validation median. "
        "Validation is not held-out test performance.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "training_curves_thesis")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    plt.close(fig)
    return base


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
    # 150k steps in full digits overruns the axis and the labels collide;
    # the same "120k" formatter the held-out figures use.
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v/1000:.0f}k")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "training_curves")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    plt.close(fig)
    return base


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot SAC training curves for the thesis.")
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Path to a run directory for single-run diagnostics")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                         help=f"Validation log directory (default: {LOG_DIR})")
    parser.add_argument("--run-glob", type=str, default=RUN_GLOB,
                         help=f"Validation run glob (default: {RUN_GLOB})")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir:
        base = plot_training_curves(args.run_dir, output_dir=args.out)
    else:
        base = plot_validation_runs(
            log_dir=args.log_dir,
            run_glob=args.run_glob,
            output_dir=args.out,
        )
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
