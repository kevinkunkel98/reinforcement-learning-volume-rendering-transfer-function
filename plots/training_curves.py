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

# The one-shot pipeline the thesis reports. `out/rl_logs` holds the retired
# ten-step runs, and defaulting there once put a figure of the wrong experiment
# in front of a reader.
LOG_DIR = "out/rl_v2"
RUN_GLOB = "oneshot_v3_seed*"
OUTPUT_DIR = "plots/output"


def _find_latest_run(log_dir: str, run_glob: str = RUN_GLOB) -> str:
    candidates = glob.glob(os.path.join(log_dir, run_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No {run_glob} directories found under {log_dir!r} -- run an SB3 training script first."
        )
    return max(candidates, key=os.path.getmtime)


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


def main():
    parser = argparse.ArgumentParser(description="Plot SAC training curves for the thesis.")
    parser.add_argument("--run-dir", type=str, default=None,
                         help=f"Path to a run directory (default: most recent {RUN_GLOB} under {LOG_DIR})")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir or _find_latest_run(LOG_DIR)
    base = plot_training_curves(run_dir, output_dir=args.out)
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
