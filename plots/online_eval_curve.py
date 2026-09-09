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
