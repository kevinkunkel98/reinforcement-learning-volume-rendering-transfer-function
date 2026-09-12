"""Plot the held-out learned-reward score and the true mass_fraction metric
(recorded by rl/reward_model_train.py's eval_progress.csv) over RL training
against the goal-conditioned reward model, side by side."""
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


def _find_latest_run(log_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(log_dir, "reward_model_run_seed*")))
    if not candidates:
        raise FileNotFoundError(
            f"No reward_model_run_seed* directories found under {log_dir!r} -- "
            "run `python -m rl.reward_model_train` first."
        )
    return candidates[-1]


def plot_reward_model_eval_curve(run_dir: str, output_dir: str = OUTPUT_DIR) -> str:
    apply_style()
    csv_path = os.path.join(run_dir, "eval_progress.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No eval_progress.csv found in {run_dir!r} -- pass a directory "
            "produced by `python -m rl.reward_model_train`."
        )
    cols = load_progress(csv_path)
    x = cols["timesteps"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, cols["mean_reward_model_score"], color=NAVY, linewidth=1.8,
            marker="o", markersize=3, label="Mean learned-reward score")
    ax.set_xlabel("RL training timesteps")
    ax.set_ylabel("Mean learned-reward score", color=NAVY)
    ax.set_title("RLHF policy training vs. the goal-conditioned reward model")

    ax2 = ax.twinx()
    ax2.plot(x, cols["mean_automatic_mass_fraction_reward"], color=AMBER, linewidth=1.3,
             linestyle=":", marker="s", markersize=3, label="Mean automatic mass_fraction reward")
    ax2.set_ylabel("Mean automatic mass_fraction reward", color=AMBER)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=7)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "reward_model_eval_curve")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    plt.close(fig)
    return base


def main():
    parser = argparse.ArgumentParser(description="Plot RLHF reward-model-training eval curve.")
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Path to a reward_model_run_seed* directory (default: most recent under out/rl_logs)")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir or _find_latest_run(LOG_DIR)
    base = plot_reward_model_eval_curve(run_dir, output_dir=args.out)
    print(f"Wrote {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()
