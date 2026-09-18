"""Thesis figures for the held-out comparison: the cost/quality frontier, the
per-instruction-kind breakdown, and the learning curves.

Reads the result files `rl.vis_eval` writes -- nothing here recomputes a
score, so a figure can never disagree with the table it illustrates. Regenerate
the inputs with:

    python -m rl.vis_eval --policy out/rl_v2/oneshot_v3_seed0/best.zip \
        --split test --episodes 200 --formulation one_shot --refine 3 \
        --out out/rl_v2/eval_rerun_v3_seed0.json
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plots.style import AMBER, BLUE, NAVY, SAGE, apply_style

OUTPUT_DIR = "plots/output"
RESULT_GLOB = "out/rl_v2/eval_rerun_v3_seed*.json"

# Evaluations each method spends per instruction. The policy's 0 is the point of
# the whole comparison: it answers without consulting the objective at all.
EVALUATIONS = {
    "B0_do_nothing": 0,
    "B1_current_executor": 0,
    "B2_random_policy": 0,
    "B5_occlusion_rule": 0,
    "policy": 0,
    "policy_plus_refine": 4,
    "B3_hill_climb_10": 10,
    "B4_hill_climb_200": 200,
}

LABELS = {
    "policy": "policy",
    "policy_plus_refine": "policy + 3 refinements",
    "B0_do_nothing": "do nothing",
    "B1_current_executor": "rule executor",
    "B2_random_policy": "random",
    "B3_hill_climb_10": "search (10 evals)",
    "B4_hill_climb_200": "search (200 evals)",
    "B5_occlusion_rule": "occlusion heuristic",
}

KIND_ORDER = ("relative", "compound", "show_only", "absolute", "brightness")


def load_results(pattern: str = RESULT_GLOB) -> dict:
    """{arm: {"median": [per seed], "share_positive": [per seed]}}."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no result files matched {pattern!r} -- run rl.vis_eval first")
    out = {}
    for path in files:
        summary = json.load(open(path))["summary"]
        for arm, stats in summary.items():
            entry = out.setdefault(arm, {"median": [], "share_positive": []})
            entry["median"].append(stats["median"])
            entry["share_positive"].append(stats["share_positive"])
    return out


def plot_frontier(results: dict, output_dir: str = OUTPUT_DIR) -> str:
    """Quality against the number of objective evaluations spent per
    instruction -- the claim the thesis actually makes."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    search = [("B3_hill_climb_10", 10), ("B4_hill_climb_200", 200)]
    xs = [n for _arm, n in search]
    ys = [float(np.median(results[arm]["median"])) for arm, _n in search]
    ax.plot(xs, ys, color=NAVY, marker="o", linewidth=1.6, label="hill-climb search", zorder=3)

    for arm, colour, marker in (("policy", SAGE, "D"), ("policy_plus_refine", AMBER, "s")):
        if arm not in results:
            continue
        values = results[arm]["median"]
        x = max(EVALUATIONS[arm], 0.6)  # 0 evaluations plotted at the left edge of a log axis
        ax.errorbar(x, float(np.median(values)),
                    yerr=[[float(np.median(values) - min(values))], [float(max(values) - np.median(values))]],
                    color=colour, marker=marker, markersize=7, capsize=3, linewidth=1.4,
                    label=LABELS[arm], zorder=4)

    for arm, colour in (("B1_current_executor", "#8a8a8a"), ("B2_random_policy", "#b0b0b0")):
        if arm in results:
            level = float(np.median(results[arm]["median"]))
            ax.axhline(level, color=colour, linestyle=":", linewidth=1.1, zorder=1)
            ax.text(0.62, level - 0.012, LABELS[arm], color=colour, fontsize=8,
                    va="top", ha="left")

    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("objective evaluations per instruction")
    ax.set_ylabel("median attainment")
    ax.set_title("Quality against cost, 200 held-out instructions")
    ax.set_xlim(0.5, 400)
    # The policy spends no evaluations at all; a log axis has no 0, so its point
    # sits at the left edge and the tick says what it means.
    ax.set_xticks([0.6, 4, 10, 200])
    ax.set_xticklabels(["0", "4", "10", "200"])
    ax.minorticks_off()
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "frontier.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_reliability(results: dict, output_dir: str = OUTPUT_DIR) -> str:
    """Median attainment beside the share of instructions improved: the policy
    wins the first and loses the second, which is the honest shape of the
    result."""
    apply_style()
    arms = ["B4_hill_climb_200", "policy_plus_refine", "policy", "B3_hill_climb_10",
            "B1_current_executor", "B2_random_policy", "B5_occlusion_rule"]
    arms = [a for a in arms if a in results]
    y = np.arange(len(arms))

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)

    medians = [float(np.median(results[a]["median"])) for a in arms]
    colours = [SAGE if a.startswith("policy") else NAVY for a in arms]
    left.barh(y, medians, color=colours, height=0.62)
    left.axvline(0.0, color="#333333", linewidth=0.8)
    left.set_yticks(y, [LABELS[a] for a in arms])
    left.invert_yaxis()
    left.set_xlabel("median attainment")

    shares = [float(np.mean(results[a]["share_positive"])) * 100 for a in arms]
    right.barh(y, shares, color=colours, height=0.62)
    right.set_xlabel("instructions improved (%)")
    right.set_xlim(0, 100)

    fig.suptitle("Quality and reliability are different questions", y=0.99)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "reliability.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_kind_curves(run_dirs: list, output_dir: str = OUTPUT_DIR) -> str:
    """Per-instruction-kind attainment over training, from each run's
    eval_progress.csv. Shows both what the policy learns well and that the runs
    had not converged when the step budget ran out."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colours = {"relative": NAVY, "compound": "#8a4b8a", "show_only": BLUE,
               "absolute": SAGE, "brightness": AMBER}

    series = {kind: [] for kind in KIND_ORDER}
    steps = None
    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "eval_progress.csv")
        rows = np.genfromtxt(csv_path, delimiter=",", names=True)
        steps = rows["timesteps"]
        for kind in KIND_ORDER:
            series[kind].append(rows[f"attainment_{kind}"])

    for kind in KIND_ORDER:
        stacked = np.vstack(series[kind])
        median = np.median(stacked, axis=0)
        ax.plot(steps, median, color=colours[kind], linewidth=1.7, label=kind.replace("_", " "))
        ax.fill_between(steps, stacked.min(axis=0), stacked.max(axis=0),
                        color=colours[kind], alpha=0.12, linewidth=0)

    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("training steps")
    ax.set_ylabel("attainment on the evaluation episodes")
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v/1000:.0f}k")
    ax.set_title("What the policy learns, and when")
    ax.text(0.5, 1.015, "median of 3 seeds, range shaded", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color="#555555")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "kind_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=RESULT_GLOB,
                     help="glob for rl.vis_eval result files (default: the v3 re-run)")
    ap.add_argument("--runs", nargs="*", default=sorted(glob.glob("out/rl_v2/oneshot_v3_seed*")),
                     help="training run directories holding eval_progress.csv")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    results = load_results(args.results)
    for path in (plot_frontier(results, args.output_dir),
                 plot_reliability(results, args.output_dir),
                 plot_kind_curves(args.runs, args.output_dir)):
        print(f"[plots] wrote {path}")


if __name__ == "__main__":
    main()
