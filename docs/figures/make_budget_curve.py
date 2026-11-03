"""Budget curve: median attainment against the number of renders a search
may spend, starting cold (the task's start TF), from the policy's answer, and
from the policy's answer refined by planning. Test split, task v3c, the fixed
300 episodes, seeds 0/1/2 (line = mean of the per-seed medians, band =
min..max over seeds). The cold search is policy-independent (one run).

Draft: the evaluation so far has 0 / 30 / 200 renders for the warm starts and
only 200 for the cold search; 10 and 100 (and the cold curve in between)
need another eval run.

    .venv/bin/python docs/figures/make_budget_curve.py
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVAL_DIR = "out/rl_mb_v3c/eval"
OUT = "docs/figures/budget_curve"
SPLIT = "test"
# Reference categorical palette, slots 1-3 (validated for the light surface;
# slot 3 is below 3:1 contrast, so every line is also direct-labelled and has
# its own marker).
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6a63", "#e6e5e0", "#fcfcfb"
# (label, eval keys, budgets, colour, marker, end-label vertical offset in points,
#  line style: solid where the curve has measured points close together, dotted
#  where only the two ends are measured)
SERIES = [
    ("policy + planning, then search", ["plan", "plan_warm30", "plan_warm200"], [0, 30, 200], "#2a78d6", "o", 6, "-"),
    ("policy, then search", ["policy", "warm30", "warm200"], [0, 30, 200], "#eb6834", "s", -6, "-"),
    ("search from the start", [None, "cold200"], [0, 200], "#1baf7a", "D", 0, ":"),
]


def rows(seed):
    out = []
    for f in sorted(glob.glob(os.path.join(EVAL_DIR, f"v3c_s{seed}_{SPLIT}_*.json"))):
        out.extend(json.load(open(f)))
    return out


def main():
    per_seed = {s: rows(s) for s in (0, 1, 2)}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(6.2, 3.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for label, keys, budgets, color, marker, dy, style in SERIES:
        medians = []
        for key in keys:
            if key is None:  # the start itself: attainment 0 by definition
                medians.append([0.0, 0.0, 0.0])
                continue
            vals = []
            for s in (0, 1, 2):
                v = [r[key] for r in per_seed[s] if key in r]
                if v:
                    vals.append(float(np.median(v)))
            medians.append(vals)
        mean = np.array([np.mean(m) for m in medians])
        lo = np.array([np.min(m) for m in medians])
        hi = np.array([np.max(m) for m in medians])
        x = np.array(budgets, dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, linestyle=style, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(f"{mean[-1]:.2f}", (x[-1], mean[-1]), xytext=(7, dy), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
        if key_point := dict(zip(budgets, mean)).get(30) if label.startswith("policy + planning") else None:
            ax.annotate("30 renders here ≈ 200 cold", (30, key_point), xytext=(-95, 28), textcoords="offset points",
                        fontsize=8.5, color=INK, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})
        print(f"{label:<34}", " ".join(f"{b}:{m:.3f}" for b, m in zip(budgets, mean)))
    # The headline: 30 renders from the policy + planning reach what the cold search needs 200 for.
    cold200 = np.mean([float(np.median([r["cold200"] for r in per_seed[s] if "cold200" in r]))
                       for s in (0, 1, 2) if any("cold200" in r for r in per_seed[s])])
    ax.axhline(cold200, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.2, cold200 - 0.045, f"cold search, 200 renders ({cold200:.2f})", fontsize=8.5, color=MUTED)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([0, 10, 30, 100, 200])
    ax.set_xticklabels(["0", "10", "30", "100", "200"])
    ax.set_xlim(-0.5, 200)
    ax.set_ylim(0, 1)
    ax.set_xlabel("renders spent on search (log scale)")
    ax.set_ylabel(f"median attainment ({SPLIT}, 300 episodes)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.99, 0.02, "dotted: not measured between 0 and 200", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.06))
    fig.subplots_adjust(right=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
