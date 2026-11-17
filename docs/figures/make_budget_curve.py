"""Budget curve: median attainment against the number of renders a search
may spend, starting cold (the task's start TF), from the policy's answer, and
from the policy's answer refined by planning. The larger held-out split
test_extra (60 patients never used in development, 600 fixed episodes), the
three-seed retrain v3d (line = mean of the per-seed medians, band = min..max
over seeds; docs/experiments/2026-11-14-larger-held-out-splits.md). The cold
search is policy-independent (one run).

The cold search at 10 / 30 / 100 renders comes from
docs/experiments/2026-11-17-scripts/cold_budget.py (same episodes and shell);
the warm starts are measured at 0 / 30 / 200. Dark theme, for the README.

    .venv/bin/python docs/figures/make_budget_curve.py
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVAL_DIR = "out/rl_mb_v3d/eval_extra"
COLD_DIR = "out/cold_budget"
OUT = "docs/figures/budget_curve"
SPLIT = "test_extra"
EPISODES = 600
# Dark surface (GitHub's dark page colour, so the figure sits flush in the
# README); the three line colours are lightened for >= 4.5:1 against it, and
# every line also has its own marker and an end label.
INK, MUTED, GRID, SURFACE = "#e6edf3", "#9198a1", "#2a3038", "#0d1117"
# (label, eval keys, budgets, colour, marker, end-label vertical offset in points)
SERIES = [
    ("policy + planning, then search", ["plan", "plan_warm30", "plan_warm200"], [0, 30, 200], "#58a6ff", "o", 6),
    ("policy, then search", ["policy", "warm30", "warm200"], [0, 30, 200], "#ff8a4c", "s", -6),
    ("search from the start", [None, "cold10", "cold30", "cold100", "cold200"], [0, 10, 30, 100, 200],
     "#3fcf8e", "D", 0),
]


def rows(seed):
    out = []
    for f in sorted(glob.glob(os.path.join(EVAL_DIR, f"d{seed}_{SPLIT}_*.json"))):
        out.extend(json.load(open(f)))
    return out


def cold_rows():
    out = []
    for f in sorted(glob.glob(os.path.join(COLD_DIR, f"{SPLIT}_*.json"))):
        out.extend(json.load(open(f)))
    return out


def main():
    per_seed = {s: rows(s) for s in (0, 1, 2)}
    cold = cold_rows()
    assert len(cold) == EPISODES, f"{len(cold)} cold rows in {COLD_DIR}"
    # The cold search is policy-independent: one run, the same for every seed.
    for s in per_seed:
        by_i = {r["i"]: r for r in cold}
        for r in per_seed[s]:
            # the recomputed cold200 must match the evaluation's: same episodes, same shell
            assert abs(by_i[r["i"]]["cold200"] - r["cold200"]) < 1e-6, r["i"]
            for key in ("cold10", "cold30", "cold100"):
                r[key] = by_i[r["i"]][key]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(6.2, 3.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for label, keys, budgets, color, marker, dy in SERIES:
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
        ax.plot(x, mean, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(f"{mean[-1]:.2f}", (x[-1], mean[-1]), xytext=(7, dy), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
        if key_point := dict(zip(budgets, mean)).get(30) if label.startswith("policy + planning") else None:
            ax.annotate("30 renders here ≈ 200 from the start", (30, key_point), xytext=(-62, -78), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})
        print(f"{label:<34}", " ".join(f"{b}:{m:.3f}" for b, m in zip(budgets, mean)))
    # The headline: 30 renders from the policy + planning reach what the cold search needs 200 for.
    cold200 = np.mean([float(np.median([r["cold200"] for r in per_seed[s] if "cold200" in r]))
                       for s in (0, 1, 2) if any("cold200" in r for r in per_seed[s])])
    ax.axhline(cold200, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.2, cold200 + 0.018, f"search from the start, 200 renders ({cold200:.2f})", fontsize=8.5, color=MUTED)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([0, 10, 30, 100, 200])
    ax.set_xticklabels(["0", "10", "30", "100", "200"])
    ax.set_xlim(-0.5, 200)
    ax.set_ylim(0, 1)
    ax.set_xlabel("renders spent on search (log scale)")
    ax.set_ylabel(f"median attainment (60 patients, {EPISODES} episodes)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.subplots_adjust(right=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
