"""Attainment over the render budget, test split (6 patients, 300 episodes),
v3c, median over the three seeds of each seed's median
(docs/experiments/2026-10-19-task-v3.md, "Results after the ceiling fix").
Budgets 0 / 30 / 200 are the ones evaluated there; the search from the start
was evaluated at 200 only on this split. Light theme.

    .venv/bin/python docs/thesis/scripts/make_budget.py
"""
import thesis_style as ts
import matplotlib.pyplot as plt

BUDGETS = [0, 30, 200]
SERIES = [("policy + planning, then search", [0.64, 0.77, 0.91], ts.BLUE, "o"),
          ("policy, then search", [0.47, 0.67, 0.88], ts.ORANGE, "s")]
COLD200 = 0.70


def main():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = [1, 30, 200]          # symlog-free: budget 0 drawn at 1 on the log axis
    for label, values, colour, marker in SERIES:
        ax.plot(x, values, color=colour, marker=marker, linewidth=2, label=label)
        ax.text(230, values[-1], f"{values[-1]:.2f}", va="center", fontsize=8, color=colour)
    ax.scatter([200], [COLD200], color=ts.GREEN, marker="D", zorder=3, label="search from the start")
    ax.axhline(COLD200, color=ts.GREEN, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(1.1, COLD200 + 0.02, f"search from the start, 200 renders ({COLD200:.2f})", fontsize=7.5,
            color=ts.GREEN)
    ax.set_xscale("log")
    ax.set_xticks(x, ["0", "30", "200"])
    ax.set_xlim(0.8, 320)
    ax.set_ylim(0, 1)
    ax.set_xlabel("renders spent on search (log scale)")
    ax.set_ylabel("median attainment (test)")
    ax.legend(loc="lower right", fontsize=7.6)
    ts.save(fig, "budget_curve.png")


if __name__ == "__main__":
    main()
