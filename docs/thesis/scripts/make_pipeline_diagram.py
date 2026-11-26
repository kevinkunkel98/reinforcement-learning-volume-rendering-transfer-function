"""The README's pipeline diagram (docs/figures/pipeline_rl_warmstart.drawio)
redrawn in the thesis's light theme with matplotlib, so it can be rebuilt
without draw.io.

Numbers: task v3 with fixed ceilings (v3c), test split (6 patients, 300
episodes), the median over the three seeds of each seed's median attainment
(docs/experiments/2026-10-19-task-v3.md, "Results after the ceiling fix"):
policy 0.48/0.47/0.47 -> 0.47, planning 0.62/0.65/0.64 -> 0.64,
+ 30 renders 0.77/0.78/0.77 -> 0.77, + 200 renders 0.91/0.92/0.90 -> 0.91,
search from the start 0.70. (The README diagram mixes seeds: 0.48, 0.90.)

    .venv/bin/python docs/thesis/scripts/make_pipeline_diagram.py
"""
import thesis_style as ts
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

POLICY, PLANNING, SEARCH30, SEARCH200, BASELINE = "0.47", "0.64", "0.77", "0.91", "0.70"
W, H = 200.0, 96.0


def panel(ax, x0, y0, x1, y1, title, face):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=face, edgecolor=ts.GRID, linewidth=1.0, zorder=0))
    ax.text(x0 + 2, y1 - 3, title, fontsize=10, fontweight="bold", va="top", color=ts.INK)


def box(ax, x0, y0, x1, y1, lines, rl=False, number=None):
    """`lines`: [(text, style)] with style in {"b", "", "i"}; `number`: big accent value at the bottom."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.8",
                                facecolor=ts.BLUE_FILL if rl else ts.PAGE,
                                edgecolor=ts.BLUE if rl else ts.MUTED, linewidth=1.1, zorder=2))
    rows = len(lines) + (1.6 if number else 0)
    step = 2.7
    y = (y0 + y1) / 2 + (rows - 1) * step / 2
    for text, style in lines:
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=7.9, zorder=3, color=ts.INK,
                fontweight="bold" if style == "b" else "normal", fontstyle="italic" if style == "i" else "normal")
        y -= step
    if number:
        ax.text((x0 + x1) / 2, y - 1.6, number, ha="center", va="center", fontsize=15, fontweight="bold",
                color=ts.BLUE, zorder=3)
    return (x0, y0, x1, y1)


def arrow(ax, p, q, color=ts.MUTED, style="-", width=1.1, label=None, label_at=0.5, rad=0.0, face=ts.PANEL):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, linewidth=width,
                                 linestyle=style, connectionstyle=f"arc3,rad={rad}", zorder=1,
                                 shrinkA=0, shrinkB=0))
    if label:
        x, y = p[0] + (q[0] - p[0]) * label_at, p[1] + (q[1] - p[1]) * label_at
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color=ts.INK, zorder=4,
                bbox=dict(facecolor=face, edgecolor="none", pad=1.2))


def poly(ax, points, color=ts.MUTED, style=(0, (3, 2.5)), width=1.0):
    """A dashed polyline ending in an arrow head (the draw.io orthogonal edges)."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, linestyle=style, linewidth=width, zorder=1)
    ax.plot(xs[-2:-1] + xs[-1:], ys[-2:-1] + ys[-1:], color=color, linestyle=style, linewidth=width, zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=10, color=color,
                                 linewidth=0, zorder=1, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, face=ts.PAGE):
    ax.text(x, y, text, ha="center", va="center", fontsize=8, color=ts.INK, zorder=4,
            bbox=dict(facecolor=face, edgecolor="none", pad=1.2))


def main():
    fig, ax = plt.subplots(figsize=(13.2, 6.35))
    ax.set_xlim(0, W)
    ax.set_ylim(-4, H)
    ax.axis("off")
    ax.text(2, H - 1.5, f"Model-based RL as a warm start: from {POLICY} to {SEARCH200}",
            fontsize=13, fontweight="bold", va="top")

    train_face, use_face = ts.PANEL, "#fbfcfd"
    panel(ax, 1, 50, 199, 89, "1 · Training (offline, once)", train_face)
    panel(ax, 1, 2, 199, 46, "2 · Use (per instruction)", use_face)

    data = box(ax, 6, 60, 40, 78, [("Training data", "b"), ("TF variants → real renders", ""),
                                   ("→ visibility & brightness", ""), ("per organ", "")])
    ens = box(ax, 62, 60, 98, 78, [("Ensemble surrogate model", "b"), ("(differentiable)", ""),
                                   ("TF → visibility, brightness", ""), ("a learned “renderer”", "i")], rl=True)
    pol = box(ax, 120, 60, 156, 78, [("Policy", "b"), ("instruction + start state → TF", ""),
                                     ("trained by gradients", "i"), ("through the model", "i")], rl=True)
    arrow(ax, (data[2], 69), (ens[0], 69), label="learns")
    arrow(ax, (ens[2], 69), (pol[0], 69), color=ts.BLUE, width=1.8, label="gradients\nthrough the model", face=ts.PANEL)
    # Dyna: from both RL boxes back to the training data
    poly(ax, [(92, 60), (92, 54), (23, 54), (23, 60)])
    poly(ax, [(151, 60), (151, 54), (93, 54)], style=(0, (3, 2.5)))
    label(ax, 92, 54, "Dyna: collect more real data where the ensemble disagrees", face=train_face)

    speech = box(ax, 3, 15, 22, 28, [("Speech / text", "b"), ("“more liver”", "")])
    parser = box(ax, 26, 15, 46, 28, [("Parser", "b"), ("→ goal per organ", "")])
    p1 = box(ax, 51, 12, 75, 31, [("Policy", "b"), ("(one-shot)", ""), ("0 renders", "")], rl=True, number=POLICY)
    p2 = box(ax, 80, 12, 107, 31, [("Planning", "b"), ("through the model", ""), ("0 renders, ≈0.1 s", "")],
             rl=True, number=PLANNING)
    p3 = box(ax, 112, 11, 141, 32, [("Search (hill climbing)", "b"), ("from this starting point", ""),
                                    (f"30 renders → {SEARCH30}", "")], rl=True, number=f"200 → {SEARCH200}")
    keep = box(ax, 146, 14, 170, 29, [("Keep Repair", "b"), ("+ Colour Guard", "b"),
                                      ("no side effects", "i"), ("on other organs", "i")])
    rend = box(ax, 175, 15, 196, 28, [("Rendering", "b"), ("browser / VR", "")])
    for a, b, rl in ((speech, parser, False), (parser, p1, False), (p1, p2, True), (p2, p3, True),
                     (p3, keep, False), (keep, rend, False)):
        arrow(ax, (a[2], 21.5), (b[0], 21.5), color=ts.BLUE if rl else ts.MUTED, width=1.6 if rl else 1.1)

    # trained policy and the model reach the use panel
    poly(ax, [(138, 60), (138, 48), (63, 48), (63, p1[3])], color=ts.BLUE, style=(0, (2, 2)))
    label(ax, 101, 48, "trained policy")
    poly(ax, [(75, 60), (75, 37), (93.5, 37), (93.5, p2[3])], style=(0, (2, 2)))
    label(ax, 84, 37, "model (for planning)", face=use_face)

    ax.text(3, 7.6, r"$\bf{Key\ result:}$ from the RL starting point, 30 renders"
            f" ({SEARCH30}) reach the level of\n200 renders of search from the start ({BASELINE});"
            " paired Wilcoxon p = 0.14–0.9 (level, not better)", fontsize=8.2, va="center", linespacing=1.4)
    ax.add_patch(FancyBboxPatch((146, 4.5), 50, 7, boxstyle="round,pad=0,rounding_size=1.4", facecolor=use_face,
                                edgecolor=ts.GREEN, linewidth=1.1, linestyle=(0, (3, 2))))
    ax.text(171, 8, f"Baseline: search from the start (no RL),\n200 renders → {BASELINE}", ha="center",
            va="center", fontsize=8.0, linespacing=1.3)
    ax.text(199, -1.2, "Median attainment, test split (6 patients, 300 episodes), median over 3 seeds · "
            "1 = goal reached, 0 = no progress", fontsize=7.0, color=ts.MUTED, ha="right", va="top")
    ts.save(fig, "pipeline.png", dpi=250)


if __name__ == "__main__":
    main()
