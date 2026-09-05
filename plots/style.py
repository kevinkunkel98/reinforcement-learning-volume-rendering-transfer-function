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
