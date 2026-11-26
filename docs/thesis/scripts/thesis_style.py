"""Shared light theme for the thesis figures (the README keeps its dark ones),
and volume renders with a transparent background.

A volume render on background colour b is C + (1 - A) * b, with C the
premultiplied colour and A the accumulated opacity. Rendering the same view
on black (C) and on white (C + 1 - A) therefore gives A exactly, and the
render becomes an RGBA image that sits on any page colour.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = os.path.join(ROOT, "docs", "thesis", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# GitHub's light palette, matching the README figures' dark one in hue.
PAGE, INK, MUTED, GRID, PANEL = "#ffffff", "#1f2328", "#59636e", "#d1d9e0", "#f6f8fa"
BLUE, ORANGE, GREEN, RED, PURPLE = "#0969da", "#bc4c00", "#1a7f37", "#cf222e", "#8250df"
BLUE_FILL = "#ddf4ff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.facecolor": PAGE, "figure.facecolor": PAGE,
    "axes.facecolor": PAGE,
})


def save(fig, name, dpi=220):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[figure] wrote {os.path.relpath(path, ROOT)}")
    return path


def render_rgba(volume, params, spacing, camera, frame_bounds, labels) -> np.ndarray:
    """The view as float RGBA in [0, 1] (unpremultiplied colour, alpha = opacity)."""
    import render as render_module
    planes = []
    for background in (0.0, 1.0):
        win = render_module.render(volume, params, spacing=spacing, camera=camera,
                                   frame_bounds=frame_bounds, labels=labels)
        renderer = win.GetRenderers().GetFirstRenderer()
        if renderer.GetBackground() != (background,) * 3:
            renderer.SetBackground(background, background, background)
            win.Render()
        planes.append(render_module.grab(win).astype(np.float64) / 255.0)
        renderer.SetBackground(0.0, 0.0, 0.0)   # leave the cached pipeline as render.py set it
    black, white = planes
    alpha = np.clip(1.0 - (white - black).mean(axis=2), 0.0, 1.0)
    colour = np.where(alpha[..., None] > 1e-3, black / np.maximum(alpha[..., None], 1e-3), 0.0)
    return np.dstack([np.clip(colour, 0.0, 1.0), alpha])


def crop_box(images, margin=10, threshold=0.04) -> tuple:
    """(left, top, right, bottom) around everything visible in any of the RGBA images."""
    mask = np.zeros(images[0].shape[:2], bool)
    for im in images:
        mask |= im[..., 3] > threshold
    rows, cols = np.where(mask)
    h, w = mask.shape
    return (max(cols.min() - margin, 0), max(rows.min() - margin, 0),
            min(cols.max() + margin, w), min(rows.max() + margin, h))


def crop(image, box):
    left, top, right, bottom = box
    return image[top:bottom, left:right]
