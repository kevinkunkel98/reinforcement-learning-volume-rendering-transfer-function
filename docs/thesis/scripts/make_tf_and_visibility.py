"""Two foundation/task figures on the start transfer function of ts_s0477:

- tf_start.png: the label-scoped transfer function (one Gaussian opacity peak
  per class, in the class's colour, plus the unlabelled shell: the skeleton
  and soft-tissue peaks at UNLABELLED_OPACITY) next to its rendering.
- visibility.png: the rendering from the visibility model's frontal view
  (views.py view 0, the same camera), which class dominates each of that
  view's rays in the compositing estimate (visibility.py), and the
  per-class visibility vector the learner sees.

    .venv/bin/python docs/thesis/scripts/make_tf_and_visibility.py
"""
import numpy as np
import torch

import thesis_style as ts
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

import goals
import scans
import transfer
import visibility
from anatomy import CANONICAL_PEAK_ORDER
from views import cameras_for_volume

SCAN = "ts_s0477"
LABELS = {"skeleton": "skeleton", "lungs": "lungs", "heart": "heart", "vessels": "vessels", "liver": "liver",
          "kidneys": "kidneys", "spleen": "spleen", "soft": "soft tissue", "other": "unlabelled"}
ROW_ORDER = ["lungs", "soft", "kidneys", "heart", "spleen", "liver", "vessels", "skeleton"]


def class_colour(params, name):
    if name == "other":
        return to_rgb("#8c959f")
    return transfer.peak_internal(params, transfer.ANATOMICAL_PEAK_INDEX[name])["rgb"]


def tf_figure(params, render):
    hu = np.linspace(*transfer.CENTER_RANGE, 3051)
    rows = transfer.label_scoped_tables(params, hu, CANONICAL_PEAK_ORDER)
    curves = {name: rows[i + 1][0] for i, name in enumerate(CANONICAL_PEAK_ORDER)}
    curves["other"] = rows[0][0]
    order = ROW_ORDER + ["other"]
    fig = plt.figure(figsize=(7.2, 3.9))
    grid = fig.add_gridspec(len(order), 2, width_ratios=[2.15, 1.0], hspace=0.15, wspace=0.08)
    for row, name in enumerate(order):
        ax = fig.add_subplot(grid[row, 0])
        colour = class_colour(params, name)
        ax.fill_between(hu, curves[name], color=colour, alpha=0.75, linewidth=0)
        ax.plot(hu, curves[name], color=np.array(colour) * 0.7, linewidth=0.8)
        ax.set_xlim(*transfer.CENTER_RANGE)
        ax.set_ylim(0, 0.5)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.text(-0.01, 0.25, LABELS[name], transform=ax.transAxes, ha="right", va="center", fontsize=7.6)
        if row < len(order) - 1:
            ax.set_xticks([])
            ax.spines["bottom"].set_color(ts.GRID)
        else:
            ax.set_xticks([-1000, -500, 0, 500, 1000, 1500, 2000])
            ax.set_xlabel("CT intensity (HU)")
        if row == 0:
            ax.set_title("opacity per class (peak height 0–0.5 per row)", fontsize=8, color=ts.MUTED, loc="left")
    ax = fig.add_subplot(grid[:, 1])
    ax.imshow(render)
    ax.axis("off")
    ax.set_title("rendering", fontsize=8, color=ts.MUTED)
    ts.save(fig, "tf_start.png")


def ray_classes(model, params, view=0):
    """(per-ray dominant class index into ['other'] + CLASSES, per-ray accumulated opacity) for one view."""
    alpha, _ = model._alpha_luminance(params)
    weights, accumulated = model._composite(alpha)
    w, labels = weights[view], model._labels[view].long()          # depth x u x v
    sums = torch.stack([(w * (labels == c - 1)).sum(dim=0) for c in range(len(visibility.CLASSES) + 1)])
    return sums.argmax(dim=0).numpy(), accumulated[view].numpy(), sums.numpy()


def visibility_figure(params, render, model):
    dominant, accumulated, _ = ray_classes(model, params)
    names = ["other"] + list(visibility.CLASSES)
    palette = np.array([class_colour(params, n) for n in names])
    image = palette[dominant]
    shade = np.clip(accumulated / 0.6, 0.0, 1.0)[..., None]          # fade rays that show almost nothing
    image = image * shade + (1.0 - shade)
    # The cube's (u, v) axes against the camera's (right, up): oriented to match the rendering.
    image = np.flipud(image)                                            # superior up, patient right on the left
    coverage = np.flipud(accumulated)
    rows, cols = np.where(coverage > 0.05)                              # crop to the body, as the rendering
    image = image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    vis = model.features(params)["vis"]
    fig = plt.figure(figsize=(7.4, 3.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.8, 0.8, 1.35], wspace=0.55)
    ax = fig.add_subplot(grid[0])
    ax.imshow(render)
    ax.axis("off")
    ax.set_title("rendering (view 0)", fontsize=8, color=ts.MUTED)
    ax = fig.add_subplot(grid[1])
    ax.imshow(image, interpolation="nearest")
    ax.axis("off")
    ax.set_title("dominant class per ray\n(80 × 80 estimate)", fontsize=8, color=ts.MUTED)
    ax = fig.add_subplot(grid[2])
    order = sorted(names, key=lambda n: vis[n])
    values = [100.0 * vis[n] for n in order]
    ax.barh([LABELS[n] for n in order], values, color=[class_colour(params, n) for n in order],
            edgecolor=ts.MUTED, linewidth=0.4)
    for y, v in enumerate(values):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=7, color=ts.INK)
    ax.set_xlabel("share of the image (%, six views)")
    ax.grid(axis="y", visible=False)
    ax.set_title("visibility vector", fontsize=8, color=ts.MUTED)
    ax.tick_params(axis="y", labelsize=7.6)
    ts.save(fig, "visibility.png")


def main():
    params = goals.starting_params()
    volume, spacing = scans.load_volume(SCAN)
    labels = scans.load_labels(SCAN)
    model = visibility.for_volume(SCAN)
    front = cameras_for_volume(volume, spacing)[0]
    render = ts.render_rgba(volume, params, spacing, front, None, labels)
    render = ts.crop(render, ts.crop_box([render], margin=6))
    tf_figure(params, render)
    visibility_figure(params, render, model)


if __name__ == "__main__":
    main()
