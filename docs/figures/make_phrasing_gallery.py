"""Phrasing gallery for the README: free-form instructions and what the viewer
renders for them. Every command runs from the task's start on the default
scan through the viewer's own path (hybrid parser, policy + planning + the
200-evaluation refine), in a throwaway session so a running viewer's history
is untouched. Needs Ollama for the phrasings the rule parser does not cover.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python docs/figures/make_phrasing_gallery.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from session import Session  # noqa: E402

SCAN = "ts_s0477"
OUT = "docs/figures/phrasing_gallery.png"
COMMANDS = [
    "I want to focus on the liver, hide everything else",
    "the heart is way too faint, crank it up",
    "the soft tissue is in the way, fade it out",
    "hide everything but the lungs and the heart",
    "remove everything except the bones",
    "liver and spleen both a lot more visible",
    "I'd like the lungs to be more see-through",
    "the picture is too dark overall",
]
COLS = 3
BACKGROUND, INK, MUTED = "#0b0b0c", "#f2f2ef", "#a3a29b"


def run() -> list:
    """(phrase, understood, parser, image) per command; the start first."""
    with tempfile.TemporaryDirectory() as tmp:
        sess = Session(SCAN, path=os.path.join(tmp, "session.json"))
        start = sess.state()["current"]
        tiles = [("Start", "the task's default transfer function", None, Image.open(start["image_path"]))]
        for text in COMMANDS:
            state = sess.command(text, parser="hybrid", mode="policy", defer_refine=False)
            step = state["current"]
            tiles.append((text, step.get("understood") or "", step.get("parser_used"),
                          Image.open(step["image_path"]).convert("RGB")))
            print(f"[gallery] {text!r} -> {step.get('understood')} ({step.get('parser_used')}, {step['mode']})")
            while sess.state()["cursor"] > 0:
                sess.back()
    return tiles


def crop_box(images, margin=12) -> tuple:
    """One box around everything non-black in any render, so every tile has the same framing."""
    mask = np.zeros(np.asarray(images[0]).shape[:2], bool)
    for im in images:
        mask |= np.asarray(im.convert("L")) > 12
    rows, cols = np.where(mask)
    h, w = mask.shape
    return (max(cols.min() - margin, 0), max(rows.min() - margin, 0),
            min(cols.max() + margin, w), min(rows.max() + margin, h))


def draw(tiles) -> None:
    box = crop_box([t[3] for t in tiles])
    rows = (len(tiles) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3.4, rows * 4.1), facecolor=BACKGROUND)
    for ax in axes.flat:
        ax.set_facecolor(BACKGROUND)
        ax.axis("off")
    for ax, (text, understood, parser, image) in zip(axes.flat, tiles):
        ax.imshow(image.crop(box))
        quoted = text if text == "Start" else f"“{text}”"
        ax.set_title(quoted, color=INK, fontsize=9.5, pad=18, wrap=True)
        note = understood if parser is None else f"→ {understood}  ·  {parser} parser"
        ax.text(0.5, 1.012, note, transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.04, hspace=0.22)
    fig.savefig(OUT, dpi=150, facecolor=BACKGROUND)
    print(f"[gallery] wrote {OUT}")


if __name__ == "__main__":
    draw(run())
