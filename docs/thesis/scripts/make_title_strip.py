"""Title-page strip: four instructions from the README gallery, side by side,
each with what the viewer renders for it. Every command runs from the task's
start on the default scan through the viewer's own path (hybrid parser,
policy + planning + the 200-evaluation refine) in a throwaway session.
Light theme, renders on a transparent background (thesis_style.render_rgba).

    .venv/bin/python docs/thesis/scripts/make_title_strip.py
"""
import os
import tempfile
import textwrap

import thesis_style as ts  # noqa: F401  (sets the working directory and style first)
import matplotlib.pyplot as plt

import visibility
from session import Session

SCAN = "ts_s0477"
COMMANDS = [
    "hide everything but the lungs and the heart",
    "remove everything except the bones",
    "I want to focus on the liver, hide everything else",
    "the soft tissue is in the way, fade it out",
]


def run():
    visibility.use_device("auto")
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        sess = Session(SCAN, path=os.path.join(tmp, "session.json"))
        for text in COMMANDS:
            sess.command(text, parser="hybrid", mode="policy", defer_refine=False)
            step = sess.state()["current"]
            image = ts.render_rgba(sess.volume, step["params"], sess.spacing, step["camera"],
                                   sess._frame_bounds, sess.labels)
            tiles.append((text, step.get("understood") or "", image))
            print(f"[title] {text!r} -> {step.get('understood')} ({step.get('parser_used')}, {step['mode']})")
            while sess.state()["cursor"] > 0:
                sess.back()
    return tiles


def draw(tiles):
    box = ts.crop_box([t[2] for t in tiles])
    fig, axes = plt.subplots(1, len(tiles), figsize=(len(tiles) * 2.05, 4.0))
    for ax, (text, understood, image) in zip(axes, tiles):
        ax.imshow(ts.crop(image, box))
        ax.axis("off")
        quoted = "\n".join(textwrap.wrap(f"“{text}”", width=31))
        # Anchored at the bottom, so one- and two-line instructions share a baseline.
        ax.text(0.5, 1.055, quoted, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.6, color=ts.INK, linespacing=1.15)
        ax.text(0.5, 1.01, f"→ {understood}", transform=ax.transAxes, ha="center", va="bottom",
                color=ts.MUTED, fontsize=6.6)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.84, bottom=0.0, wspace=0.06)
    ts.save(fig, "title_renders.png", dpi=300)


if __name__ == "__main__":
    import pickle
    cache = os.path.join("out", "thesis", "title_tiles.pkl")
    if os.path.exists(cache) and os.environ.get("RERENDER") != "1":
        tiles = pickle.load(open(cache, "rb"))
    else:
        tiles = run()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(tiles, open(cache, "wb"))
    draw(tiles)
