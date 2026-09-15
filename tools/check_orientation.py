"""Write maximum-intensity projections of canonical (RAS) volumes for a visual
orientation check: in the coronal and sagittal panels the head (superior) must
be at the top; in the axial panel the front of the body (anterior) must be at
the top.

    python -m tools.check_orientation ct_chest ts_s0011
"""
import argparse
import os

import numpy as np
from PIL import Image

from datasets import load_dataset

OUT_DIR = "out/orientation"
PANEL_HEIGHT = 320
WINDOW_HU = (-200.0, 1000.0)


def projection_panels(volume: np.ndarray):
    """(coronal, sagittal, axial) uint8 images; row 0 is the top of the image."""
    lo, hi = WINDOW_HU
    v = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    to_u8 = lambda plane: (plane * 255).round().astype(np.uint8)
    coronal = v.max(axis=1).T[::-1]      # rows: superior -> inferior, cols: R axis
    sagittal = v.max(axis=0).T[::-1]     # rows: superior -> inferior, cols: A axis
    axial = v.max(axis=2).T[::-1]        # rows: anterior -> posterior, cols: R axis
    return to_u8(coronal), to_u8(sagittal), to_u8(axial)


def _panel_image(plane, row_mm, col_mm):
    width = max(1, round(PANEL_HEIGHT * plane.shape[1] * col_mm / (plane.shape[0] * row_mm)))
    return Image.fromarray(plane).resize((width, PANEL_HEIGHT))


def write_projection(name: str, out_dir: str = OUT_DIR) -> str:
    volume, (sx, sy, sz) = load_dataset(name, canonical=True)
    coronal, sagittal, axial = projection_panels(volume)
    panels = [_panel_image(coronal, sz, sx), _panel_image(sagittal, sz, sy), _panel_image(axial, sy, sx)]
    sheet = Image.new("L", (sum(p.width for p in panels) + 20 * (len(panels) - 1), PANEL_HEIGHT))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + 20
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        print(write_projection(name))


if __name__ == "__main__":
    main()
