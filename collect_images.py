"""Render the six standard views of a transfer function as one grid image,
for the preference-collection page (`collect.py`): the rater judges exactly
what the reward model will later see, not a live 3D viewport.

`_render_views` is the one seam that touches VTK/dataset loading -- tests
stub it out, so `view_grid`'s caching and grid-composition logic is testable
without a real renderer. A labelled volume renders label-scoped with the
current task's unlabelled-shell opacity (`transfer.UNLABELLED_OPACITY`, set by
`goals.use_task_version`), exactly as the viewer shows it.
"""
import base64
import hashlib
import io
import os

import numpy as np
from PIL import Image

import render as render_module
import totalseg
import transfer
import views
from anatomy_layers import default_layers
from datasets import _dataset_version, label_cache_identity, load_dataset

CACHE_DIR = "out/cache/collect_images"
# 2: label-scoped renders with the task's shell opacity, frames cropped
# instead of squeezed into a square (2026-10-19).
CACHE_VERSION = 2
CROP_MARGIN = 1.08          # around the widest horizontal extent the six views can show


def _cache_key(volume_id: str, volume_version: str, params: np.ndarray, size: int,
               columns: int, label_identity: str = "") -> str:
    param_str = ",".join(f"{v:.6f}" for v in np.asarray(params, dtype=np.float64).tolist())
    payload = "|".join([volume_id, volume_version, label_identity, param_str,
                         str(size), str(columns), f"{transfer.UNLABELLED_OPACITY:.3f}",
                         str(CACHE_VERSION)])
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(cache_key: str, cache_dir: str = None) -> str:
    return os.path.join(cache_dir or CACHE_DIR, f"{cache_key}.png")


def _crop_columns(volume, spacing, width: int, height: int) -> tuple:
    """The pixel columns any of the six views can draw the volume in.

    The views turn around the head-to-foot axis with a parallel projection of
    half-height `views.PARALLEL_SCALE_FACTOR` x the largest extent, so the
    widest a volume can appear is its axial diagonal -- the same columns for
    every view and transfer function of a volume, so candidates stay
    comparable pixel for pixel."""
    extent = np.asarray(volume.shape, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
    px_per_mm = height / (2.0 * views.PARALLEL_SCALE_FACTOR * float(extent.max()))
    half = int(np.ceil(0.5 * float(np.hypot(extent[0], extent[1])) * CROP_MARGIN * px_per_mm))
    centre = width // 2
    return max(0, centre - half), min(width, centre + half)


def _render_views(volume_id: str, params: np.ndarray) -> list:
    """The six standard views of `volume_id` under `params`, one RGB array
    per `views.cameras_for_volume` camera, via `render.render`/`render.grab`,
    cropped to the columns the volume can occupy (`_crop_columns`)."""
    volume, spacing = load_dataset(volume_id, canonical=True)
    labels = totalseg.load_labels(volume_id) if totalseg.is_totalseg(volume_id) and \
        totalseg.has_labels(volume_id) else None
    layers = default_layers() if labels is not None else None
    cameras = views.cameras_for_volume(volume, spacing)
    frames = []
    for camera in cameras:
        win = render_module.render(volume, params, spacing, camera, labels=labels, layers=layers)
        frame = render_module.grab(win)
        left, right = _crop_columns(volume, spacing, frame.shape[1], frame.shape[0])
        frames.append(frame[:, left:right, :3])
    return frames


def _compose_grid(frames: list, size: int, columns: int) -> Image.Image:
    """`frames` as a grid `columns` wide, each cell `size` pixels high and
    as wide as the frame's own aspect ratio allows."""
    rows = -(-len(frames) // columns)  # ceil division
    first = np.asarray(frames[0])
    cell_w = max(1, round(size * first.shape[1] / first.shape[0]))
    grid = Image.new("RGB", (cell_w * columns, size * rows), (0, 0, 0))
    for index, frame in enumerate(frames):
        thumb = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((cell_w, size), Image.Resampling.BILINEAR)
        row, col = divmod(index, columns)
        grid.paste(thumb, (col * cell_w, row * size))
    return grid


def view_grid(volume_id: str, params: np.ndarray, size: int = 320, columns: int = 3,
              cache_dir: str = None, label_identity: str = "") -> bytes:
    """PNG bytes of the six standard views of `volume_id` under `params`,
    laid out `columns`-wide. Cached on disk under `<cache_dir>/<sha256 of
    volume version, params and size>.png`; written atomically (`.tmp` then
    `os.replace`) so a reader never sees a partial file."""
    volume_version = _dataset_version(volume_id)
    params = np.asarray(params, dtype=np.float64)
    if not label_identity:
        try:
            label_identity = label_cache_identity(volume_id)
        except FileNotFoundError:
            pass
    key = _cache_key(volume_id, volume_version, params, size, columns, label_identity)
    path = _cache_path(key, cache_dir)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()

    frames = _render_views(volume_id, params)
    grid = _compose_grid(frames, size, columns)
    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)
    os.replace(tmp_path, path)
    return png_bytes


def grid_data_url(volume_id: str, params: np.ndarray, size: int = 320, columns: int = 3,
                  cache_dir: str = None, label_identity: str = "") -> str:
    """`view_grid`'s PNG as a `data:image/png;base64,...` string, for
    embedding directly in a JSON response."""
    png_bytes = view_grid(volume_id, params, size=size, columns=columns,
                          cache_dir=cache_dir, label_identity=label_identity)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()
