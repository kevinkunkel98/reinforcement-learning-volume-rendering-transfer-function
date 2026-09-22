"""Render the six standard views of a transfer function as one grid image,
for the preference-collection page (`collect.py`): the rater judges exactly
what the reward model will later see, not a live 3D viewport.

`_render_views` is the one seam that touches VTK/dataset loading -- tests
stub it out, so `view_grid`'s caching and grid-composition logic is testable
without a real renderer.
"""
import base64
import hashlib
import io
import os

import numpy as np
from PIL import Image

import render as render_module
import views
from datasets import _dataset_version, load_dataset

CACHE_DIR = "out/cache/collect_images"
CACHE_VERSION = 1


def _cache_key(volume_id: str, volume_version: str, params: np.ndarray, size: int, columns: int) -> str:
    param_str = ",".join(f"{v:.6f}" for v in np.asarray(params, dtype=np.float64).tolist())
    payload = "|".join([volume_id, volume_version, param_str, str(size), str(columns), str(CACHE_VERSION)])
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(cache_key: str, cache_dir: str = None) -> str:
    return os.path.join(cache_dir or CACHE_DIR, f"{cache_key}.png")


def _render_views(volume_id: str, params: np.ndarray) -> list:
    """The six standard views of `volume_id` under `params`, one RGB array
    per `views.cameras_for_volume` camera, via `render.render`/`render.grab`."""
    volume, spacing = load_dataset(volume_id, canonical=True)
    cameras = views.cameras_for_volume(volume, spacing)
    frames = []
    for camera in cameras:
        win = render_module.render(volume, params, spacing, camera)
        frames.append(render_module.grab(win))
    return frames


def _compose_grid(frames: list, size: int, columns: int) -> Image.Image:
    rows = -(-len(frames) // columns)  # ceil division
    grid = Image.new("RGB", (size * columns, size * rows), (0, 0, 0))
    for index, frame in enumerate(frames):
        thumb = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
        row, col = divmod(index, columns)
        grid.paste(thumb, (col * size, row * size))
    return grid


def view_grid(volume_id: str, params: np.ndarray, size: int = 224, columns: int = 3,
              cache_dir: str = None) -> bytes:
    """PNG bytes of the six standard views of `volume_id` under `params`,
    laid out `columns`-wide. Cached on disk under `<cache_dir>/<sha256 of
    volume version, params and size>.png`; written atomically (`.tmp` then
    `os.replace`) so a reader never sees a partial file."""
    volume_version = _dataset_version(volume_id)
    params = np.asarray(params, dtype=np.float64)
    key = _cache_key(volume_id, volume_version, params, size, columns)
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


def grid_data_url(volume_id: str, params: np.ndarray, size: int = 224, columns: int = 3,
                  cache_dir: str = None) -> str:
    """`view_grid`'s PNG as a `data:image/png;base64,...` string, for
    embedding directly in a JSON response."""
    png_bytes = view_grid(volume_id, params, size=size, columns=columns, cache_dir=cache_dir)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()
