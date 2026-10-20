import glob
import io
import os

import numpy as np
from PIL import Image

import collect_images
import goals
from collect_images import grid_data_url, view_grid


def _stub_frames(calls, size=100, n=6):
    def render_views(volume_id, params):
        calls.append((volume_id, tuple(np.asarray(params, dtype=np.float64).tolist())))
        rng = np.random.default_rng(len(calls))
        return [rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8) for _ in range(n)]
    return render_views


def test_grid_has_expected_pixel_size(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collect_images, "_render_views", _stub_frames(calls))

    png_bytes = view_grid("synthetic", goals.starting_params(), size=224, columns=3,
                          cache_dir=str(tmp_path))

    image = Image.open(io.BytesIO(png_bytes))
    assert image.size == (224 * 3, 224 * 2)


def test_cache_is_written_once_and_reused(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collect_images, "_render_views", _stub_frames(calls))

    first = view_grid("synthetic", goals.starting_params(), cache_dir=str(tmp_path))
    second = view_grid("synthetic", goals.starting_params(), cache_dir=str(tmp_path))

    assert first == second
    assert len(calls) == 1


def test_a_different_transfer_function_produces_a_different_cache_key(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collect_images, "_render_views", _stub_frames(calls))

    params_a = goals.starting_params()
    params_b = params_a.copy()
    params_b[2] = params_b[2] + 0.5

    view_grid("synthetic", params_a, cache_dir=str(tmp_path))
    view_grid("synthetic", params_b, cache_dir=str(tmp_path))

    assert len(calls) == 2
    assert len(glob.glob(os.path.join(str(tmp_path), "*.png"))) == 2


def test_label_identity_changes_collect_image_cache_key():
    params = goals.starting_params()
    assert collect_images._cache_key("synthetic", "v1", params, 224, 3) != collect_images._cache_key(
        "synthetic", "v1", params, 224, 3, "anatomy-v2|dims=2,3,4|classes=liver|version=labels-v1")


def test_no_tmp_file_remains(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collect_images, "_render_views", _stub_frames(calls))

    view_grid("synthetic", goals.starting_params(), cache_dir=str(tmp_path))

    assert glob.glob(os.path.join(str(tmp_path), "*.tmp")) == []


def test_grid_data_url_is_a_base64_png_data_url(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collect_images, "_render_views", _stub_frames(calls))

    url = grid_data_url("synthetic", goals.starting_params(), cache_dir=str(tmp_path))

    assert url.startswith("data:image/png;base64,")


def test_crop_columns_cover_the_axial_diagonal_and_stay_centred():
    volume = np.zeros((300, 200, 600))
    left, right = collect_images._crop_columns(volume, (1.0, 1.0, 1.0), width=1024, height=800)
    px_per_mm = 800 / (2 * collect_images.views.PARALLEL_SCALE_FACTOR * 600)
    assert right - left >= np.hypot(300, 200) * px_per_mm
    assert left + right == 1024
    # a volume wider than the frame is clamped to it
    assert collect_images._crop_columns(np.zeros((900, 900, 100)), (1.0, 1.0, 1.0), 1024, 800) == (0, 1024)


def test_grid_keeps_the_frame_aspect_ratio(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_images, "_render_views",
                        lambda volume_id, params: [np.zeros((200, 100, 3), dtype=np.uint8)] * 6)
    image = Image.open(io.BytesIO(view_grid("synthetic", goals.starting_params(), size=320, columns=3,
                                            cache_dir=str(tmp_path))))
    assert image.size == (160 * 3, 320 * 2)
