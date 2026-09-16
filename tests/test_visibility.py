import os

import numpy as np
import pytest

import visibility
from transfer import TISSUE_BANDS, default_params, PARAMS_PER_PEAK


def _slab_volume(front_hu, back_hu, n=24):
    """Two slabs stacked along the anterior axis: front slab is nearer to view 0."""
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[:, n // 2:, :] = front_hu      # higher anterior index = nearer the front camera
    volume[:, : n // 2, :] = back_hu
    return volume


def _params(heights):
    """A transfer function with the given per-peak heights (unit scale), default widths."""
    params = default_params().copy()
    for index, height in enumerate(heights):
        params[index * PARAMS_PER_PEAK + 2] = height * 2.0 - 1.0
    return params


def _model(volume, spacing=(2.0, 2.0, 2.0), **kwargs):
    return visibility.VisibilityModel.from_volume(volume, spacing, **kwargs)


def test_transparent_transfer_function_is_invisible():
    model = _model(_slab_volume(700.0, 900.0))
    features = model.features(_params([0.0, 0.0, 0.0, 0.0]))
    assert features["coverage"] == 0.0
    for tissue in visibility.TISSUES:
        assert features["vis"][tissue] == pytest.approx(0.0, abs=1e-6)
        assert features["bright"][tissue] == 0.0


def _core_volume(core_hu, shell_hu, n=24):
    """A core wrapped on every side (in the axial plane) by another tissue, so
    every one of the 6 views looks through the shell to reach the core."""
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[n // 6: 5 * n // 6, n // 6: 5 * n // 6, :] = shell_hu
    volume[3 * n // 8: 5 * n // 8, 3 * n // 8: 5 * n // 8, :] = core_hu
    return volume


def test_surrounding_tissue_occludes_the_core():
    # bone core inside soft tissue, as ribs or a spine sit inside a body
    volume = _core_volume(core_hu=900.0, shell_hu=50.0)
    model = _model(volume)
    occluded = model.features(_params([0.0, 0.9, 0.0, 0.9]))["vis"]["bone"]
    cleared = model.features(_params([0.0, 0.0, 0.0, 0.9]))["vis"]["bone"]
    assert cleared > occluded * 2.0


def test_visibility_rises_with_opacity_until_it_saturates():
    model = _model(_slab_volume(-1000.0, 900.0))       # nothing in front of the bone
    values = [model.features(_params([0, 0, 0, h]))["vis"]["bone"] for h in (0.1, 0.3, 0.6)]
    assert values[0] < values[1] < values[2]


def test_brightness_follows_peak_colour():
    volume = _slab_volume(-1000.0, 900.0)
    model = _model(volume)
    dark = default_params().copy()
    bright = default_params().copy()
    for channel in range(3):
        dark[3 * PARAMS_PER_PEAK + 3 + channel] = -0.6      # dim bone colour
        bright[3 * PARAMS_PER_PEAK + 3 + channel] = 1.0     # white bone colour
    assert model.features(bright)["bright"]["bone"] > model.features(dark)["bright"]["bone"]


def test_coverage_counts_rays_that_accumulate_opacity():
    model = _model(_slab_volume(-1000.0, 900.0))
    thin = model.features(_params([0, 0, 0, 0.02]))["coverage"]
    thick = model.features(_params([0, 0, 0, 1.0]))["coverage"]
    assert 0.0 <= thin < thick <= 1.0


def test_views_see_different_things():
    """Bone behind soft tissue is hidden from the front and open from behind."""
    n = 24
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[:, n // 2: 3 * n // 4, :] = 900.0     # bone in the middle
    volume[:, 3 * n // 4:, :] = 50.0             # soft tissue in front of it (anterior)
    model = _model(volume, n_views=2)            # view 0 anterior, view 1 posterior
    per_view = model.per_view_visibility(_params([0.0, 0.9, 0.0, 0.9]), "bone")
    assert len(per_view) == 2
    assert per_view[1] > per_view[0] * 1.5       # seen from behind, bone is not occluded


def test_solo_max_is_the_visibility_of_that_tissue_alone():
    model = _model(_slab_volume(50.0, 900.0))
    solo = model.solo_max("bone")
    assert solo > model.features(_params([0.0, 0.9, 0.0, 0.9]))["vis"]["bone"]
    assert 0.0 < solo <= 1.0


def test_histogram_is_normalised_and_shaped():
    model = _model(_slab_volume(50.0, 900.0))
    histogram = model.histogram
    assert histogram.shape == (visibility.HISTOGRAM_BINS,)
    assert histogram.sum() == pytest.approx(1.0)
    assert (histogram >= 0).all()


def test_labels_follow_tissue_bands():
    lo, hi = TISSUE_BANDS["bone"]
    model = _model(_slab_volume(-1000.0, (lo + hi) / 2.0))
    assert model.features(_params([0, 0, 0, 0.9]))["vis"]["bone"] > 0.0
    assert model.features(_params([0, 0, 0.9, 0]))["vis"]["spongy"] == pytest.approx(0.0, abs=1e-6)


def test_features_are_deterministic():
    model = _model(_slab_volume(50.0, 900.0))
    params = _params([0.2, 0.4, 0.1, 0.7])
    assert model.features(params) == model.features(params)


def test_cache_round_trip_reproduces_features(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    volume = _slab_volume(50.0, 900.0)
    params = _params([0.2, 0.5, 0.1, 0.8])
    built = visibility.VisibilityModel.from_volume(volume, (2.0, 2.0, 2.0), volume_id="fake")
    path = built.save_cache("version-1")
    assert os.path.exists(path)
    loaded = visibility.VisibilityModel.load_cache("fake", "version-1")
    assert loaded is not None
    assert loaded.features(params) == built.features(params)
    assert loaded.step_mm == built.step_mm
    assert np.array_equal(loaded.histogram, built.histogram)


def test_cache_miss_on_different_version(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    visibility.VisibilityModel.from_volume(_slab_volume(50.0, 900.0), (2.0, 2.0, 2.0),
                                           volume_id="fake").save_cache("version-1")
    assert visibility.VisibilityModel.load_cache("fake", "version-2") is None
    assert visibility.VisibilityModel.load_cache("other", "version-1") is None


def test_for_volume_builds_once_then_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    calls = []
    real_from_volume = visibility.VisibilityModel.from_volume

    def counting(volume, spacing, **kwargs):
        calls.append(1)
        return real_from_volume(volume, spacing, **kwargs)

    monkeypatch.setattr(visibility.VisibilityModel, "from_volume", staticmethod(counting))
    monkeypatch.setattr(visibility, "_load_volume", lambda name: (_slab_volume(50.0, 900.0), (2.0, 2.0, 2.0)))
    monkeypatch.setattr(visibility, "_volume_version", lambda name: "v1")
    first = visibility.for_volume("fake")
    second = visibility.for_volume("fake")
    assert len(calls) == 1
    assert first.features(_params([0, 0, 0, 0.5])) == second.features(_params([0, 0, 0, 0.5]))
