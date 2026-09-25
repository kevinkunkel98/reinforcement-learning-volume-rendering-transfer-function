import os

import numpy as np
import pytest

import visibility
from anatomy import MEASURED_CLASSES
from transfer import CENTER_RANGE, N_PEAKS, WIDTH_RANGE, default_params, PARAMS_PER_PEAK
from transfer import ANATOMICAL_PEAK_INDEX


def _slab_volume(front_hu, back_hu, n=24):
    """Two slabs stacked along the anterior axis: front slab is nearer to view 0."""
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[:, n // 2:, :] = front_hu      # higher anterior index = nearer the front camera
    volume[:, : n // 2, :] = back_hu
    return volume


def _params(heights):
    """A transfer function with the given per-peak heights (unit scale), default widths."""
    params = default_params().copy()
    # Keep compact test calls readable while targeting canonical peaks.
    peak_order = ("lungs", "soft", "heart", "skeleton")
    for height, name in zip(heights, peak_order):
        index = ANATOMICAL_PEAK_INDEX[name]
        params[index * PARAMS_PER_PEAK + 2] = height * 2.0 - 1.0
    return params


def _single_peak_params(center_hu, height=0.9, width_hu=80.0, peak=0):
    """A transfer function with one peak at an arbitrary HU center, the rest off."""
    params = default_params().copy()
    lo, hi = CENTER_RANGE
    wlo, whi = WIDTH_RANGE
    base = peak * PARAMS_PER_PEAK
    params[base + 0] = 2.0 * (center_hu - lo) / (hi - lo) - 1.0
    params[base + 1] = 2.0 * (width_hu - wlo) / (whi - wlo) - 1.0
    params[base + 2] = 2.0 * height - 1.0
    for i in range(N_PEAKS):
        if i != peak:
            params[i * PARAMS_PER_PEAK + 2] = -1.0
    return params


def _model(volume, spacing=(2.0, 2.0, 2.0), **kwargs):
    return visibility.VisibilityModel.from_volume(volume, spacing, **kwargs)


def _core_volume(core_hu, shell_hu, n=24):
    """A core wrapped on every side (in the axial plane) by another tissue, so
    every one of the 6 views looks through the shell to reach the core."""
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[n // 6: 5 * n // 6, n // 6: 5 * n // 6, :] = shell_hu
    volume[3 * n // 8: 5 * n // 8, 3 * n // 8: 5 * n // 8, :] = core_hu
    return volume


def _core_labels(class_name, n=24):
    """Labels matching _core_volume's core region as `class_name`, shell/rest 'other'."""
    labels = np.zeros((n, n, n), dtype=np.uint8)
    labels[3 * n // 8: 5 * n // 8, 3 * n // 8: 5 * n // 8, :] = visibility.CLASSES.index(class_name) + 1
    return labels


def test_transparent_transfer_function_is_invisible():
    model = _model(_slab_volume(700.0, 900.0))
    features = model.features(_params([0.0, 0.0, 0.0, 0.0]))
    assert features["coverage"] == 0.0
    for name in visibility.CLASSES:
        assert features["vis"][name] == pytest.approx(0.0, abs=5e-5)
        assert features["bright"][name] == pytest.approx(0.0, abs=1e-6)


def test_surrounding_tissue_occludes_the_core():
    # bone-HU core inside soft tissue, as ribs or a spine sit inside a body;
    # with no explicit labels, HU 900 falls into the skeleton fallback band.
    volume = _core_volume(core_hu=900.0, shell_hu=50.0)
    model = _model(volume)
    occluded = model.features(_params([0.0, 0.9, 0.0, 0.9]))["vis"]["skeleton"]
    cleared = model.features(_params([0.0, 0.0, 0.0, 0.9]))["vis"]["skeleton"]
    assert cleared > occluded * 2.0


def test_visibility_rises_with_opacity_until_it_saturates():
    model = _model(_slab_volume(-1000.0, 900.0))       # nothing in front of the bone
    values = [model.features(_params([0, 0, 0, h]))["vis"]["skeleton"] for h in (0.1, 0.3, 0.6)]
    assert values[0] < values[1] < values[2]


def test_brightness_follows_peak_colour():
    volume = _slab_volume(-1000.0, 900.0)
    model = _model(volume)
    dark = default_params().copy()
    bright = default_params().copy()
    for channel in range(3):
        index = ANATOMICAL_PEAK_INDEX["skeleton"]
        dark[index * PARAMS_PER_PEAK + 3 + channel] = -0.6
        bright[index * PARAMS_PER_PEAK + 3 + channel] = 1.0
    assert model.features(bright)["bright"]["skeleton"] > model.features(dark)["bright"]["skeleton"]


def test_coverage_counts_rays_that_accumulate_opacity():
    model = _model(_slab_volume(-1000.0, 900.0))
    thin = model.features(_params([0, 0, 0, 0.02]))["coverage"]
    thick = model.features(_params([0, 0, 0, 1.0]))["coverage"]
    assert 0.0 <= thin < thick <= 1.0


def test_features_reports_unlabeled_tissue_as_other():
    """A transfer function can render tissue that belongs to none of the five
    named classes (fat, connective tissue, partial-volume edges): HU -200
    falls in the gap the intensity fallback leaves unclassified, between
    lungs (<=-500) and organs (>=-30). That contribution has to be visible
    somewhere in features(), or a caller scoring only the five named classes
    cannot tell "everything hidden" from "an opaque wall of unclassified
    material" -- exactly what goals.distance's "other" keep term needs."""
    volume = _slab_volume(-1000.0, -200.0)
    model = _model(volume)
    params = _single_peak_params(center_hu=-200.0, height=0.9, width_hu=80.0, peak=0)
    features = model.features(params)
    assert features["coverage"] > 0.5
    assert sum(v for name, v in features["vis"].items() if name != "other") < 0.01
    assert features["vis"]["other"] > 0.3


def test_explicit_layer_opacity_populates_effective_fields():
    model = _model(_slab_volume(-1000.0, 900.0))
    params = _params([0.0, 0.0, 0.0, 0.9])
    raw = model.features(params)
    effective = model.features(params, {"skeleton": {"opacity": 0.0}})

    assert effective["vis"]["skeleton"] == pytest.approx(raw["vis"]["skeleton"])
    assert effective["effective_vis"]["skeleton"] == 0.0


def test_views_see_different_things():
    """Bone behind soft tissue is hidden from the front and open from behind."""
    n = 24
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[:, n // 2: 3 * n // 4, :] = 900.0     # bone in the middle
    volume[:, 3 * n // 4:, :] = 50.0             # soft tissue in front of it (anterior)
    model = _model(volume, n_views=2)            # view 0 anterior, view 1 posterior
    per_view = model.per_view_visibility(_params([0.0, 0.9, 0.0, 0.9]), "skeleton")
    assert len(per_view) == 2
    assert per_view[1] > per_view[0] * 1.5       # seen from behind, bone is not occluded


def test_solo_max_is_the_visibility_of_that_class_alone():
    model = _model(_slab_volume(50.0, 900.0))
    solo = model.solo_max("skeleton")
    assert solo > model.features(_params([0.0, 0.9, 0.0, 0.9]))["vis"]["skeleton"]
    assert 0.0 < solo <= 1.0


def test_histogram_is_normalised_and_shaped():
    model = _model(_slab_volume(50.0, 900.0))
    histogram = model.histogram
    assert histogram.shape == (visibility.HISTOGRAM_BINS,)
    assert histogram.sum() == pytest.approx(1.0)
    assert (histogram >= 0).all()


def test_classes_come_from_the_label_volume():
    """A bone-HU core carrying the 'organs' label shows up under organs, not
    skeleton -- labels, not intensities, decide the class."""
    volume = _core_volume(core_hu=900.0, shell_hu=50.0)
    labels = _core_labels("soft")
    model = visibility.VisibilityModel.from_volume(volume, (2.0, 2.0, 2.0), labels=labels)
    vis = model.features(_params([0.0, 0.0, 0.0, 0.9]))["vis"]
    assert vis["soft"] > 0.0
    assert vis["skeleton"] == pytest.approx(0.0, abs=1e-6)


def test_label_source_is_reported():
    volume = _slab_volume(50.0, 900.0)
    labels = np.zeros_like(volume, dtype=np.uint8)
    with_labels = _model(volume, labels=labels)
    without_labels = _model(volume)
    assert with_labels.label_source == "anatomy"
    assert without_labels.label_source == "intensity"


def test_intensity_fallback_maps_to_the_same_class_names():
    bone_model = _model(_slab_volume(-1000.0, 900.0))
    assert bone_model.label_source == "intensity"
    assert bone_model.features(_single_peak_params(900.0))["vis"]["skeleton"] > 0.0

    lung_model = _model(_slab_volume(-1000.0, -800.0))
    assert lung_model.features(_single_peak_params(-800.0))["vis"]["lungs"] > 0.0


def test_intensity_fallback_leaves_muscle_and_vessels_empty():
    model = _model(_slab_volume(-1000.0, 900.0))
    features = model.features(_single_peak_params(900.0))
    assert features["vis"]["skeleton"] > 0.0
    assert features["vis"]["vessels"] == pytest.approx(0.0, abs=1e-6)


def test_visibility_classes_use_shared_anatomy_registry():
    assert visibility.CLASSES == MEASURED_CLASSES[:-1]
    assert visibility.CLASSES == (
        "skeleton", "lungs", "heart", "vessels", "liver", "kidneys", "spleen", "soft"
    )


def test_intensity_fallback_only_populates_sensible_classes():
    model = _model(_slab_volume(-1000.0, 50.0))
    features = model.features(_single_peak_params(50.0))
    assert features["vis"]["soft"] > 0.0
    for name in ("heart", "vessels", "liver", "kidneys", "spleen"):
        assert features["vis"][name] == pytest.approx(0.0, abs=1e-6)


def test_solo_max_uses_the_best_single_peak():
    volume = _core_volume(core_hu=900.0, shell_hu=50.0)
    labels = _core_labels("skeleton")
    model = visibility.VisibilityModel.from_volume(volume, (2.0, 2.0, 2.0), labels=labels)
    solo = model.solo_max("skeleton")
    assert solo > 0.0
    assert solo >= model.features(default_params())["vis"]["skeleton"]


def test_features_are_deterministic():
    model = _model(_slab_volume(50.0, 900.0))
    params = _params([0.2, 0.4, 0.1, 0.7])
    assert model.features(params) == model.features(params)


def test_features_report_hu_and_effective_layer_contributions_separately():
    volume = _slab_volume(-1000.0, 900.0)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[:, : volume.shape[1] // 2, :] = visibility.CLASSES.index("liver") + 1
    model = _model(volume, labels=labels)
    params = _single_peak_params(900.0)

    features = model.features(params, {"liver": {"opacity": 0.0}})

    assert features["vis"]["liver"] > 0.0
    assert features["effective_vis"]["liver"] == pytest.approx(0.0)
    assert features["effective_class_contribution"]["liver"] == pytest.approx(0.0)
    assert features["effective_vis"]["other"] == pytest.approx(features["vis"]["other"])


def test_layer_isolation_metrics_measure_target_and_cross_class_leakage():
    volume = _slab_volume(900.0, 900.0)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[:, : volume.shape[1] // 2, :] = visibility.CLASSES.index("liver") + 1
    labels[:, volume.shape[1] // 2 :, :] = visibility.CLASSES.index("kidneys") + 1
    model = _model(volume, labels=labels)

    metrics = model.layer_metrics(_single_peak_params(900.0), "liver")

    assert metrics["target_contribution"] > 0.0
    assert metrics["isolation"] == pytest.approx(1.0)
    assert metrics["cross_class_leakage"] == pytest.approx(0.0)


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
    assert loaded.label_source == built.label_source
    assert np.array_equal(loaded.class_ids, built.class_ids)


def test_cache_miss_on_different_version(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    visibility.VisibilityModel.from_volume(_slab_volume(50.0, 900.0), (2.0, 2.0, 2.0),
                                           volume_id="fake").save_cache("version-1")
    assert visibility.VisibilityModel.load_cache("fake", "version-2") is None
    assert visibility.VisibilityModel.load_cache("other", "version-1") is None


def test_cache_key_uses_versioned_anatomy_layout():
    model = _model(_slab_volume(50.0, 900.0))
    assert visibility.CLASS_LAYOUT_VERSION == "anatomy-v2"
    assert visibility.CACHE_VERSION >= 3
    assert len(model.cache_key("version-1")) == 16


def test_cache_key_includes_layer_layout_renderer_and_label_metadata():
    model = _model(_slab_volume(50.0, 900.0))
    base = model.cache_key("version-1", "labels-v1")
    assert base != model.cache_key("version-1", "labels-v2")
    assert base != model.cache_key("version-1", "labels-v1", layer_layout="other")
    assert base != model.cache_key("version-1", "labels-v1", renderer_version="other")


def test_label_cache_identity_changes_visibility_key():
    model = _model(_slab_volume(50.0, 900.0))
    base = model.cache_key("volume-v1")
    labeled = model.cache_key("volume-v1", "anatomy-v2|dims=2,3,4|classes=liver|version=labels-v1")
    assert labeled != base


def test_label_file_digest_changes_label_cache_identity(monkeypatch):
    model = _model(_slab_volume(50.0, 900.0))
    first = model.cache_key("volume-v1", "labels-a")
    second = model.cache_key("volume-v1", "labels-b")
    assert first != second


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
    monkeypatch.setattr(visibility, "_has_labels", lambda name: False)
    first = visibility.for_volume("fake")
    second = visibility.for_volume("fake")
    assert len(calls) == 1
    assert first.features(_params([0, 0, 0, 0.5])) == second.features(_params([0, 0, 0, 0.5]))


def _stub_for_volume(monkeypatch, tmp_path, version="v1"):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(visibility, "_load_volume",
                         lambda name: (_slab_volume(50.0, 900.0), (2.0, 2.0, 2.0)))
    monkeypatch.setattr(visibility, "_volume_version", lambda name: version)
    monkeypatch.setattr(visibility, "_has_labels", lambda name: False)


def test_for_volume_returns_the_same_instance_so_solo_max_stays_warm(tmp_path, monkeypatch):
    """`solo_max` memoises on the instance, but a fresh instance per call threw
    that memo away every time -- four single-peak probes, ~190 ms, repaid on
    every policy answer. Reusing the instance is what makes the memo work."""
    visibility.clear_model_cache()
    _stub_for_volume(monkeypatch, tmp_path)

    first = visibility.for_volume("fake")
    first.solo_max("skeleton")
    second = visibility.for_volume("fake")

    assert second is first
    assert "skeleton" in second._solo_max


def test_for_volume_rebuilds_when_the_volume_version_changes(tmp_path, monkeypatch):
    """The cache must not outlive the data it describes."""
    visibility.clear_model_cache()
    _stub_for_volume(monkeypatch, tmp_path, version="v1")
    first = visibility.for_volume("fake")

    monkeypatch.setattr(visibility, "_volume_version", lambda name: "v2")
    second = visibility.for_volume("fake")

    assert second is not first


def test_model_cache_is_bounded(tmp_path, monkeypatch):
    """Training sweeps every volume in a split; an unbounded cache would hold
    all of them at ~6 MB each."""
    visibility.clear_model_cache()
    _stub_for_volume(monkeypatch, tmp_path)

    names = [f"vol{i}" for i in range(visibility.MODEL_CACHE_SIZE + 2)]
    for name in names:
        visibility.for_volume(name)

    assert len(visibility._MODEL_CACHE) == visibility.MODEL_CACHE_SIZE
    # the oldest is gone, the newest retained
    assert not any(key[0] == names[0] for key in visibility._MODEL_CACHE)
    assert any(key[0] == names[-1] for key in visibility._MODEL_CACHE)


def test_solo_max_probes_at_the_anatomical_peak_centres():
    """solo_max is the ceiling used for absolute levels ("high lungs" = 0.8 x
    solo_max) and fed to the policy as the achievable ceiling. It probes with
    single-peak transfer functions, and those must sit at the centres the rest
    of the system uses (transfer.anatomical_params: lungs -800 HU), not at the
    retired band layout's (default_params: peak 0 at -100 HU, i.e. fat). With
    the fat centre, a volume made entirely of lung parenchyma reports a lung
    ceiling of ~0 and every lung instruction becomes unsatisfiable.
    """
    n = 24
    volume = np.full((n, n, n), -800.0, dtype=np.float32)   # lung parenchyma throughout
    labels = np.full((n, n, n), visibility.CLASSES.index("lungs") + 1, dtype=np.uint8)
    model = _model(volume, labels=labels)

    assert model.solo_max("lungs") > 0.1, "lungs unreachable: probe peak is not at lung HU"


def test_solo_max_still_finds_bone():
    n = 24
    volume = np.full((n, n, n), 900.0, dtype=np.float32)
    labels = np.full((n, n, n), visibility.CLASSES.index("skeleton") + 1, dtype=np.uint8)
    model = _model(volume, labels=labels)

    assert model.solo_max("skeleton") > 0.1
