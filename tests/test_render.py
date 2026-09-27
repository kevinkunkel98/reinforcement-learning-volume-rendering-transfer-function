import numpy as np
import pytest
from phantom import build_phantom
from transfer import ANATOMICAL_PEAK_INDEX, N_PEAKS, PARAMS_PER_PEAK, default_params
import render as render_module
import views
from render import render, grab, features
from anatomy import CANONICAL_CLASSES


CLASS_IDS = {name: index for index, name in enumerate(CANONICAL_CLASSES, 1)}


TOP_CAMERA = {"position": (0.0, 0.0, 80.0), "focal_point": (0.0, 0.0, 0.0),
              "view_up": (0.0, 1.0, 0.0), "parallel_scale": 20.0}


def _only_peaks(*names, height=0.9):
    """Every peak off except `names`, each at `height` (unit scale)."""
    params = default_params().copy()
    for i in range(N_PEAKS):
        params[i * PARAMS_PER_PEAK + 2] = -1.0
    for name in names:
        params[ANATOMICAL_PEAK_INDEX[name] * PARAMS_PER_PEAK + 2] = 2.0 * height - 1.0
    return params


def _liver_kidney_volume(hu=50.0, n=20):
    volume = np.full((n, n, n), hu, dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[: n // 2, :, :] = CLASS_IDS["liver"]
    labels[n // 2:, :, :] = CLASS_IDS["kidneys"]
    return volume, labels


def _halves(image):
    """Summed brightness of the image's left and right column halves."""
    gray = image.astype(np.float64).sum(axis=2)
    middle = gray.shape[1] // 2
    return gray[:, :middle].sum(), gray[:, middle:].sum()


def test_label_scoped_volume_puts_each_label_in_its_own_band():
    render_module.clear_pipeline_cache()
    volume = np.full((6, 6, 6), 40.0, dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[3:] = CLASS_IDS["liver"]
    banded = render_module.label_scoped_volume(volume, labels)
    lo = render_module.CENTER_RANGE[0]
    band = render_module.LABEL_BAND

    assert banded[labels == 0] == pytest.approx(40.0 - lo, abs=1.0)
    assert banded[labels == CLASS_IDS["liver"]] == pytest.approx(
        40.0 - lo + CLASS_IDS["liver"] * band, abs=1.0)


def test_label_scoped_volume_is_cached_independent_of_layers():
    render_module.clear_pipeline_cache()
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    labels = np.ones(volume.shape, dtype=np.uint8)

    first = render_module.label_scoped_volume(volume, labels)
    again = render_module.label_scoped_volume(volume, labels)

    assert first is again
    assert len(render_module._MASKED_VOLUME_CACHE) <= render_module.MASKED_VOLUME_CACHE_SIZE


def test_equal_hu_organs_are_shaded_by_their_own_peak():
    render_module.clear_pipeline_cache()
    volume, labels = _liver_kidney_volume()
    # Centred on the volume, so the liver and kidney halves fall on opposite
    # image halves.
    camera = {**TOP_CAMERA, "position": (9.5, 9.5, 80.0), "focal_point": (9.5, 9.5, 9.5)}
    liver = _halves(grab(render(volume, _only_peaks("liver"), camera=camera, labels=labels)))
    kidneys = _halves(grab(render(volume, _only_peaks("kidneys"), camera=camera, labels=labels)))

    # Each peak lights one half, and the two light opposite halves.
    assert max(liver) > 20 * max(min(liver), 1.0)
    assert max(kidneys) > 20 * max(min(kidneys), 1.0)
    assert np.argmax(liver) != np.argmax(kidneys)


def test_layer_opacity_scales_its_class():
    render_module.clear_pipeline_cache()
    volume, labels = _liver_kidney_volume()
    params = _only_peaks("liver", "kidneys")
    visible = grab(render(volume, params, camera=TOP_CAMERA, labels=labels))
    hidden = grab(render(volume, params, camera=TOP_CAMERA, labels=labels,
                         layers={"liver": {"opacity": 0.0, "rgb": [1.0, 0.0, 0.0]}}))

    assert hidden.sum() < visible.sum()
    assert hidden.sum() > 0


def test_unlabelled_voxels_ignore_organ_peaks():
    render_module.clear_pipeline_cache()
    volume = np.full((12, 12, 12), 50.0, dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    organ = grab(render(volume, _only_peaks("liver"), camera=TOP_CAMERA, labels=labels))
    soft = grab(render(volume, _only_peaks("soft"), camera=TOP_CAMERA, labels=labels))

    assert organ.sum() == 0
    assert soft.sum() > 0


def test_labelled_pipeline_holds_a_single_volume():
    render_module.clear_pipeline_cache()
    volumes = [np.full((12, 12, 12), value, dtype=np.float32) for value in (10.0, 20.0)]
    labels = [np.full(volume.shape, CLASS_IDS["liver"], dtype=np.uint8) for volume in volumes]

    for index in (0, 1, 0):
        render(volumes[index], default_params(), labels=labels[index])

    assert len(render_module._PIPELINE_CACHE) == 2
    for entry in render_module._PIPELINE_CACHE.values():
        assert entry[1].GetVolumes().GetNumberOfItems() == 1


def test_clear_pipeline_cache_releases_labeled_pipeline():
    render_module.clear_pipeline_cache()
    volume = np.full((12, 12, 12), 10.0, dtype=np.float32)
    labels = np.full(volume.shape, CLASS_IDS["liver"], dtype=np.uint8)

    render(volume, default_params(), labels=labels)
    assert render_module._PIPELINE_CACHE

    render_module.clear_pipeline_cache()

    assert not render_module._PIPELINE_CACHE
    assert not render_module._MASKED_VOLUME_CACHE
    render(volume, default_params(), labels=labels)


def test_render_grab_shape_and_dtype():
    vol = build_phantom(size=48)
    win = render(vol, default_params())
    img = grab(win)
    assert img.shape == (render_module.HEIGHT, render_module.WIDTH, 3)
    assert img.dtype == np.uint8


def test_render_deterministic():
    vol = build_phantom(size=48)
    win1 = render(vol, default_params())
    win2 = render(vol, default_params())
    img1, img2 = grab(win1), grab(win2)
    assert np.array_equal(img1, img2)


def test_features_keys_and_ranges():
    vol = build_phantom(size=48)
    img = grab(render(vol, default_params()))
    f = features(img)
    assert set(f.keys()) == {"mean", "std", "coverage", "entropy"}
    assert 0.0 <= f["coverage"] <= 1.0
    assert f["entropy"] >= 0.0


def test_render_accepts_custom_camera_and_defaults_match_old_behavior():
    # render() reuses one render window per volume (see render._get_pipeline),
    # so each render() call must be grab()bed before the next render() call
    # on the same volume -- the window is a mutable, reused resource now,
    # not an independent snapshot.
    volume = build_phantom(size=48)
    params = default_params()

    img_default = grab(render(volume, params))
    img_custom = grab(render(volume, params, camera={"azimuth": 90.0, "elevation": 0.0, "zoom": 1.0}))
    assert not np.array_equal(img_default, img_custom)


def test_render_accepts_a_renderer_neutral_camera():
    volume = build_phantom()
    params = default_params()
    cameras = views.cameras_for_volume(volume, (1.0, 1.0, 1.0))
    front = grab(render(volume, params, (1.0, 1.0, 1.0), cameras[0]))
    side = grab(render(volume, params, (1.0, 1.0, 1.0), cameras[2]))
    assert front.shape == side.shape
    assert front.max() > 0                       # something is visible
    assert not np.array_equal(front, side)


def test_render_still_accepts_the_azimuth_camera():
    volume = build_phantom()
    params = default_params()
    image = grab(render(volume, params, (1.0, 1.0, 1.0),
                        {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}))
    assert image.shape[2] == 3 and image.max() > 0


def test_same_camera_renders_deterministically():
    volume = build_phantom()
    params = default_params()
    camera = views.cameras_for_volume(volume, (1.0, 1.0, 1.0))[1]
    first = grab(render(volume, params, (1.0, 1.0, 1.0), camera))
    second = grab(render(volume, params, (1.0, 1.0, 1.0), camera))
    assert np.array_equal(first, second)


def _camera_state(win):
    cam = win.GetRenderers().GetFirstRenderer().GetActiveCamera()
    return np.array(list(cam.GetPosition()) + list(cam.GetFocalPoint()))


def test_relative_camera_reframes_with_the_transfer_function_without_frame_bounds():
    # The behaviour this documents is the problem: with no frame_bounds, the
    # camera is reset to whatever the *current* transfer function makes
    # visible, so two steps of one conversation are framed differently and
    # can't be compared side by side.
    volume = build_phantom(size=48)
    everything = default_params()
    almost_nothing = default_params()
    for i in range(4):
        almost_nothing[i * 6 + 2] = -0.98  # crush every peak's height
    camera = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}

    wide = _camera_state(render(volume, everything, camera=camera))
    narrow = _camera_state(render(volume, almost_nothing, camera=camera))
    assert not np.allclose(wide, narrow)


def test_frame_bounds_pins_the_camera_across_transfer_function_changes():
    volume = build_phantom(size=48)
    everything = default_params()
    almost_nothing = default_params()
    for i in range(4):
        almost_nothing[i * 6 + 2] = -0.98
    camera = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    bounds = render_module.frame_bounds(volume, everything, (1.0, 1.0, 1.0))

    wide = _camera_state(render(volume, everything, camera=camera, frame_bounds=bounds))
    narrow = _camera_state(render(volume, almost_nothing, camera=camera, frame_bounds=bounds))
    assert np.allclose(wide, narrow)


def test_frame_bounds_still_honours_camera_movement():
    volume = build_phantom(size=48)
    params = default_params()
    bounds = render_module.frame_bounds(volume, params, (1.0, 1.0, 1.0))
    straight = _camera_state(render(volume, params, camera={"azimuth": 0.0, "elevation": 0.0, "zoom": 1.0},
                                     frame_bounds=bounds))
    rotated = _camera_state(render(volume, params, camera={"azimuth": 90.0, "elevation": 0.0, "zoom": 1.0},
                                    frame_bounds=bounds))
    assert not np.allclose(straight, rotated)


def test_same_camera_dict_renders_the_same_viewpoint_every_time():
    # render() reuses one renderer per volume and Azimuth/Elevation are
    # relative, so without re-seating the camera the rotation accumulated:
    # each command in a conversation quietly spun the volume another 30
    # degrees, making before/after steps impossible to compare.
    volume = build_phantom(size=48)
    params = default_params()
    camera = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    first = _camera_state(render(volume, params, camera=camera))
    for _ in range(3):
        again = _camera_state(render(volume, params, camera=camera))
        assert np.allclose(first, again)


def test_invalid_anatomical_layers_are_rejected():
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="anatomy_layers"):
        render(volume, default_params(), labels=labels, layers={"unknown": {}})


def test_labels_none_matches_legacy_hu_only_output():
    volume = build_phantom(size=24)
    params = default_params()
    camera = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    old = grab(render(volume, params, camera=camera))
    explicit_none = grab(render(volume, params, camera=camera, labels=None, layers=None))
    assert np.array_equal(old, explicit_none)
