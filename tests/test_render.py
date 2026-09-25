import numpy as np
import pytest
from phantom import build_phantom
from transfer import default_params
import render as render_module
import views
from render import render, grab, features
from anatomy import CANONICAL_CLASSES


CLASS_IDS = {name: index for index, name in enumerate(CANONICAL_CLASSES, 1)}


def test_label_aware_hu_volume_masks_disabled_classes():
    volume = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    labels = np.array([[[1, 2], [0, 1]], [[2, 0], [1, 2]]], dtype=np.uint8)
    masked = render_module.label_aware_volume(
        volume, labels, {"skeleton": {"opacity": 1.0, "rgb": [1, 1, 1]},
                         "lungs": {"opacity": 0.0, "rgb": [1, 1, 1]}}
    )
    assert masked[labels == CLASS_IDS["lungs"]].tolist() == [-1.0e6, -1.0e6, -1.0e6]
    assert masked[labels == CLASS_IDS["skeleton"]].tolist() == [-1.0e6, -1.0e6, -1.0e6]
    assert masked[labels == 0].tolist() == [2.0, 5.0]


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


def test_equal_hu_labels_render_independently_without_label_leakage():
    volume = np.full((20, 20, 20), 100.0, dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[:10, :, :] = CLASS_IDS["liver"]
    labels[10:, :, :] = CLASS_IDS["kidneys"]
    params = default_params()
    camera = {"position": (0.0, 0.0, 80.0), "focal_point": (0.0, 0.0, 0.0),
              "view_up": (0.0, 1.0, 0.0), "parallel_scale": 20.0}

    liver = {"liver": {"opacity": 1.0, "rgb": [1.0, 0.0, 0.0]},
             "kidneys": {"opacity": 0.0, "rgb": [0.0, 1.0, 0.0]}}
    kidneys = {"liver": {"opacity": 0.0, "rgb": [1.0, 0.0, 0.0]},
               "kidneys": {"opacity": 1.0, "rgb": [0.0, 1.0, 0.0]}}

    liver_img = grab(render(volume, params, camera=camera, labels=labels, layers=liver))
    kidney_img = grab(render(volume, params, camera=camera, labels=labels, layers=kidneys))

    assert liver_img[..., 0].sum() > liver_img[..., 1].sum()
    assert kidney_img[..., 1].sum() > kidney_img[..., 0].sum()


def test_label_zero_is_background_and_labels_are_nearest_sampled():
    volume = np.full((12, 12, 12), 100.0, dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[2:10, 2:10, 2:10] = CLASS_IDS["liver"]
    params = default_params()
    layers = {"liver": {"opacity": 1.0, "rgb": [1.0, 0.0, 0.0]}}

    image = grab(render(volume, params, labels=labels, layers=layers))
    assert image[..., 0].sum() > image[..., 1].sum()


def test_invalid_anatomical_layers_are_rejected():
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="anatomy_layers"):
        render(volume, default_params(), labels=labels, layers={"unknown": {}})


def test_disabling_liver_removes_only_liver_region_contribution():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[:10, :, :] = CLASS_IDS["liver"]
    params = default_params()
    for peak in range(4):
        params[peak * 6 + 2] = -0.98
    camera = {"position": (0.0, 0.0, 80.0), "focal_point": (0.0, 0.0, 0.0),
              "view_up": (0.0, 1.0, 0.0), "parallel_scale": 20.0}
    baseline = grab(render(volume, params, camera=camera))
    visible = grab(render(volume, params, camera=camera, labels=labels,
                          layers={"liver": {"opacity": 1.0, "rgb": [1.0, 0.0, 0.0]}}))
    hidden = grab(render(volume, params, camera=camera, labels=labels,
                        layers={"liver": {"opacity": 0.0, "rgb": [1.0, 0.0, 0.0]}}))
    assert np.count_nonzero(visible != baseline) > 0
    assert hidden[..., 0].sum() < visible[..., 0].sum()
    assert hidden[..., 0].sum() < baseline[..., 0].sum()


def test_label_zero_has_no_anatomical_pixels_when_hu_is_hidden():
    volume = np.zeros((12, 12, 12), dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    params = default_params()
    for peak in range(4):
        params[peak * 6 + 2] = -0.98
    baseline = grab(render(volume, params))
    image = grab(render(volume, params, labels=labels,
                        layers={"liver": {"opacity": 1.0, "rgb": [1.0, 0.0, 0.0]}}))
    assert np.array_equal(image, baseline)


def test_nearest_labels_do_not_mix_anatomical_class_colors():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[:10, :, :] = CLASS_IDS["liver"]
    labels[10:, :, :] = CLASS_IDS["kidneys"]
    params = default_params()
    for peak in range(4):
        params[peak * 6 + 2] = -0.98
    liver = grab(render(volume, params, labels=np.where(labels == CLASS_IDS["liver"], labels, 0), layers={
        "liver": {"opacity": 1.0, "rgb": [1.0, 0.0, 0.0]},
        "kidneys": {"opacity": 1.0, "rgb": [0.0, 1.0, 0.0]},
    }))
    kidneys = grab(render(volume, params, labels=np.where(labels == CLASS_IDS["kidneys"], labels, 0), layers={
        "liver": {"opacity": 1.0, "rgb": [1.0, 0.0, 0.0]},
        "kidneys": {"opacity": 1.0, "rgb": [0.0, 1.0, 0.0]},
    }))
    assert liver[..., 0].sum() > liver[..., 1].sum()
    assert kidneys[..., 1].sum() > kidneys[..., 0].sum()


def test_labels_none_matches_legacy_hu_only_output():
    volume = build_phantom(size=24)
    params = default_params()
    camera = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    old = grab(render(volume, params, camera=camera))
    explicit_none = grab(render(volume, params, camera=camera, labels=None, layers=None))
    assert np.array_equal(old, explicit_none)
