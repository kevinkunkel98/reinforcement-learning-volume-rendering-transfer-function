import numpy as np
from phantom import build_phantom
from transfer import default_params
import render as render_module
import views
from render import render, grab, features


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
