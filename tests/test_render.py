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
