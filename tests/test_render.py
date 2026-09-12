import numpy as np
from phantom import build_phantom
from transfer import default_params
import render as render_module
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
