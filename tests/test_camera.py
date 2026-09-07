"""Tests for camera.py -- pure camera-state math, no rendering involved."""
from camera import DEFAULT_CAMERA, ELEVATION_RANGE, ZOOM_RANGE, apply_camera_command, apply_camera_delta


def test_rotate_right_increases_azimuth():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "rotate", "direction": "right", "strength": "moderately"}
    after = apply_camera_command(cmd, camera)
    assert after["azimuth"] > camera["azimuth"]


def test_rotate_left_decreases_azimuth_and_wraps():
    camera = {"azimuth": 10.0, "elevation": 0.0, "zoom": 1.0}
    cmd = {"action": "rotate", "direction": "left", "strength": "strongly"}
    after = apply_camera_command(cmd, camera)
    # 10 - 60 = -50, wrapped into [0, 360)
    assert after["azimuth"] == 310.0


def test_tilt_up_increases_elevation_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "tilt", "direction": "up", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["elevation"] <= ELEVATION_RANGE[1]


def test_tilt_down_decreases_elevation_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "tilt", "direction": "down", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["elevation"] >= ELEVATION_RANGE[0]


def test_zoom_in_increases_zoom_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "zoom", "direction": "in", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["zoom"] <= ZOOM_RANGE[1]


def test_zoom_out_decreases_zoom_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "zoom", "direction": "out", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["zoom"] >= ZOOM_RANGE[0]


def test_apply_camera_command_does_not_mutate_input():
    camera = dict(DEFAULT_CAMERA)
    original = dict(camera)
    apply_camera_command({"action": "rotate", "direction": "right", "strength": "moderately"}, camera)
    assert camera == original


def test_rotate_right_wraps_above_360():
    camera = {"azimuth": 350.0, "elevation": 0.0, "zoom": 1.0}
    cmd = {"action": "rotate", "direction": "right", "strength": "strongly"}
    after = apply_camera_command(cmd, camera)
    assert after["azimuth"] == 50.0  # 350 + 60 = 410 -> wraps to 50


def test_apply_camera_delta_increases_azimuth():
    camera = dict(DEFAULT_CAMERA)
    after = apply_camera_delta(camera, d_azimuth=15.0, d_elevation=0.0, d_zoom_factor=1.0)
    assert after["azimuth"] == camera["azimuth"] + 15.0


def test_apply_camera_delta_wraps_azimuth():
    camera = {"azimuth": 350.0, "elevation": 0.0, "zoom": 1.0}
    after = apply_camera_delta(camera, d_azimuth=20.0, d_elevation=0.0, d_zoom_factor=1.0)
    assert after["azimuth"] == 10.0  # 350 + 20 = 370 -> wraps to 10


def test_apply_camera_delta_clamps_elevation():
    camera = dict(DEFAULT_CAMERA)
    after = apply_camera_delta(camera, d_azimuth=0.0, d_elevation=500.0, d_zoom_factor=1.0)
    assert after["elevation"] == ELEVATION_RANGE[1]


def test_apply_camera_delta_clamps_zoom():
    camera = dict(DEFAULT_CAMERA)
    after = apply_camera_delta(camera, d_azimuth=0.0, d_elevation=0.0, d_zoom_factor=1000.0)
    assert after["zoom"] == ZOOM_RANGE[1]


def test_apply_camera_delta_does_not_mutate_input():
    camera = dict(DEFAULT_CAMERA)
    original = dict(camera)
    apply_camera_delta(camera, d_azimuth=15.0, d_elevation=5.0, d_zoom_factor=1.1)
    assert camera == original
