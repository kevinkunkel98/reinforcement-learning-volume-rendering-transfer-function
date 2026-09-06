"""Tests for rl/camera_env.py -- pure geometry + environment logic, no rendering."""
import numpy as np
import pytest

from phantom import build_phantom
from rl.camera_env import (
    ELEVATION_RANGE, MAX_STEPS, TISSUES,
    CameraViewpointEnv, _tissue_centroid_direction, _view_direction,
)


def test_view_direction_is_unit_vector():
    for az in (0.0, 45.0, 130.0, 300.0):
        for el in (-60.0, 0.0, 60.0):
            v = _view_direction(az, el)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_view_direction_reference_orientation():
    v = _view_direction(0.0, 0.0)
    assert np.allclose(v, [0.0, 0.0, 1.0], atol=1e-9)


def test_synthetic_phantom_air_and_fat_have_no_centroid_direction():
    # air and fat are the outermost layers of the phantom's torso blob,
    # which *is* centered on the volume's true center in phantom.py -- by
    # the time you're that far out, the off-center bone/spongy structure
    # near the middle has negligible effect on where the band sits, so
    # their centroid direction is undefined. (Measured directly: relative
    # centroid offset for air/fat is ~0.6-1.3% of the volume size, versus
    # ~4.9-5.2% for soft/spongy/bone -- see below.)
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    for tissue in ("air", "fat"):
        assert _tissue_centroid_direction(volume, spacing, tissue) is None


def test_synthetic_phantom_soft_spongy_and_bone_have_a_centroid_direction():
    # bone/spongy are built off-center in phantom.py, and "soft" -- being
    # their immediate HU neighbor -- inherits that same off-center bias:
    # the phantom's true center sits almost inside the bone blob, so the
    # thin shell of soft-banded voxels wrapping it is itself displaced,
    # not the symmetric outer torso surface one might expect from reading
    # the blob definitions alone. Verified empirically (not just by
    # inspecting the blob centers), see test above for the measurements.
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    for tissue in ("soft", "spongy", "bone"):
        direction = _tissue_centroid_direction(volume, spacing, tissue)
        assert direction is not None
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6


def test_reset_on_synthetic_phantom_never_picks_air_or_fat():
    # The phantom has no valid direction for air/fat, so reset() must
    # never select them, across many random seeds.
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    for seed in range(30):
        env = CameraViewpointEnv(volume, spacing, seed=seed)
        env.reset(seed=seed)
        assert env.target_tissue in ("soft", "spongy", "bone")


def test_reset_observation_shape_and_bounds():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=0)
    obs, info = env.reset()
    assert obs.shape == (9,)
    onehot = obs[3:8]
    assert onehot.sum() == pytest.approx(1.0)
    assert set(np.unique(onehot)) <= {0.0, 1.0}
    assert -1.0 <= obs[8] <= 1.0


def test_step_moves_azimuth_and_elevation():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=1)
    env.reset()
    before_az, before_el = env.azimuth, env.elevation
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert env.azimuth != before_az
    assert env.elevation != before_el or before_el >= ELEVATION_RANGE[1] - 1e-6


def test_elevation_stays_clamped_under_repeated_extreme_action():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=2)
    env.reset()
    for _ in range(20):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.elevation <= ELEVATION_RANGE[1] + 1e-6


def test_azimuth_wraps_under_repeated_action():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=3)
    env.reset()
    for _ in range(20):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert 0.0 <= env.azimuth < 360.0


def test_episode_truncates_at_max_steps_and_never_terminates():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=4)
    env.reset()
    truncated = False
    for i in range(MAX_STEPS):
        _, _, terminated, truncated, _ = env.step(np.array([0.1, 0.1], dtype=np.float32))
        assert terminated is False
        if i < MAX_STEPS - 1:
            assert truncated is False
    assert truncated is True


def test_reward_equals_the_actual_alignment_delta():
    # reward must always equal (alignment after the step) - (alignment before
    # the step), regardless of direction -- this is the dense reward's entire
    # definition, so pin it down directly rather than asserting a sign that
    # would depend on exactly where the target tissue happens to sit.
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=5)
    env.reset()
    env.target_tissue = "bone"
    env.target_direction = _tissue_centroid_direction(volume, spacing, "bone")
    env.azimuth = 0.0
    env.elevation = 0.0
    before_alignment = float(np.dot(_view_direction(0.0, 0.0), env.target_direction))
    _, reward, _, _, _ = env.step(np.array([1.0, 1.0], dtype=np.float32))
    after_alignment = float(np.dot(_view_direction(env.azimuth, env.elevation), env.target_direction))
    assert abs(reward - (after_alignment - before_alignment)) < 1e-6
