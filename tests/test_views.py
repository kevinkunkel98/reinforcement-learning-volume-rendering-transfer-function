import numpy as np

import views


def test_six_directions_are_unit_and_distinct():
    directions = views.view_directions()
    assert len(directions) == views.N_VIEWS == 6
    for d in directions:
        assert d.shape == (3,)
        assert np.isclose(np.linalg.norm(d), 1.0)
        assert np.isclose(d[2], 0.0)                     # rotations about the superior axis
    for i in range(6):
        for j in range(i + 1, 6):
            assert not np.allclose(directions[i], directions[j])


def test_first_direction_looks_from_anterior_towards_posterior():
    assert np.allclose(views.view_directions()[0], [0.0, -1.0, 0.0])


def test_directions_are_evenly_spaced_by_sixty_degrees():
    directions = views.view_directions()
    for i in range(6):
        angle = np.degrees(np.arccos(np.clip(directions[i] @ directions[(i + 1) % 6], -1, 1)))
        assert np.isclose(angle, 60.0)


def test_cameras_frame_the_volume_from_each_direction():
    extent = (200.0, 100.0, 300.0)                        # physical size in mm
    cameras = views.cameras_for_extent(extent)
    assert len(cameras) == 6
    center = np.array(extent) / 2.0
    for camera, direction in zip(cameras, views.view_directions()):
        assert np.allclose(camera["focal_point"], center)
        assert np.allclose(camera["view_up"], [0.0, 0.0, 1.0])
        offset = center - np.array(camera["position"])
        assert np.allclose(offset / np.linalg.norm(offset), direction)
        assert np.linalg.norm(offset) > max(extent)       # outside the volume
        assert camera["parallel_scale"] >= max(extent) / 2.0
