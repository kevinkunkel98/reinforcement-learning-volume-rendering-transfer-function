"""The fixed set of views RL v2 looks at a volume from.

Volumes are in canonical RAS axis order (axis 0 -> patient right, 1 ->
anterior, 2 -> superior; see datasets.load_dataset(..., canonical=True)).
View 0 looks from in front of the patient towards the back; views 1-5 are
that direction rotated around the patient's head-to-foot axis in 60 degree
steps. The same views are used by the visibility estimate, by the rendered
views the reward model sees, and as the start camera of the collect page, so
all three describe the same thing.
"""
import numpy as np

N_VIEWS = 6
VIEW_UP = np.array([0.0, 0.0, 1.0])       # superior
CAMERA_DISTANCE_FACTOR = 2.2              # times the largest extent, so the camera clears the volume
PARALLEL_SCALE_FACTOR = 0.55              # half-height of the parallel projection, times the largest extent


def view_directions(n: int = N_VIEWS) -> list:
    """Unit vectors pointing from each camera towards the volume."""
    angles = np.deg2rad(360.0 * np.arange(n) / n)
    return [np.array([np.sin(a), -np.cos(a), 0.0]) for a in angles]


def cameras_for_extent(extent) -> list:
    """Renderer-neutral cameras for a volume of this physical size (mm)."""
    extent = np.asarray(extent, dtype=np.float64)
    center = extent / 2.0
    largest = float(extent.max())
    cameras = []
    for direction in view_directions():
        cameras.append({
            "position": (center - direction * largest * CAMERA_DISTANCE_FACTOR).tolist(),
            "focal_point": center.tolist(),
            "view_up": VIEW_UP.tolist(),
            "parallel_scale": largest * PARALLEL_SCALE_FACTOR,
        })
    return cameras


def cameras_for_volume(volume, spacing) -> list:
    """Renderer-neutral cameras for a loaded volume."""
    extent = np.asarray(volume.shape, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
    return cameras_for_extent(extent)
