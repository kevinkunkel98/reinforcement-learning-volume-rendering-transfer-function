"""Gymnasium environment for training a continuous-control camera-viewpoint
agent against a centroid-alignment reward -- no VTK rendering, computed
directly on the loaded volume's HU array."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from transfer import TISSUE_BANDS

TISSUES = ["air", "fat", "soft", "spongy", "bone"]
MAX_DELTA_DEGREES = 30.0
MAX_STEPS = 20
ELEVATION_RANGE = (-85.0, 85.0)
OBS_SIZE = 2 + 1 + len(TISSUES) + 1  # sin/cos azimuth, elevation, tissue one-hot, alignment

# A tissue band's centroid is never *exactly* at the volume center -- grid
# discretization and additive noise always nudge it by a small amount, even
# for a genuinely symmetric band. So "no meaningful direction" can't be a
# check against literal zero; it has to be a magnitude threshold relative to
# the volume's size. Empirically (see tests/test_camera_env.py), on the
# synthetic phantom the truly symmetric outer layers (air, fat) sit at
# ~0.6-1.3% of the volume's diagonal extent, while tissues tied to the
# off-center bone/spongy structure (soft, spongy, bone) sit at ~5-5.2%,
# consistently across volume sizes -- a comfortable order-of-magnitude gap.
MIN_RELATIVE_CENTROID_OFFSET = 0.02


def _view_direction(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])


def _tissue_centroid_direction(volume, spacing, tissue):
    """Unit vector from the volume's physical center to the tissue's voxel
    centroid, or None if no voxel falls in that tissue's HU band, or if the
    centroid is within MIN_RELATIVE_CENTROID_OFFSET of the volume center
    (no meaningful direction -- see the constant's docstring above)."""
    lo, hi = TISSUE_BANDS[tissue]
    mask = (volume >= lo) & (volume < hi)
    if not mask.any():
        return None
    idx = np.nonzero(mask)
    sx, sy, sz = spacing
    centroid = np.array([idx[0].mean() * sx, idx[1].mean() * sy, idx[2].mean() * sz])
    extent = np.array(volume.shape) * np.array(spacing)
    center = extent / 2.0
    offset = centroid - center
    norm = np.linalg.norm(offset)
    if norm < MIN_RELATIVE_CENTROID_OFFSET * np.linalg.norm(extent):
        return None
    return offset / norm


class CameraViewpointEnv(gym.Env):
    """One episode = one target tissue; the action moves the camera's
    azimuth/elevation to maximize alignment with that tissue's centroid
    direction from the volume center."""

    metadata = {"render_modes": []}

    def __init__(self, volume: np.ndarray, spacing: tuple, seed: int | None = None):
        super().__init__()
        self.volume = volume
        self.spacing = spacing
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.azimuth = 0.0
        self.elevation = 0.0
        self.target_tissue = None
        self.target_direction = None
        self.step_count = 0

    def _resolve_target(self):
        tissues = list(TISSUES)
        self._rng.shuffle(tissues)
        for tissue in tissues:
            direction = _tissue_centroid_direction(self.volume, self.spacing, tissue)
            if direction is not None:
                return tissue, direction
        return "bone", _tissue_centroid_direction(self.volume, self.spacing, "bone")

    def _alignment(self) -> float:
        cam_dir = _view_direction(self.azimuth, self.elevation)
        return float(np.dot(cam_dir, self.target_direction))

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(len(TISSUES), dtype=np.float32)
        onehot[TISSUES.index(self.target_tissue)] = 1.0
        az_rad = np.radians(self.azimuth)
        return np.concatenate([
            np.array([np.sin(az_rad), np.cos(az_rad), self.elevation / 85.0], dtype=np.float32),
            onehot,
            np.array([self._alignment()], dtype=np.float32),
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.azimuth = float(self._rng.uniform(0.0, 360.0))
        self.elevation = float(self._rng.uniform(*ELEVATION_RANGE))
        self.target_tissue, self.target_direction = self._resolve_target()
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        before = self._alignment()
        d_az = float(np.clip(action[0], -1.0, 1.0)) * MAX_DELTA_DEGREES
        d_el = float(np.clip(action[1], -1.0, 1.0)) * MAX_DELTA_DEGREES
        self.azimuth = (self.azimuth + d_az) % 360.0
        self.elevation = float(np.clip(self.elevation + d_el, *ELEVATION_RANGE))
        after = self._alignment()
        reward = after - before
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        terminated = False
        info = {"alignment_after": after}
        return self._obs(), reward, terminated, truncated, info
