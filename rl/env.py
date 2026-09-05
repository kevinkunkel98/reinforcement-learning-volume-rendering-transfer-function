"""Gymnasium environment for training a continuous-control TF agent directly
against the opacity_mass metric -- no rendering involved, matching how
evaluate.objective() already scores hill-climbing steps on raw params."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from commands import _find_or_create_peak
from search import propose_step
from transfer import N_PEAKS, PARAMS_PER_PEAK, default_params, mass_fraction

TISSUES = ["air", "fat", "soft", "spongy", "bone"]
DIRECTIONS = ["increase", "decrease"]
MAX_DELTA = 0.2
MAX_STEPS = 20
HEIGHT_NOISE = 0.3
OBS_SIZE = N_PEAKS * PARAMS_PER_PEAK + len(TISSUES) + 2


class TFEnv(gym.Env):
    """One episode = one (target_tissue, direction) goal; the action moves
    the resolved target peak's height each step."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.params = None
        self.target_tissue = None
        self.direction = None
        self.peak_idx = None
        self.step_count = 0

    def _sample_start_params(self) -> np.ndarray:
        params = default_params().copy()
        for i in range(N_PEAKS):
            base = i * PARAMS_PER_PEAK
            noise = self._rng.uniform(-HEIGHT_NOISE, HEIGHT_NOISE)
            params[base + 2] = float(np.clip(params[base + 2] + noise, -1.0, 1.0))
        return params

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(len(TISSUES), dtype=np.float32)
        onehot[TISSUES.index(self.target_tissue)] = 1.0
        direction_flag = 1.0 if self.direction == "increase" else -1.0
        frac = mass_fraction(self.params, self.target_tissue)
        return np.concatenate([
            self.params.astype(np.float32),
            onehot,
            np.array([direction_flag, frac], dtype=np.float32),
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        params = self._sample_start_params()
        self.target_tissue = TISSUES[self._rng.integers(0, len(TISSUES))]
        self.direction = DIRECTIONS[self._rng.integers(0, len(DIRECTIONS))]
        self.params, self.peak_idx = _find_or_create_peak(params, self.target_tissue)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        direction_sign = 1.0 if self.direction == "increase" else -1.0
        delta = float(np.clip(action[0], -1.0, 1.0)) * MAX_DELTA
        before_frac = mass_fraction(self.params, self.target_tissue)
        self.params = propose_step(self.params, self.peak_idx, sign=1.0, step=delta)
        after_frac = mass_fraction(self.params, self.target_tissue)
        reward = direction_sign * (after_frac - before_frac)
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        terminated = False
        info = {"mass_fraction_after": after_frac}
        return self._obs(), reward, terminated, truncated, info
