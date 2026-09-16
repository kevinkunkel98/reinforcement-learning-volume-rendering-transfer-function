"""Gymnasium environment: edit a transfer function to follow a visibility
instruction.

Each episode picks a volume and an instruction (`goals.sample_instruction`);
the agent edits the transfer function's 12 controllable values (height,
width, brightness per peak -- centres and hue stay fixed) over `MAX_STEPS`
steps. The reward is the drop in `goals.distance` to the instruction's goal,
minus a penalty when the resulting state shows effectively nothing.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import goals
import visibility
from rl.baselines import CONTROLLABLE
from transfer import N_PEAKS, PARAMS_PER_PEAK

OBSERVATION_SIZE = 62      # 12 params + 16 goal + 4 log vis + 4 bright + 4 c + 4 b + 1 coverage + 16 histogram + 1 progress
ACTION_SIZE = 12
MAX_STEPS = 10
STEP_SCALE = 0.1           # per action unit, in normalized parameter units
USELESS_PENALTY = 1.0

# Generous finite bound for the observation Box: every packed quantity (params
# in [-1, 1], goal/log-vis/bright/progress within a handful of units,
# coverage/histogram in [0, 1], step fraction in [0, 1]) stays well inside it.
OBSERVATION_BOUND = 10.0

# Reset noise (normalized parameter units, clipped to [-1, 1] after adding).
START_HEIGHT_NOISE = 0.3
START_WIDTH_NOISE = 0.2
START_BRIGHTNESS_NOISE = 0.2

# Below this, distance(goal, start, start) is ~0 (the state already meets the
# goal) and attainment's ratio is meaningless -- report 0 instead of dividing
# by ~0.
_ATTAINMENT_FLOOR = 1e-9


class VisibilityTFEnv(gym.Env):
    """One goal-conditioned episode: `MAX_STEPS` transfer-function edits
    toward a sampled visibility/brightness instruction on a randomly chosen
    volume from `volume_ids`."""

    metadata = {"render_modes": []}

    def __init__(self, volume_ids, model_for_volume=visibility.for_volume):
        super().__init__()
        if not volume_ids:
            raise ValueError("volume_ids must not be empty")
        self.volume_ids = list(volume_ids)
        self._model_for_volume = model_for_volume
        self._model_cache = {}

        self.observation_space = spaces.Box(
            low=-OBSERVATION_BOUND, high=OBSERVATION_BOUND, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)

        self._volume = None
        self._model = None
        self._params = None
        self._instruction = None
        self._start_agg = None
        self._start_distance = None
        self._prev_distance = None
        self._step_count = 0

    def _get_model(self, volume: str):
        # visibility.for_volume() already caches to disk; this in-memory cache
        # additionally avoids re-reading/re-decompressing that cache file on
        # every one of the many resets a training run does.
        model = self._model_cache.get(volume)
        if model is None:
            model = self._model_for_volume(volume)
            self._model_cache[volume] = model
        return model

    def _sample_start_params(self, rng) -> np.ndarray:
        params = goals.starting_params()
        for index in goals.PEAK_INDEX.values():
            base = index * PARAMS_PER_PEAK
            params[base + 1] += rng.uniform(-START_WIDTH_NOISE, START_WIDTH_NOISE)
            params[base + 2] += rng.uniform(-START_HEIGHT_NOISE, START_HEIGHT_NOISE)
            brightness_noise = rng.uniform(-START_BRIGHTNESS_NOISE, START_BRIGHTNESS_NOISE)
            params[base + 3] += brightness_noise
            params[base + 4] += brightness_noise
            params[base + 5] += brightness_noise
        return np.clip(params, -1.0, 1.0)

    def _build_observation(self, params: np.ndarray, agg: dict) -> np.ndarray:
        controllable = []
        for peak in range(N_PEAKS):
            base = peak * PARAMS_PER_PEAK
            controllable.append(float(params[base + 1]))                       # width
            controllable.append(float(params[base + 2]))                       # height
            controllable.append(float(np.mean(params[base + 3:base + 6])))     # brightness

        log_vis = [math.log10(agg["vis"][c] + goals.EPSILON) for c in goals.GOAL_CLASSES]
        bright = [agg["bright"][c] for c in goals.GOAL_CLASSES]
        c, b = goals.progress(self._start_agg, agg)
        c_vals = [c[goal_class] for goal_class in goals.GOAL_CLASSES]
        b_vals = [b[goal_class] for goal_class in goals.GOAL_CLASSES]
        step_fraction = self._step_count / MAX_STEPS

        values = (controllable + list(self._instruction["goal"]) + log_vis + bright
                  + c_vals + b_vals + [agg["coverage"]] + list(self._model.histogram) + [step_fraction])
        return np.asarray(values, dtype=np.float32)

    def _attainment(self, agg: dict) -> float:
        if self._start_distance <= _ATTAINMENT_FLOOR:
            return 0.0
        return goals.attainment(self._instruction["goal"], self._start_agg, agg)

    def _info(self, distance: float, agg: dict, useless: bool) -> dict:
        return {"attainment": self._attainment(agg), "distance": distance,
                "kind": self._instruction["kind"], "volume": self._volume,
                "text": self._instruction["text"], "useless": useless}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        volume = str(rng.choice(self.volume_ids))
        model = self._get_model(volume)
        params = self._sample_start_params(rng)
        raw_features = model.features(params)
        start_agg = goals.aggregate(raw_features)
        instruction = goals.sample_instruction(volume, model, start_agg, rng)
        start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

        self._volume = volume
        self._model = model
        self._params = params
        self._instruction = instruction
        self._start_agg = start_agg
        self._start_distance = start_distance
        self._prev_distance = start_distance
        self._step_count = 0

        obs = self._build_observation(params, start_agg)
        info = self._info(start_distance, start_agg, goals.is_useless(raw_features))
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        params = self._params.copy()
        for group, value in zip(CONTROLLABLE, action):
            delta = float(value) * STEP_SCALE
            for index in group:
                params[index] = float(np.clip(params[index] + delta, -1.0, 1.0))
        self._params = params

        raw_features = self._model.features(params)
        agg = goals.aggregate(raw_features)
        distance = goals.distance(self._instruction["goal"], self._start_agg, agg)
        useless = goals.is_useless(raw_features)
        reward = self._prev_distance - distance - (USELESS_PENALTY if useless else 0.0)
        self._prev_distance = distance

        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS
        terminated = False

        obs = self._build_observation(params, agg)
        info = self._info(distance, agg, useless)
        return obs, float(reward), terminated, truncated, info
