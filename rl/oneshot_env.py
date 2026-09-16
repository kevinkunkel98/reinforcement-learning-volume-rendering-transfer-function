"""Gymnasium environment: output a transfer function in one shot to follow a
visibility instruction.

Each episode picks a volume and an instruction (`goals.sample_instruction`);
unlike `vis_env.VisibilityTFEnv`'s ten-step edits, here the single action *is*
the new value of each of the 12 controllable parameter groups
(`rl.baselines.CONTROLLABLE`: height, width and colour per peak -- centres
stay fixed), not a delta, and the episode ends immediately. The reward is the
attainment of the resulting transfer function (`goals.attainment`), clipped
to `[-1, 1]`, minus a penalty when the result shows effectively nothing.

See docs/superpowers/plans/2026-09-16-rl-v2-plan6-oneshot.md for why: the
ten-step formulation asks the policy to learn a blind search procedure, which
did not work, while amortising a single forward pass over the mapping from
(instruction, start state) to transfer function does.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import goals
import visibility
from rl.baselines import CONTROLLABLE

OBSERVATION_SIZE = 53   # 16 goal + 16 histogram + 4 log10 visibility + 4 brightness + 12 start parameters + 1 coverage
ACTION_SIZE = 12        # the new value of each controllable parameter group, in [-1, 1]
USELESS_PENALTY = 1.0
REWARD_CLIP = 1.0

# Reset noise (normalized parameter units, clipped to [-1, 1] after adding),
# uniform per controllable group -- matches VisibilityTFEnv's reset noise.
START_NOISE = 0.3

# Generous finite bound for the observation Box: every packed quantity (goal
# components, log-vis/brightness within a handful of units, start params and
# coverage in/near [-1, 1]) stays well inside it.
OBSERVATION_BOUND = 10.0

# Below this, distance(goal, start, start) is ~0 (the instruction asks for
# effectively nothing) and attainment's ratio is meaningless -- report 0
# instead of dividing by ~0.
_ATTAINMENT_FLOOR = 1e-9


class OneShotEnv(gym.Env):
    """One-step episode: the action is the new transfer function, scored by
    how much closer it gets to a sampled visibility/brightness instruction on
    a randomly chosen volume from `volume_ids`."""

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
        self._start_params = None
        self._params = None
        self._instruction = None
        self._start_agg = None
        self._start_distance = None

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
        for group in CONTROLLABLE:
            noise = rng.uniform(-START_NOISE, START_NOISE)
            for index in group:
                params[index] = params[index] + noise
        return np.clip(params, -1.0, 1.0)

    def _controllable_values(self, params: np.ndarray) -> list:
        return [float(np.mean([params[i] for i in group])) for group in CONTROLLABLE]

    def _build_observation(self) -> np.ndarray:
        log_vis = [math.log10(self._start_agg["vis"][c] + goals.EPSILON) for c in goals.GOAL_CLASSES]
        bright = [self._start_agg["bright"][c] for c in goals.GOAL_CLASSES]
        controllable = self._controllable_values(self._start_params)

        values = (list(self._instruction["goal"]) + list(self._model.histogram) + log_vis + bright
                  + controllable + [self._start_agg["coverage"]])
        return np.asarray(values, dtype=np.float32)

    def _attainment(self, agg: dict) -> float:
        if self._start_distance <= _ATTAINMENT_FLOOR:
            return 0.0
        return goals.attainment(self._instruction["goal"], self._start_agg, agg)

    def _info(self, attainment: float, useless: bool) -> dict:
        return {"attainment": attainment, "kind": self._instruction["kind"],
                "volume": self._volume, "text": self._instruction["text"], "useless": useless}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        volume = str(rng.choice(self.volume_ids))
        model = self._get_model(volume)
        start_params = self._sample_start_params(rng)
        raw_features = model.features(start_params)
        start_agg = goals.aggregate(raw_features)
        instruction = goals.sample_instruction(volume, model, start_agg, rng)
        start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

        self._volume = volume
        self._model = model
        self._start_params = start_params
        self._params = start_params
        self._instruction = instruction
        self._start_agg = start_agg
        self._start_distance = start_distance

        obs = self._build_observation()
        info = self._info(self._attainment(start_agg), goals.is_useless(raw_features))
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        params = self._start_params.copy()
        for group, value in zip(CONTROLLABLE, action):
            for index in group:
                params[index] = float(value)
        self._params = params

        raw_features = self._model.features(params)
        final_agg = goals.aggregate(raw_features)
        useless = goals.is_useless(raw_features)
        attainment = self._attainment(final_agg)
        reward = float(np.clip(attainment, -REWARD_CLIP, REWARD_CLIP)) - (USELESS_PENALTY if useless else 0.0)

        obs = self._build_observation()
        info = self._info(attainment, useless)
        return obs, float(reward), True, False, info
