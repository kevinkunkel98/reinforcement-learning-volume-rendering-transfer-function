"""Gymnasium environment: output a transfer function in one shot to follow a
visibility instruction.

Each episode picks a volume and an instruction (`goals.sample_instruction`);
unlike `vis_env.VisibilityTFEnv`'s ten-step edits, here the single action *is*
the new value of each controllable parameter group
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
import json
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import goals
import visibility
from anatomy import LAYOUT_VERSION
from rl.baselines import CONTROLLABLE, apply_controllable

N_GOAL_CLASSES = len(goals.GOAL_CLASSES)
OBSERVATION_SIZE = 4 * N_GOAL_CLASSES + 16 + 3 * N_GOAL_CLASSES + len(CONTROLLABLE) + 1
ACTION_SIZE = len(CONTROLLABLE)
POLICY_VERSION = "oneshot-v6"
V7_POLICY_VERSION = "oneshot-v7"
USELESS_PENALTY = 1.0
REWARD_CLIP = 1.0

# Reset noise (normalized parameter units, clipped to [-1, 1] after adding),
# uniform per controllable group -- matches VisibilityTFEnv's reset noise.
START_NOISE = 0.3

# Hindsight target noise, in the same normalized parameter units as
# START_NOISE: the target transfer function is the start's controllable values
# plus U(-HINDSIGHT_NOISE, HINDSIGHT_NOISE) per group, not a uniform draw over
# the whole action space. A uniform draw lands far from the start and asks for
# visibility changes of a factor of ~40 (median 1.62 log10, up to the full 3.0
# floor-to-visible range), while a real instruction asks for 0.15 to 1.0
# (goals.VISIBILITY_STRENGTH and goals.HIDE_STRENGTH) -- training on the former
# and evaluating on the latter is a goal-distribution shift. It also moves
# every peak at once, including those of the classes the goal does not mention,
# which goals.distance then charges as keep-term drift, so the target itself
# could not score near 1.
#
# 0.25 is calibrated, not round: swept over 200 synthetic episodes each, the
# median largest |delta| runs 0.19 (s=0.1), 0.36 (0.2), 0.44 (0.25), 0.53
# (0.3), 0.97 (0.5). 0.25 puts the median inside the band real instructions
# ask for and at the plateau of the share that lands in [0.15, 1.0] (62%,
# against 63% at 0.3 and 52% at 0.15), while giving the cleanest oracle --
# median attainment 0.985 for the action that produced the target, against
# 0.729 under the uniform draw this replaced.
HINDSIGHT_NOISE = 0.25

# Generous finite bound for the observation Box: every packed quantity (goal
# components, log-vis/brightness within a handful of units, start params and
# coverage in/near [-1, 1]) stays well inside it.
OBSERVATION_BOUND = 10.0

# Below this, distance(goal, start, start) is ~0 (the instruction asks for
# effectively nothing) and attainment's ratio is meaningless -- report 0
# instead of dividing by ~0.
_ATTAINMENT_FLOOR = 1e-9


def resolve_policy_metadata(metadata=None) -> dict:
    values = {"policy_version": POLICY_VERSION, "action_mode": "absolute",
              "reward_mode": "attainment"}
    if metadata:
        values.update({key: metadata[key] for key in values if key in metadata})
    if values["policy_version"] == POLICY_VERSION and values["action_mode"] != "absolute":
        raise ValueError("v6 only supports absolute action mode")
    if values["action_mode"] not in ("absolute", "residual"):
        raise ValueError("action_mode must be absolute or residual")
    if values["reward_mode"] not in ("attainment", "target"):
        raise ValueError("reward_mode must be attainment or target")
    return values


def load_policy_metadata(checkpoint_path: str) -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "metadata.json")
    if not os.path.exists(path):
        return resolve_policy_metadata()
    with open(path) as stream:
        return resolve_policy_metadata(json.load(stream))


def observation_metadata(policy_version: str = POLICY_VERSION, action_mode: str = "absolute",
                         reward_mode: str = "attainment") -> dict:
    """Return the serialized contract shared by training and inference."""
    resolved = resolve_policy_metadata({"policy_version": policy_version,
                                        "action_mode": action_mode,
                                        "reward_mode": reward_mode})
    return {
        "policy_version": resolved["policy_version"],
        "anatomy_layout": LAYOUT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "goal_classes": list(goals.GOAL_CLASSES),
        "controllable_groups": len(CONTROLLABLE),
        **({"action_mode": action_mode, "reward_mode": reward_mode}
           if policy_version == V7_POLICY_VERSION else {}),
    }


def build_observation(goal, histogram, start_agg: dict, solo_max_log, controllable) -> np.ndarray:
    """One-shot observation: goal (4n) + histogram (16) + log10 visibility (n)
    + brightness (n) + log10 solo_max ceiling (n) + controllable start
    parameters + coverage (1), all at
    the start state. The single implementation of this layout -- used by
    `OneShotEnv._build_observation` during training/rollout and by
    `rl.candidates` to query a policy standalone for preference collection,
    so the two can never silently drift apart and feed the policy a
    different observation than it was trained on."""
    log_vis = [math.log10(start_agg["vis"][c] + goals.EPSILON) for c in goals.GOAL_CLASSES]
    bright = [start_agg["bright"][c] for c in goals.GOAL_CLASSES]
    values = (list(goal) + list(histogram) + log_vis + bright
              + list(solo_max_log) + list(controllable) + [start_agg["coverage"]])
    return np.asarray(values, dtype=np.float32)


class OneShotEnv(gym.Env):
    """One-step episode: the action is the new transfer function, scored by
    how much closer it gets to a sampled visibility/brightness instruction on
    a randomly chosen volume from `volume_ids`.

    Observation layout derives from `len(goals.GOAL_CLASSES)` and
    `len(CONTROLLABLE)`.
    """

    metadata = {"render_modes": []}

    HINDSIGHT_MAX_MENTIONS = 2

    # A hindsight target whose largest class delta is below this asks for
    # effectively nothing: the goal is trivially already met, `_start_distance`
    # sits at the attainment floor, and the episode carries no gradient. Resample
    # instead. 0.05 log10 units is a factor of ~1.12 -- a third of the smallest
    # spoken strength ("slightly", 0.15), so nothing an instruction could ask
    # for is excluded.
    HINDSIGHT_MIN_DELTA = 0.05
    HINDSIGHT_MAX_TRIES = 20

    def __init__(self, volume_ids, model_for_volume=visibility.for_volume,
                 hindsight_ratio: float = 0.0, policy_version: str = POLICY_VERSION,
                 action_mode: str = "absolute", reward_mode: str = "attainment",
                 balance_classes: bool = False):
        super().__init__()
        if not volume_ids:
            raise ValueError("volume_ids must not be empty")
        self.volume_ids = list(volume_ids)
        self._model_for_volume = model_for_volume
        if policy_version not in (POLICY_VERSION, V7_POLICY_VERSION):
            raise ValueError("unsupported one-shot policy version")
        if policy_version == V7_POLICY_VERSION and action_mode != "residual":
            raise ValueError("oneshot-v7 requires residual action mode")
        if action_mode not in ("absolute", "residual"):
            raise ValueError("action_mode must be absolute or residual")
        if reward_mode not in ("attainment", "target"):
            raise ValueError("reward_mode must be attainment or target")
        if not 0.0 <= hindsight_ratio <= 1.0:
            raise ValueError("hindsight_ratio must be between 0 and 1")
        self.policy_version = policy_version
        self.action_mode = action_mode
        self.reward_mode = reward_mode
        self.balance_classes = balance_classes
        self._class_counts = {}
        # A sampled instruction can be unanswerable on the volume it lands on
        # ("a bit more lungs" on a pelvis scan, where the lung ceiling is a
        # fraction of a percent of the image): the episode teaches nothing and
        # pollutes evaluation. This share of episodes inverts the sampling
        # instead -- draw a *reachable* target transfer function, measure what
        # it achieved, and make that the goal -- so every such episode has a
        # demonstrated solution, the action that produced the target.
        self.hindsight_ratio = float(hindsight_ratio)
        self._model_cache = {}
        self._solo_max_log_cache = {}

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
        # Set by reset() in both branches, so a hindsight episode can never
        # leave its oracle action behind for a later instruction episode.
        self._hindsight_action = None

    def contract_metadata(self) -> dict:
        return observation_metadata(self.policy_version, self.action_mode, self.reward_mode)

    def _get_model(self, volume: str):
        # visibility.for_volume() already caches to disk; this in-memory cache
        # additionally avoids re-reading/re-decompressing that cache file on
        # every one of the many resets a training run does.
        model = self._model_cache.get(volume)
        if model is None:
            model = self._model_for_volume(volume)
            self._model_cache[volume] = model
        return model

    def _solo_max_log(self, volume: str, model) -> list:
        """log10(ceiling + EPSILON) per `goals.GOAL_CLASSES`, where ceiling is
        the most of that goal class a single-peak transfer function can show
        (`model.solo_max`, summed over `goals.MEASURED_FOR_GOAL` the way
        `goals._absolute_target_delta` does, since a goal class like "soft"
        has no single underlying `model.solo_max` name of its own).
        `model.solo_max` already caches per class on the model, but this
        additionally avoids redoing the summation/log10 work on every reset
        of an already-seen volume.
        """
        layers = getattr(self, "_active_layers", None)
        cache_key = volume if layers is None else (volume, tuple(
            (name, settings.get("opacity", 1.0))
            for name, settings in layers.items()))
        cached = self._solo_max_log_cache.get(cache_key)
        if cached is None:
            cached = [math.log10(goals.class_ceiling(model, c, layers) + goals.EPSILON)
                      for c in goals.GOAL_CLASSES]
            self._solo_max_log_cache[cache_key] = cached
        return cached

    def _sample_start_params(self, rng) -> np.ndarray:
        params = goals.starting_params()
        for group in CONTROLLABLE:
            noise = rng.uniform(-START_NOISE, START_NOISE)
            for index in group:
                params[index] = params[index] + noise
        return np.clip(params, -1.0, 1.0)

    def _controllable_values(self, params: np.ndarray) -> list:
        return [float(np.mean([params[i] for i in group])) for group in CONTROLLABLE]

    # `self._instruction`/`self._start_agg`/etc. are set to None in __init__
    # and only ever read here, after reset() has always already filled them
    # in -- the standard Gymnasium convention (reset() before step()), not a
    # real possibility of None reaching these lines.
    def _build_observation(self) -> np.ndarray:
        solo_max_log = self._solo_max_log(self._volume, self._model)
        controllable = self._controllable_values(self._start_params)
        return build_observation(self._instruction["goal"], self._model.histogram, self._start_agg,  # pyright: ignore[reportOptionalSubscript]
                                  solo_max_log, controllable)

    def _features(self, params):
        return goals.features(self._model, params, getattr(self, "_active_layers", None))

    def _attainment(self, agg: dict) -> float:
        if self._start_distance <= _ATTAINMENT_FLOOR:
            return 0.0
        return goals.attainment(self._instruction["goal"], self._start_agg, agg)  # pyright: ignore[reportOptionalSubscript]

    def _info(self, attainment: float, useless: bool) -> dict:
        info = {"attainment": attainment, "kind": self._instruction["kind"],  # pyright: ignore[reportOptionalSubscript]
                "volume": self._volume, "text": self._instruction["text"], "useless": useless,  # pyright: ignore[reportOptionalSubscript]
                "goal_source": "hindsight" if self._hindsight_action is not None else "instruction"}
        if self.policy_version == V7_POLICY_VERSION:
            info["policy_version"] = self.policy_version
        return info

    def hindsight_action(self):
        """The action that produced this episode's target, or None when the
        episode came from an instruction. Test and distillation hook."""
        return None if self._hindsight_action is None else self._hindsight_action.copy()

    def _sample_hindsight_goal(self, rng, model, start_params, start_agg):
        """(instruction-shaped dict, action) from a reachable target."""
        start_controllable = np.asarray(self._controllable_values(start_params))
        for _ in range(self.HINDSIGHT_MAX_TRIES):
            noise = rng.uniform(-HINDSIGHT_NOISE, HINDSIGHT_NOISE, size=len(CONTROLLABLE))
            action = np.clip(start_controllable + noise, -1.0, 1.0)
            target_params = apply_controllable(start_params, action)
            target_agg = goals.aggregate(self._features(target_params), self._active_layers)

            deltas = {}
            for goal_class in goals.GOAL_CLASSES:
                start_vis = start_agg["vis"][goal_class]
                target_vis = target_agg["vis"][goal_class]
                # Same convention as goals.progress, so the goal the policy is
                # handed and the change the reward measures are one quantity.
                deltas[goal_class] = math.log10(target_vis + goals.EPSILON) \
                    - math.log10(start_vis + goals.EPSILON)

            ranked = sorted(goals.GOAL_CLASSES, key=lambda c: -abs(deltas[c]))
            if abs(deltas[ranked[0]]) >= self.HINDSIGHT_MIN_DELTA:
                break
        # A real instruction names one or two classes, never all classes; a
        # hindsight goal that mentioned every class would shift the observation
        # distribution the policy sees away from the one it is evaluated on.
        count = int(rng.integers(1, self.HINDSIGHT_MAX_MENTIONS + 1))
        # HINDSIGHT_MAX_TRIES (20) is a fixed positive constant, so the loop
        # above always runs at least once and these are always bound; Pyright
        # can't fold that constant across the `range()` call to see it.
        targets = {c: {"vis": deltas[c]} for c in ranked[:count]}  # pyright: ignore[reportPossiblyUnboundVariable]

        instruction = {
            "kind": "hindsight",
            "text": None,
            "targets": targets,
            "goal": goals.goal_vector(targets),
        }
        return instruction, action  # pyright: ignore[reportPossiblyUnboundVariable]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        volume = str(rng.choice(self.volume_ids))
        model = self._get_model(volume)
        self._model = model
        start_params = self._sample_start_params(rng)
        self._active_layers = getattr(self, "_active_layers", None)
        raw_features = goals.features(model, start_params, self._active_layers)
        start_agg = goals.aggregate(raw_features, self._active_layers)
        if rng.random() < self.hindsight_ratio:
            instruction, hindsight_action = self._sample_hindsight_goal(rng, model, start_params, start_agg)
        else:
            instruction, hindsight_action = (goals.sample_instruction(
                volume, model, start_agg, rng, self._active_layers,
                self._class_counts, self.balance_classes), None)
            for goal_class in instruction["targets"]:
                self._class_counts[goal_class] = self._class_counts.get(goal_class, 0) + 1
        start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

        self._volume = volume
        self._start_params = start_params
        self._params = start_params
        self._instruction = instruction
        self._start_agg = start_agg
        self._start_distance = start_distance
        self._hindsight_action = hindsight_action

        obs = self._build_observation()
        info = self._info(self._attainment(start_agg), goals.is_useless(raw_features, self._active_layers))
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.action_mode == "residual":
            action = np.asarray(self._controllable_values(self._start_params)) + action
        params = apply_controllable(self._start_params, np.clip(action, -1.0, 1.0))
        self._params = params

        raw_features = self._features(params)
        active_layers = getattr(self, "_active_layers", None)
        final_agg = goals.aggregate(raw_features, active_layers)
        useless = (goals.is_useless(raw_features, active_layers)
                   if active_layers is not None else goals.is_useless(raw_features))
        attainment = self._attainment(final_agg)
        if self.reward_mode == "target":
            start_distance = goals.distance(self._instruction["goal"], self._start_agg, self._start_agg)
            final_distance = goals.distance(self._instruction["goal"], self._start_agg, final_agg)
            reward = float(np.clip(start_distance - final_distance, -REWARD_CLIP, REWARD_CLIP))
            mentioned = set(self._instruction["targets"])
            drift = sum(abs(value) for key, value in goals.progress(self._start_agg, final_agg)[0].items()
                        if key not in mentioned)
            reward -= float(drift)
            if useless:
                reward -= USELESS_PENALTY
            info = self._info(attainment, useless)
            info.update({"target_progress": start_distance - final_distance, "drift": drift})
        else:
            reward = float(np.clip(attainment, -REWARD_CLIP, REWARD_CLIP)) - (USELESS_PENALTY if useless else 0.0)
            info = self._info(attainment, useless)

        obs = self._build_observation()
        return obs, float(reward), True, False, info
