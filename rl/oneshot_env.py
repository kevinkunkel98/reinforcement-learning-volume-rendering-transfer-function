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
import transfer
import visibility
from anatomy import LAYOUT_VERSION
from anatomy_layers import default_layers, normalize_layers
from rl.baselines import CONTROLLABLE, apply_controllable

N_GOAL_CLASSES = len(goals.GOAL_CLASSES)
OBSERVATION_SIZE = 4 * N_GOAL_CLASSES + 16 + 3 * N_GOAL_CLASSES + len(CONTROLLABLE) + 1
ACTION_SIZE = len(CONTROLLABLE)
POLICY_VERSION = "oneshot-v6"
V7_POLICY_VERSION = "oneshot-v7"
V8_POLICY_VERSION = "oneshot-v8"
# v9: v7's residual/target contract on the label-scoped transfer function
# (transfer.DEFAULT_TRANSFER_MODE) -- each organ's peak shades only voxels
# segmented as that organ, so no layer-opacity actions are needed.
V9_POLICY_VERSION = "oneshot-v9"
RESIDUAL_POLICY_VERSIONS = (V7_POLICY_VERSION, V8_POLICY_VERSION, V9_POLICY_VERSION)
POLICY_VERSIONS = (POLICY_VERSION,) + RESIDUAL_POLICY_VERSIONS
LAYER_NAMES = tuple(goals.GOAL_CLASSES)
V8_OBSERVATION_SIZE = OBSERVATION_SIZE + len(LAYER_NAMES)
V8_ACTION_SIZE = ACTION_SIZE + len(LAYER_NAMES)
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

# Brightness hindsight: move one visible class's colour group by up to this
# much (normalized units, so up to 0.4 in unit brightness -- the "strongly"
# end of goals.BRIGHTNESS_STRENGTH) and nothing else, so the demonstrated
# action changes that class's brightness without charging keep drift.
HINDSIGHT_BRIGHTNESS_NOISE = 0.8
HINDSIGHT_BRIGHTNESS_MIN_DELTA = 0.05
# Brightness is only measured on visible tissue (visibility.features reports
# 0 below 1e-4), so a class needs this much visibility to be demonstrable.
HINDSIGHT_BRIGHTNESS_MIN_VIS = 1e-3

# Distance already contains the keep penalty for unmentioned classes. Keep
# auxiliary drift small so target progress remains primary signal instead of
# double-penalizing ordinary occlusion side effects.
TARGET_DRIFT_WEIGHT = 0.1

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
    if values["policy_version"] not in POLICY_VERSIONS:
        raise ValueError(f"unsupported policy version: {values['policy_version']}")
    if values["policy_version"] == V9_POLICY_VERSION and metadata and \
            metadata.get("transfer_mode", transfer.DEFAULT_TRANSFER_MODE) != transfer.DEFAULT_TRANSFER_MODE:
        raise ValueError(f"oneshot-v9 checkpoint uses transfer mode {metadata['transfer_mode']!r}, "
                         f"this build evaluates {transfer.DEFAULT_TRANSFER_MODE!r}")
    if values["policy_version"] == POLICY_VERSION and values["action_mode"] != "absolute":
        raise ValueError("v6 only supports absolute action mode")
    if values["policy_version"] == POLICY_VERSION and values["reward_mode"] != "attainment":
        raise ValueError("v6 only supports attainment reward")
    if values["action_mode"] not in ("absolute", "residual"):
        raise ValueError("action_mode must be absolute or residual")
    if values["reward_mode"] not in ("attainment", "target"):
        raise ValueError("reward_mode must be attainment or target")
    if values["policy_version"] == V7_POLICY_VERSION and (
            values["action_mode"] != "residual" or values["reward_mode"] != "target"):
        raise ValueError("oneshot-v7 requires residual actions and target reward")
    if values["policy_version"] in (V8_POLICY_VERSION, V9_POLICY_VERSION) and (
            values["action_mode"] != "residual" or values["reward_mode"] != "target"):
        raise ValueError(f"{values['policy_version']} requires residual actions and target reward")
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
    metadata = {
        "policy_version": resolved["policy_version"],
        "anatomy_layout": LAYOUT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": V8_ACTION_SIZE if resolved["policy_version"] == V8_POLICY_VERSION else ACTION_SIZE,
        "goal_classes": list(goals.GOAL_CLASSES),
        "controllable_groups": len(CONTROLLABLE),
        **({"action_mode": action_mode, "reward_mode": reward_mode}
           if policy_version in RESIDUAL_POLICY_VERSIONS else {}),
    }
    if resolved["policy_version"] == V9_POLICY_VERSION:
        metadata["transfer_mode"] = transfer.DEFAULT_TRANSFER_MODE
    if resolved["policy_version"] == V8_POLICY_VERSION:
        metadata["observation_size"] = V8_OBSERVATION_SIZE
        metadata["anatomy_layers"] = list(LAYER_NAMES)
    return metadata


def build_observation(goal, histogram, start_agg: dict, solo_max_log, controllable,
                      layer_opacity=None) -> np.ndarray:
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
    if layer_opacity is not None:
        values.extend(layer_opacity)
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
                 balance_classes: bool = False, hindsight_brightness_share: float = 0.0):
        super().__init__()
        if not volume_ids:
            raise ValueError("volume_ids must not be empty")
        self.volume_ids = list(volume_ids)
        self._model_for_volume = model_for_volume
        if policy_version not in POLICY_VERSIONS:
            raise ValueError("unsupported one-shot policy version")
        if policy_version == POLICY_VERSION and action_mode != "absolute":
            raise ValueError("v6 only supports absolute action mode")
        if policy_version in RESIDUAL_POLICY_VERSIONS and action_mode != "residual":
            raise ValueError(f"{policy_version} requires residual action mode")
        if action_mode not in ("absolute", "residual"):
            raise ValueError("action_mode must be absolute or residual")
        if reward_mode not in ("attainment", "target"):
            raise ValueError("reward_mode must be attainment or target")
        if policy_version in RESIDUAL_POLICY_VERSIONS and reward_mode != "target":
            raise ValueError(f"{policy_version} requires target reward")
        if not 0.0 <= hindsight_ratio <= 1.0:
            raise ValueError("hindsight_ratio must be between 0 and 1")
        if not 0.0 <= hindsight_brightness_share <= 1.0:
            raise ValueError("hindsight_brightness_share must be between 0 and 1")
        # Share of hindsight episodes whose goal is a brightness change
        # (`_sample_hindsight_brightness`) rather than a visibility change.
        self.hindsight_brightness_share = float(hindsight_brightness_share)
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

        observation_size = V8_OBSERVATION_SIZE if policy_version == V8_POLICY_VERSION else OBSERVATION_SIZE
        action_size = V8_ACTION_SIZE if policy_version == V8_POLICY_VERSION else ACTION_SIZE
        self.observation_space = spaces.Box(
            low=-OBSERVATION_BOUND, high=OBSERVATION_BOUND, shape=(observation_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)

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
        layer_opacity = None
        if self.policy_version == V8_POLICY_VERSION:
            layer_opacity = [self._active_layers[name]["opacity"] for name in LAYER_NAMES]
        return build_observation(self._instruction["goal"], self._model.histogram, self._start_agg,  # pyright: ignore[reportOptionalSubscript]
                                   solo_max_log, controllable, layer_opacity)

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
        if self.policy_version in (V7_POLICY_VERSION, V9_POLICY_VERSION):
            info["policy_version"] = self.policy_version
        return info

    def hindsight_action(self):
        """The action that produced this episode's target, or None when the
        episode came from an instruction. Test and distillation hook."""
        return None if self._hindsight_action is None else self._hindsight_action.copy()

    def hindsight_policy_action(self):
        """The hindsight oracle in the policy's own action space: the absolute
        controllable values for "absolute" mode, their offset from the start
        state for "residual" mode (layer actions, v8 only, left at 0 -- no
        change). None for an instruction episode. The supervised target for
        `rl.pretrain`."""
        if self._hindsight_action is None:
            return None
        action = self._hindsight_action.copy()
        if self.action_mode == "residual":
            action = action - np.asarray(self._controllable_values(self._start_params))
        if self.policy_version == V8_POLICY_VERSION:
            action = np.concatenate([action, np.zeros(len(LAYER_NAMES))])
        return action.astype(np.float32)

    def _sample_hindsight_brightness(self, rng, start_params, start_agg):
        """(instruction-shaped dict, action) for a brightness goal on one
        visible class, or None when no class is visible enough."""
        visible = [c for c in goals.GOAL_CLASSES
                   if start_agg["vis"][c] >= HINDSIGHT_BRIGHTNESS_MIN_VIS]
        if not visible:
            return None
        start_controllable = np.asarray(self._controllable_values(start_params))
        for _ in range(self.HINDSIGHT_MAX_TRIES):
            goal_class = str(rng.choice(visible))
            base = goals.PEAK_INDEX[goal_class] * transfer.PARAMS_PER_PEAK
            group = CONTROLLABLE.index((base + 3, base + 4, base + 5))
            action = start_controllable.copy()
            action[group] = np.clip(action[group] + rng.uniform(-HINDSIGHT_BRIGHTNESS_NOISE,
                                                                HINDSIGHT_BRIGHTNESS_NOISE), -1.0, 1.0)
            target_agg = goals.aggregate(self._features(apply_controllable(start_params, action)),
                                         self._active_layers)
            delta = target_agg["bright"][goal_class] - start_agg["bright"][goal_class]
            if abs(delta) >= HINDSIGHT_BRIGHTNESS_MIN_DELTA:
                targets = {goal_class: {"bright": float(delta)}}
                return ({"kind": "hindsight", "text": None, "targets": targets,
                         "goal": goals.goal_vector(targets)}, action)
        return None

    def _sample_hindsight_goal(self, rng, model, start_params, start_agg):
        """(instruction-shaped dict, action) from a reachable target."""
        if self.hindsight_brightness_share and rng.random() < self.hindsight_brightness_share:
            sampled = self._sample_hindsight_brightness(rng, start_params, start_agg)
            if sampled is not None:
                return sampled
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
        if self.policy_version == V8_POLICY_VERSION:
            self._active_layers = normalize_layers(
                getattr(self, "_configured_active_layers", default_layers()))
        else:
            self._active_layers = None
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
            hu_action = action[:ACTION_SIZE]
            if self.policy_version == V8_POLICY_VERSION:
                reachable = set(goals.reachable_goal_classes(self._volume, self._model, self._active_layers))
                for name, residual in zip(LAYER_NAMES, action[ACTION_SIZE:]):
                    if name in reachable:
                        self._active_layers[name]["opacity"] = float(np.clip(1.0 + residual, 0.0, 1.0))
                action = hu_action
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
            # Fraction of start_distance closed, i.e. exactly `attainment`
            # (goals.attainment divides the same two distances) -- unlike the
            # raw (start_distance - final_distance) this used to be, it is on
            # a comparable scale for a brightness goal (start_distance ~0.3)
            # and a show_only goal (start_distance ~6): both can reach +1 for
            # a perfect solve, instead of brightness being squeezed toward 0
            # by the clip while show_only saturates it almost for free.
            target_progress = (start_distance - final_distance) / start_distance
            mentioned = set(self._instruction["targets"])
            visibility_progress, brightness_progress = goals.progress(self._start_agg, final_agg)
            drift = sum(abs(value) for key, value in visibility_progress.items()
                        if key not in mentioned)
            drift += goals.KAPPA * sum(abs(value) for key, value in brightness_progress.items()
                                       if key not in mentioned)
            self._last_drift = float(drift)
            # Same start_distance denominator as target_progress, so a given
            # fraction of the goal's own budget spent on collateral drift
            # costs the same whether the goal itself was easy or hard.
            reward = target_progress - TARGET_DRIFT_WEIGHT * float(drift) / start_distance
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
            if useless:
                reward -= USELESS_PENALTY
            info = self._info(attainment, useless)
            info.update({"target_progress": target_progress, "drift": drift})
        else:
            reward = float(np.clip(attainment, -REWARD_CLIP, REWARD_CLIP)) - (USELESS_PENALTY if useless else 0.0)
            info = self._info(attainment, useless)

        obs = self._build_observation()
        return obs, float(reward), True, False, info
