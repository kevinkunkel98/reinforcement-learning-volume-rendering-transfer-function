"""Episodes of the one-shot task: a start transfer function on one volume, an
instruction, and how an answer to it is scored.

The policy sees the instruction and the start state (`Episode.observation`)
and answers with a residual action: a change to each controllable parameter
group (`rl.baselines.CONTROLLABLE`: height, width and colour per peak -- the
centres stay fixed). The answer is scored by `goals.attainment` against the
episode's start.

`EpisodeSampler.reset(seed)` draws an episode from a seed, consuming the
random stream exactly as the gymnasium environment it replaces did, so every
recorded seed (training data, the fixed evaluation episodes) still yields the
same episode.
"""
import json
import math
import os

import numpy as np

import scans
import goals
import visibility
from anatomy import LAYOUT_VERSION
from rl.search import CONTROLLABLE, apply_controllable

N_CLASSES = len(goals.GOAL_CLASSES)
HISTOGRAM_SIZE = visibility.HISTOGRAM_BINS
# goal (4n) + histogram + log10 visibility (n) + brightness (n)
# + log10 ceiling (n) + controllable start values + coverage (1)
OBSERVATION_SIZE = 4 * N_CLASSES + HISTOGRAM_SIZE + 3 * N_CLASSES + len(CONTROLLABLE) + 1
ACTION_SIZE = len(CONTROLLABLE)
# Every packed observation value stays well inside this bound.
OBSERVATION_BOUND = 10.0
# Start noise per controllable group, in normalized parameter units.
START_NOISE = 0.3
# Below this, the instruction asks for effectively nothing at the start and
# attainment's ratio is meaningless: report 0 instead of dividing by ~0.
ATTAINMENT_FLOOR = 1e-9
POLICY_VERSION = "oneshot-v9"
EVAL_EPISODES = 300


def controllable_values(params) -> list:
    """The mean of each controllable group: the values a residual action moves."""
    return [float(np.mean([params[i] for i in group])) for group in CONTROLLABLE]


def solo_max_log(model) -> list:
    """log10 of each goal class's ceiling (`goals.class_ceiling`)."""
    return [math.log10(goals.class_ceiling(model, c) + goals.EPSILON) for c in goals.GOAL_CLASSES]


def build_observation(goal, histogram, start_agg: dict, ceilings_log, controllable) -> np.ndarray:
    """The policy's input, the single implementation of its layout:
    goal + histogram + log10 visibility + brightness + log10 ceilings
    + controllable start values + coverage."""
    log_vis = [math.log10(start_agg["vis"][c] + goals.EPSILON) for c in goals.GOAL_CLASSES]
    bright = [start_agg["bright"][c] for c in goals.GOAL_CLASSES]
    values = (list(goal) + list(histogram) + log_vis + bright
              + list(ceilings_log) + list(controllable) + [start_agg["coverage"]])
    return np.asarray(values, dtype=np.float32)


def apply_residual(start_params, residual) -> np.ndarray:
    """The transfer function a residual action gives from `start_params`."""
    controllable = np.asarray(controllable_values(start_params))
    action = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
    return apply_controllable(start_params, np.clip(controllable + action, -1.0, 1.0))


def sample_start(rng) -> np.ndarray:
    """The task's start transfer function with uniform noise per controllable group."""
    params = goals.starting_params()
    for group in CONTROLLABLE:
        noise = rng.uniform(-START_NOISE, START_NOISE)
        for index in group:
            params[index] = params[index] + noise
    return np.clip(params, -1.0, 1.0)


class Episode:
    """One start state on one volume and one instruction."""

    def __init__(self, volume: str, model, start_params, instruction: dict):
        self.volume, self.model, self.instruction = volume, model, instruction
        self.start_params = np.asarray(start_params, dtype=np.float64).copy()
        self.start_agg = goals.aggregate(model.features(self.start_params))
        self.start_distance = goals.distance(instruction["goal"], self.start_agg, self.start_agg,
                                             instruction.get("weights"))

    @property
    def kind(self) -> str:
        return self.instruction["kind"]

    def observation(self) -> np.ndarray:
        return build_observation(self.instruction["goal"], self.model.histogram, self.start_agg,
                                 solo_max_log(self.model), controllable_values(self.start_params))

    def params_for(self, residual) -> np.ndarray:
        return apply_residual(self.start_params, residual)

    def aggregate(self, params) -> dict:
        return goals.aggregate(self.model.features(params))

    def attainment(self, params) -> float:
        if self.start_distance <= ATTAINMENT_FLOOR:
            return 0.0
        return goals.attainment(self.instruction["goal"], self.start_agg, self.aggregate(params),
                                self.instruction.get("weights"))

    def visible_pp(self, params):
        """`goals.visible_change`: percentage points of the image moved toward the goal."""
        return goals.visible_change(self.instruction["goal"], self.start_agg, self.aggregate(params),
                                    self.instruction.get("weights"))


class EpisodeSampler:
    """Seeded episodes over `volume_ids`. `balance_classes` makes each new
    instruction prefer the class asked about least so far (training data)."""

    def __init__(self, volume_ids, model_for_volume=visibility.for_volume, balance_classes: bool = False):
        if not volume_ids:
            raise ValueError("volume_ids must not be empty")
        self.volume_ids = list(volume_ids)
        self.model_for_volume = model_for_volume
        self.balance_classes = balance_classes
        self.class_counts = {}

    def sample_instruction(self, rng, volume: str, model, start_params, start_agg: dict) -> dict:
        """One instruction from the task's grammar, made well-posed for this start."""
        instruction = goals.sample_instruction(volume, model, start_agg, rng, self.class_counts,
                                               self.balance_classes)
        for goal_class in instruction["targets"]:
            self.class_counts[goal_class] = self.class_counts.get(goal_class, 0) + 1
        return goals.well_posed(instruction, model, start_params, start_agg)

    def reset(self, seed: int) -> Episode:
        rng = np.random.default_rng(seed)
        volume = str(rng.choice(self.volume_ids))
        model = self.model_for_volume(volume)
        start_params = sample_start(rng)
        start_agg = goals.aggregate(model.features(start_params))
        # The replaced environment drew here to pick a hindsight goal (always
        # off for this task); drawing keeps every seed's episode the same.
        rng.random()
        instruction = self.sample_instruction(rng, volume, model, start_params, start_agg)
        return Episode(volume, model, start_params, instruction)


def fixed_episodes(split: str, count: int = EVAL_EPISODES, seed: int = 0, volume_ids=None,
                   model_for_volume=visibility.for_volume) -> list:
    """The evaluation's fixed episodes: `reset(seed + i)` over the split's
    volumes, the same list every time."""
    ids = volume_ids if volume_ids is not None else scans.split_names(split)
    sampler = EpisodeSampler(ids, model_for_volume)
    return [sampler.reset(seed + i) for i in range(count)]


def policy_metadata(checkpoint_path: str) -> dict:
    """The metadata.json beside a checkpoint, checked against this task."""
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "metadata.json")
    with open(path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    expected = {"policy_version": POLICY_VERSION, "action_mode": "residual", "task_version": "v3",
                "transfer_mode": "label_scoped", "observation_size": OBSERVATION_SIZE,
                "action_size": ACTION_SIZE, "goal_classes": list(goals.GOAL_CLASSES)}
    wrong = {k: metadata.get(k) for k, v in expected.items() if metadata.get(k) != v}
    if wrong:
        raise ValueError(f"{checkpoint_path} was trained for another task or layout: {wrong}")
    return metadata


def contract_metadata() -> dict:
    """What a checkpoint trained here records about its task (`policy_metadata` checks it)."""
    return {"policy_version": POLICY_VERSION, "anatomy_layout": LAYOUT_VERSION,
            "observation_size": OBSERVATION_SIZE, "action_size": ACTION_SIZE,
            "goal_classes": list(goals.GOAL_CLASSES), "controllable_groups": len(CONTROLLABLE),
            "action_mode": "residual", "reward_mode": "target", "transfer_mode": "label_scoped",
            "task_version": "v3"}
