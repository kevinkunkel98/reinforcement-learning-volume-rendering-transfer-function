"""The search and the rule baseline, and the action space they share with the policy.

`hill_climb` is a coordinate search on `goals.distance` over the controllable
groups (`CONTROLLABLE`: width, height, and one brightness step applied to
r/g/b together, per peak -- centres never move, as for the policy). From the
task's start it is the "search from scratch" baseline; warm-started at the
policy's answer it is the viewer's final polish.

`rule_executor` does what the command executor (`commands.py`) does with an
instruction: act only on the mentioned classes' peaks.

Both take `(model, start_params, instruction)`: a `visibility.VisibilityModel`,
a transfer-function vector and what `goals.sample_instruction` returns.
"""
import math

import numpy as np

import commands
import goals
import transfer

N_PEAKS = transfer.N_PEAKS
PARAMS_PER_PEAK = transfer.PARAMS_PER_PEAK


def _controllable_dims() -> tuple:
    """Width, height, and brightness groups for every canonical peak."""
    dims = []
    for peak in range(N_PEAKS):
        base = peak * PARAMS_PER_PEAK
        dims.append((base + 1,))                      # width
        dims.append((base + 2,))                       # height
        dims.append((base + 3, base + 4, base + 5))    # brightness (r, g, b together)
    return tuple(dims)


CONTROLLABLE = _controllable_dims()


def apply_controllable(start_params: np.ndarray, action) -> np.ndarray:
    """`start_params` with each controllable group set to its action value.

    A group's value is its *mean* -- the same quantity the observation reports
    for it (`rl.oneshot_env.build_observation`'s `controllable`), so an action
    round-trips through the observation unchanged. Each index keeps its offset
    from that mean, which matters only for the (r, g, b) group: the offsets are
    the peak's hue (writing the scalar into all three channels made every
    policy render grey).
    """
    params = np.asarray(start_params, dtype=np.float64).copy()
    for group, value in zip(CONTROLLABLE, action):
        offset = float(np.mean([params[i] for i in group]))
        offsets = np.asarray([params[i] - offset for i in group], dtype=np.float64)
        # Keep RGB hue offsets while projecting them into the parameter bounds.
        for _ in range(len(group)):
            clipped = np.clip(value + offsets, -1.0, 1.0)
            residual = float(value - np.mean(clipped))
            free = (clipped > -1.0) & (clipped < 1.0)
            if not np.any(free) or abs(residual) < 1e-12:
                break
            offsets[free] += residual * len(offsets) / np.count_nonzero(free)
        for index, channel in zip(group, np.clip(value + offsets, -1.0, 1.0)):
            params[index] = float(channel)
    return np.clip(params, -1.0, 1.0)


def _nearest_key(value: float, table: dict) -> str:
    """The key of `table` whose value is closest to abs(value) -- recovers
    the strength/level word a sampled instruction's numeric delta came
    from, since `instruction` carries only the delta, not the word."""
    target = abs(value)
    return min(table, key=lambda key: abs(table[key] - target))


def _direction(value: float) -> str:
    return "increase" if value >= 0.0 else "decrease"


def _mentioned(instruction: dict, aspect: str) -> dict:
    return {goal_class: entry[aspect] for goal_class, entry in instruction["targets"].items()
            if aspect in entry}


def _infer_level(model, goal_class: str, delta: float, start_vis: float) -> str:
    """Which ABSOLUTE_LEVEL word produced `delta`, by inverting
    `goals._absolute_target_delta` against every candidate level."""
    solo = goals.class_ceiling(model, goal_class)
    target_vis = 10.0 ** (delta + math.log10(start_vis + goals.EPSILON)) - goals.EPSILON
    fraction = target_vis / solo if solo > 0.0 else 0.0
    return min(goals.ABSOLUTE_LEVEL, key=lambda key: abs(goals.ABSOLUTE_LEVEL[key] - fraction))


def rule_executor(model, start_params, instruction) -> np.ndarray:
    """What the command executor does with the instruction: act only on the
    peak of each mentioned goal class."""
    params = start_params.copy()
    kind = instruction["kind"]
    targets = instruction["targets"]

    if kind == "show_only":
        shown = {c for c, entry in targets.items() if entry.get("vis", 0.0) > 0.0}
        for goal_class, idx in transfer.ANATOMICAL_PEAK_INDEX.items():
            base = idx * PARAMS_PER_PEAK
            params[base + 2] = transfer._from_unit(0.7 if goal_class in shown else 0.0)
            if goal_class in shown:
                # A wide peak still has real opacity reaching into a
                # neighbour's HU range, so boosting it alone lights up other
                # tissue too -- cap (never widen) so an already-narrow peak
                # is untouched.
                current_width = transfer.peak_internal(params, idx)["width"]
                params[base + 1] = transfer._from_range(
                    min(current_width, transfer.SHOW_ONLY_MAX_WIDTH_HU), *transfer.WIDTH_RANGE)
        return params

    if kind == "absolute":
        start_vis = goals.aggregate(model.features(start_params))["vis"]
        for goal_class, delta in _mentioned(instruction, "vis").items():
            level = _infer_level(model, goal_class, delta, start_vis[goal_class])
            base = transfer.ANATOMICAL_PEAK_INDEX[goal_class] * PARAMS_PER_PEAK
            params[base + 2] = transfer._from_unit(commands.LEVEL_WORDS[level])
        return params

    if kind == "brightness":
        for goal_class, delta in _mentioned(instruction, "bright").items():
            strength = _nearest_key(delta, goals.BRIGHTNESS_STRENGTH)
            step = commands.STRENGTH_WORDS[strength]
            direction = _direction(delta)
            base = transfer.ANATOMICAL_PEAK_INDEX[goal_class] * PARAMS_PER_PEAK
            for offset in (3, 4, 5):
                new_ext = commands._asymptotic_step(float(params[base + offset]), direction, step)
                params[base + offset] = float(np.clip(new_ext, -1.0, 1.0))
        return params

    # relative or compound: one asymptotic step on height per mentioned class
    for goal_class, delta in _mentioned(instruction, "vis").items():
        strength = _nearest_key(delta, goals.VISIBILITY_STRENGTH)
        step = commands.STRENGTH_WORDS[strength]
        direction = _direction(delta)
        base = transfer.ANATOMICAL_PEAK_INDEX[goal_class] * PARAMS_PER_PEAK
        new_ext = commands._asymptotic_step(float(params[base + 2]), direction, step)
        params[base + 2] = float(np.clip(new_ext, -1.0, 1.0))
    return params


def _evaluate(model, goal, start_agg, params, weights: dict = None) -> float:
    return goals.distance(goal, start_agg, goals.aggregate(model.features(params)), weights)


def hill_climb(model, start_params, instruction, evaluations: int = 200,
               initial_step: float = 0.2, reference_params=None) -> np.ndarray:
    """Coordinate search on `goals.distance`: try +-step on each of the 12
    controllable values, keep improvements, halve the step once a full sweep
    finds none. `evaluations` bounds the number of `model.features` calls
    (one to score the start state, one per proposal after that). Applies
    `instruction`'s own `weights` (set for show_only; None otherwise) so this
    search optimizes the identical objective the RL policy is trained and
    judged on.

    `reference_params` (default: `start_params`) is the state the
    instruction's changes are measured from -- the episode's original start.
    Passing it separately warm-starts the search at `start_params` (e.g. a
    policy's proposal) while still optimising the episode's own objective;
    measured from the proposal instead, a relative goal would be applied a
    second time on top of the progress the proposal already made. The warm
    start then costs one extra evaluation, to score it."""
    goal = instruction["goal"]
    weights = instruction.get("weights")
    reference = start_params if reference_params is None else reference_params
    start_agg = goals.aggregate(model.features(reference))
    used = 1

    best_params = start_params.copy()
    if reference_params is None:
        best_distance = goals.distance(goal, start_agg, start_agg, weights)
    else:
        best_distance = _evaluate(model, goal, start_agg, best_params, weights)
        used += 1
    step = initial_step

    while used < evaluations and step > 1e-3:
        improved = False
        for dim in CONTROLLABLE:
            for sign in (1.0, -1.0):
                if used >= evaluations:
                    break
                candidate = best_params.copy()
                for idx in dim:
                    candidate[idx] = float(np.clip(candidate[idx] + sign * step, -1.0, 1.0))
                candidate_distance = _evaluate(model, goal, start_agg, candidate, weights)
                used += 1
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_params = candidate
                    improved = True
            if used >= evaluations:
                break
        if not improved:
            step /= 2.0
    return best_params
