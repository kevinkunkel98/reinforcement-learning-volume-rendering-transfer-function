"""Non-learned ways to satisfy an instruction -- what the policy must beat.

Every baseline has the signature `(model, start_params, instruction) ->
final_params`, where `model` is a `visibility.VisibilityModel`, `start_params`
is a 24-value transfer-function vector, and `instruction` is what
`goals.sample_instruction` returns (`{"kind", "text", "targets", "goal"}`).

B0 `do_nothing` is the reference: it returns `start_params` unchanged, and
under `goals.distance`'s KEEP_TOLERANCE it scores attainment exactly 0.

B1 `current_executor` replicates today's command executor (`commands.py`):
it acts only on the peak of the mentioned goal class(es), reading the same
strength/level words but touching nothing else. B5 `occlusion_rule` adds the
"strip everything else" heuristic a person would hand-write. B2
`random_policy` and B3/B4 `hill_climb` search a reduced 12-value action space
(height, width, and one "brightness" step applied to r/g/b together, per
peak) -- centers are never touched, matching RL v2's action space.
"""
import math

import numpy as np

import anatomy
import commands
import goals
import transfer

N_PEAKS = len(anatomy.CANONICAL_PEAK_ORDER)
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
    the peak's hue. Writing the scalar into all three channels instead (the
    original decode) forced r = g = b, so every policy render came out grey
    while the baselines -- which add a delta and so keep their offsets -- stayed
    coloured. A rater could then tell which candidate was the policy's at a
    glance, which is fatal for a blind preference comparison.
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


def _start_vis(model, start_params) -> dict:
    return goals.aggregate(model.features(start_params))["vis"]


def _infer_level(model, goal_class: str, delta: float, start_vis: float) -> str:
    """Which ABSOLUTE_LEVEL word produced `delta`, by inverting
    `goals._absolute_target_delta` against every candidate level."""
    solo = sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[goal_class])
    target_vis = 10.0 ** (delta + math.log10(start_vis + goals.EPSILON)) - goals.EPSILON
    fraction = target_vis / solo if solo > 0.0 else 0.0
    return min(goals.ABSOLUTE_LEVEL, key=lambda key: abs(goals.ABSOLUTE_LEVEL[key] - fraction))


def do_nothing(model, start_params, instruction) -> np.ndarray:
    """B0: the reference every other baseline (and any learned policy) has
    to beat. Under `goals.distance`'s KEEP_TOLERANCE, inaction scores
    attainment exactly 0 -- it is not the floor (B2 random can score below
    it), it is the "did following the instruction actually help" line."""
    return start_params.copy()


def current_executor(model, start_params, instruction) -> np.ndarray:
    """What today's command executor does with the instruction: act only on
    the peak `goals.PEAK_INDEX` gives each mentioned goal class."""
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
        start_vis = _start_vis(model, start_params)
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


def _increasing_or_shown_classes(instruction: dict) -> set:
    return {c for c, entry in instruction["targets"].items() if entry.get("vis", 0.0) > 0.0}


def occlusion_rule(model, start_params, instruction) -> np.ndarray:
    """`current_executor`, and additionally: if any class is being increased
    or shown, strip every *other* peak's height to 0.02 -- the "hide
    everything else" heuristic a person writes by hand."""
    params = current_executor(model, start_params, instruction)
    shown = _increasing_or_shown_classes(instruction)
    if shown:
        for goal_class, idx in transfer.ANATOMICAL_PEAK_INDEX.items():
            if goal_class not in shown:
                base = idx * PARAMS_PER_PEAK
                params[base + 2] = transfer._from_unit(0.02)
    return params


def random_policy(model, start_params, instruction, seed: int = 0) -> np.ndarray:
    """10 random +-0.1 steps on the 12 controllable values, seeded -- ignores
    the instruction and the objective entirely; the floor any policy (learned
    or heuristic) has to clear."""
    rng = np.random.default_rng(seed)
    params = start_params.copy()
    for _ in range(10):
        dim = CONTROLLABLE[int(rng.integers(0, len(CONTROLLABLE)))]
        delta = 0.1 if rng.integers(0, 2) == 0 else -0.1
        for idx in dim:
            params[idx] = float(np.clip(params[idx] + delta, -1.0, 1.0))
    return params


def _evaluate(model, goal, start_agg, params) -> float:
    return goals.distance(goal, start_agg, goals.aggregate(model.features(params)))


def hill_climb(model, start_params, instruction, evaluations: int = 200,
               initial_step: float = 0.2) -> np.ndarray:
    """Coordinate search on `goals.distance`: try +-step on each of the 12
    controllable values, keep improvements, halve the step once a full sweep
    finds none. `evaluations` bounds the number of `model.features` calls
    (one to score the start state, one per proposal after that)."""
    goal = instruction["goal"]
    start_agg = goals.aggregate(model.features(start_params))
    used = 1

    best_params = start_params.copy()
    best_distance = goals.distance(goal, start_agg, start_agg)
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
                candidate_distance = _evaluate(model, goal, start_agg, candidate)
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


def hill_climb_10(model, start_params, instruction) -> np.ndarray:
    return hill_climb(model, start_params, instruction, evaluations=10)


def hill_climb_200(model, start_params, instruction) -> np.ndarray:
    return hill_climb(model, start_params, instruction, evaluations=200)


BASELINES = {
    "B0_do_nothing": do_nothing,
    "B1_current_executor": current_executor,
    "B2_random_policy": random_policy,
    "B3_hill_climb_10": hill_climb_10,
    "B4_hill_climb_200": hill_climb_200,
    "B5_occlusion_rule": occlusion_rule,
}
