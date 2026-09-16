"""Turn an instruction into a measurable target.

Visibility (`visibility.for_volume(...).features(...)`) is measured over five
anatomical classes, but a transfer function cannot separate all of them:
organs and muscle overlap in intensity on both plain and contrast scans, so no
peak placement can show one without the other -- goals target them jointly as
`soft`. Vessels only stand apart from soft tissue when the scan carries
contrast, so vessels are only ever a goal on a contrast volume (see
`goal_classes_for_volume`).

A goal is a 16-value vector: a requested visibility/brightness change per goal
class, plus a flag saying whether that aspect was even mentioned (an
instruction that never mentions vessels should not be scored on what happens
to them). Visibility changes are scored in log10 units because visibility
spans three orders of magnitude (0.0003 to 0.5) -- a fixed absolute change
means very different things at each end, but a fixed factor does not.
Brightness changes are scored in plain units.
"""
import math

import numpy as np

import totalseg
from transfer import PARAMS_PER_PEAK, TOTAL_PARAMS, CENTER_RANGE, WIDTH_RANGE, _from_range, _from_unit

GOAL_CLASSES = ("skeleton", "lungs", "soft", "vessels")
MEASURED_FOR_GOAL = {"skeleton": ("skeleton",), "lungs": ("lungs",),
                     "soft": ("organs", "muscle"), "vessels": ("vessels",)}

# RL v2 peak order, by the class each peak is seeded for
PEAK_CENTRES_HU = {"lungs": -800.0, "soft": 40.0, "vessels": 300.0, "skeleton": 900.0}
PEAK_INDEX = {"lungs": 0, "soft": 1, "vessels": 2, "skeleton": 3}
PEAK_WIDTHS_HU = {"lungs": 60.0, "soft": 80.0, "vessels": 80.0, "skeleton": 280.0}
PEAK_HEIGHTS = {"lungs": 0.05, "soft": 0.15, "vessels": 0.3, "skeleton": 0.6}
PEAK_COLOURS = {"lungs": (0.55, 0.70, 0.95), "soft": (0.85, 0.35, 0.35),
                "vessels": (0.90, 0.45, 0.40), "skeleton": (0.95, 0.95, 0.90)}

EPSILON = 1e-3           # floor inside log10, so "invisible" is finite
KAPPA = 1.5              # brightness weight: 0.2 brightness ~= 0.3 log10 visibility
LAMBDA_KEEP = 0.3        # weight of "leave the unmentioned classes alone"
KEEP_TOLERANCE = 0.15    # unmentioned-class drift this small is free (see distance)
VISIBILITY_STRENGTH = {"slightly": 0.15, "moderately": 0.3, "strongly": 0.6}   # log10 units
BRIGHTNESS_STRENGTH = {"slightly": 0.1, "moderately": 0.2, "strongly": 0.4}
ABSOLUTE_LEVEL = {"low": 0.1, "medium": 0.4, "high": 0.8}                      # share of solo_max


def starting_params() -> np.ndarray:
    """RL v2's starting transfer function: one peak per goal class."""
    params = np.zeros(TOTAL_PARAMS, dtype=np.float64)
    for goal_class, index in PEAK_INDEX.items():
        base = index * PARAMS_PER_PEAK
        params[base + 0] = _from_range(PEAK_CENTRES_HU[goal_class], *CENTER_RANGE)
        params[base + 1] = _from_range(PEAK_WIDTHS_HU[goal_class], *WIDTH_RANGE)
        params[base + 2] = _from_unit(PEAK_HEIGHTS[goal_class])
        r, g, b = PEAK_COLOURS[goal_class]
        params[base + 3] = _from_unit(r)
        params[base + 4] = _from_unit(g)
        params[base + 5] = _from_unit(b)
    return params


def aggregate(features: dict) -> dict:
    """Measured classes -> goal classes.

    {"vis": {goal class: float}, "bright": {goal class: float}, "coverage": float}
    `soft` sums the visibility of organs and muscle, because no transfer
    function can separate them; its brightness is their visibility-weighted
    mean (0 when neither is visible).
    """
    vis, bright = {}, {}
    for goal_class in GOAL_CLASSES:
        measured = MEASURED_FOR_GOAL[goal_class]
        total_vis = sum(features["vis"][m] for m in measured)
        vis[goal_class] = total_vis
        if total_vis > 0.0:
            bright[goal_class] = sum(features["vis"][m] * features["bright"][m] for m in measured) / total_vis
        else:
            bright[goal_class] = 0.0
    return {"vis": vis, "bright": bright, "coverage": features["coverage"]}


def goal_vector(targets: dict) -> np.ndarray:
    """A 16-value goal: requested change and a mentioned-flag per goal class.

    targets maps goal class -> {"vis": float, "bright": float}; a missing key
    means that aspect is not part of the instruction. Layout: d[4], m[4],
    e[4], n[4] in GOAL_CLASSES order.
    """
    vector = np.zeros(4 * len(GOAL_CLASSES), dtype=np.float64)
    for i, goal_class in enumerate(GOAL_CLASSES):
        entry = targets.get(goal_class, {})
        if "vis" in entry:
            vector[i] = entry["vis"]
            vector[len(GOAL_CLASSES) + i] = 1.0
        if "bright" in entry:
            vector[2 * len(GOAL_CLASSES) + i] = entry["bright"]
            vector[3 * len(GOAL_CLASSES) + i] = 1.0
    return vector


def progress(start: dict, current: dict) -> tuple:
    """(c, b): change so far per goal class.

    c = log10(vis + EPSILON) - log10(vis_start + EPSILON); b = bright - bright_start.
    Both aggregated dicts from aggregate().
    """
    c, b = {}, {}
    for goal_class in GOAL_CLASSES:
        c[goal_class] = (math.log10(current["vis"][goal_class] + EPSILON)
                          - math.log10(start["vis"][goal_class] + EPSILON))
        b[goal_class] = current["bright"][goal_class] - start["bright"][goal_class]
    return c, b


def distance(goal: np.ndarray, start: dict, current: dict) -> float:
    """How far a state is from the goal.

    sum m*|c - d| + KAPPA*sum n*|b - e|
      + LAMBDA_KEEP*sum (1-m)*max(0, |c| - KEEP_TOLERANCE)
      + LAMBDA_KEEP*KAPPA*sum (1-n)*max(0, |b| - KEEP_TOLERANCE)

    Moving one peak inevitably shifts what occludes what elsewhere in the
    volume, so an unmentioned class always drifts a little even when the
    instruction is followed well; a keep term that punishes any drift at
    all makes leaving the transfer function untouched score better than
    acting on the instruction. KEEP_TOLERANCE gives that small, unavoidable
    side effect for free and only penalises drift beyond it.
    """
    c, b = progress(start, current)
    n_classes = len(GOAL_CLASSES)
    d, m, e, n = goal[0:n_classes], goal[n_classes:2 * n_classes], \
        goal[2 * n_classes:3 * n_classes], goal[3 * n_classes:4 * n_classes]
    total = 0.0
    for i, goal_class in enumerate(GOAL_CLASSES):
        total += m[i] * abs(c[goal_class] - d[i]) + KAPPA * n[i] * abs(b[goal_class] - e[i])
        total += LAMBDA_KEEP * (1.0 - m[i]) * max(0.0, abs(c[goal_class]) - KEEP_TOLERANCE)
        total += LAMBDA_KEEP * KAPPA * (1.0 - n[i]) * max(0.0, abs(b[goal_class]) - KEEP_TOLERANCE)
    return float(total)


def attainment(goal, start, final) -> float:
    """1 - distance(final) / distance(start); 1 = goal reached, < 0 = worse."""
    return 1.0 - distance(goal, start, final) / distance(goal, start, start)


def is_useless(features: dict) -> bool:
    """Nothing drawn, or an opaque wall hiding everything."""
    return (features["coverage"] < 0.01
            or sum(features["vis"].values()) < 0.001)


ATTAINMENT_CLIP = 1.0


def summarise_attainment(values) -> dict:
    """Aggregate episode attainment robustly.

    Attainment is 1 - D_final/D_start, so it is unbounded below: a single bad
    episode on an easy goal (small D_start) can read -17 and swamp a mean.
    Report the median and a mean of values clipped to [-1, 1] alongside the
    raw mean, plus the share of episodes that improved on doing nothing.
    """
    filtered = [v for v in values if v is not None]
    n = len(filtered)
    if n == 0:
        return {"median": None, "mean_clipped": None, "mean_raw": None,
                "share_positive": None, "n": 0}
    array = np.asarray(filtered, dtype=np.float64)
    clipped = np.clip(array, -ATTAINMENT_CLIP, ATTAINMENT_CLIP)
    return {
        "median": float(np.median(array)),
        "mean_clipped": float(np.mean(clipped)),
        "mean_raw": float(np.mean(array)),
        "share_positive": float(np.mean(array > 0.0)),
        "n": n,
    }


INSTRUCTION_MIX = (("relative", 0.40), ("compound", 0.25), ("show_only", 0.15),
                   ("absolute", 0.10), ("brightness", 0.10))

# "show only" hides the unnamed classes hard -- bigger than any VISIBILITY_STRENGTH
# so a hill-climber (or a policy) reads it as "push these away", not "nudge them".
HIDE_STRENGTH = 1.0

# Spoken noun phrases per goal class, several synonyms each.
CLASS_WORDS = {
    "skeleton": ("bone", "skeleton"),
    "lungs": ("lungs",),
    "soft": ("soft tissue",),
    "vessels": ("vessels", "blood vessels"),
}


def goal_classes_for_volume(name: str) -> list:
    """Goal classes this volume supports: those with labels present, with
    `vessels` only on contrast scans (elsewhere vessels share intensities with
    soft tissue and no transfer function can single them out)."""
    present = set(totalseg.classes_present(name))
    supported = []
    for goal_class in GOAL_CLASSES:
        if goal_class == "vessels":
            if "vessels" in present and totalseg.is_contrast(name):
                supported.append(goal_class)
        elif any(m in present for m in MEASURED_FOR_GOAL[goal_class]):
            supported.append(goal_class)
    return supported


def _class_word(goal_class: str, rng) -> str:
    return rng.choice(CLASS_WORDS[goal_class])


def _relative_text(goal_class: str, direction: str, strength: str, rng) -> str:
    word = _class_word(goal_class, rng)
    verb = "more" if direction == "increase" else "less"
    action = "increase" if direction == "increase" else "decrease"
    templates = {
        "slightly": [f"a bit {verb} {word}"],
        "moderately": [f"{verb} {word}", f"{action} opacity for the {word}"],
        "strongly": [f"a lot {verb} {word}", f"{action} opacity for the {word} strongly"],
    }[strength]
    return rng.choice(templates)


def _absolute_target_delta(model, goal_class: str, level: str, start_vis: float) -> float:
    """The vis change an absolute-level instruction requests: `level`'s share
    of solo_max (aggregated for the goal class) as a log10 change from
    `start_vis`, so "high skeleton" means the same thing on every volume."""
    solo = sum(model.solo_max(m) for m in MEASURED_FOR_GOAL[goal_class])
    target_vis = ABSOLUTE_LEVEL[level] * solo
    return math.log10(target_vis + EPSILON) - math.log10(start_vis + EPSILON)


def _absolute_text(goal_class: str, level: str, rng) -> str:
    word = _class_word(goal_class, rng)
    templates = [f"{level} opacity {word}", f"{level} opacity for the {word}"]
    return rng.choice(templates)


def _brightness_text(goal_class: str, direction: str, rng) -> str:
    word = _class_word(goal_class, rng)
    verb = "brighten" if direction == "increase" else "darken"
    return f"{verb} the {word}"


def _sample_kind(rng) -> str:
    kinds = [k for k, _ in INSTRUCTION_MIX]
    weights = [w for _, w in INSTRUCTION_MIX]
    return str(rng.choice(kinds, p=weights))


def _sample_relative(classes: list, rng) -> tuple:
    goal_class = rng.choice(classes)
    direction = rng.choice(("increase", "decrease"))
    strength = rng.choice(list(VISIBILITY_STRENGTH))
    sign = 1.0 if direction == "increase" else -1.0
    targets = {goal_class: {"vis": sign * VISIBILITY_STRENGTH[strength]}}
    return targets, _relative_text(goal_class, direction, strength, rng)


def _sample_compound(classes: list, rng) -> tuple:
    n = min(2, len(classes))
    chosen = list(rng.choice(classes, size=n, replace=False))
    targets, phrases = {}, []
    for goal_class in chosen:
        direction = rng.choice(("increase", "decrease"))
        strength = rng.choice(list(VISIBILITY_STRENGTH))
        sign = 1.0 if direction == "increase" else -1.0
        targets[goal_class] = {"vis": sign * VISIBILITY_STRENGTH[strength]}
        phrases.append(_relative_text(goal_class, direction, strength, rng))
    return targets, ", ".join(phrases)


def _sample_show_only(classes: list, rng) -> tuple:
    n = min(int(rng.integers(1, 3)), len(classes))
    shown = list(rng.choice(classes, size=n, replace=False))
    targets = {goal_class: {"vis": HIDE_STRENGTH} for goal_class in shown}
    for goal_class in classes:
        if goal_class not in shown:
            targets[goal_class] = {"vis": -HIDE_STRENGTH}
    words = [_class_word(goal_class, rng) for goal_class in shown]
    text = "show only the " + " and the ".join(words)
    return targets, text


def _sample_absolute(classes: list, model, start_features: dict, rng) -> tuple:
    goal_class = rng.choice(classes)
    level = rng.choice(list(ABSOLUTE_LEVEL))
    delta = _absolute_target_delta(model, goal_class, level, start_features["vis"][goal_class])
    targets = {goal_class: {"vis": delta}}
    return targets, _absolute_text(goal_class, level, rng)


def _sample_brightness(classes: list, rng) -> tuple:
    goal_class = rng.choice(classes)
    direction = rng.choice(("increase", "decrease"))
    strength = rng.choice(list(BRIGHTNESS_STRENGTH))
    sign = 1.0 if direction == "increase" else -1.0
    targets = {goal_class: {"bright": sign * BRIGHTNESS_STRENGTH[strength]}}
    return targets, _brightness_text(goal_class, direction, rng)


def sample_instruction(name, model, start_features, rng) -> dict:
    """One instruction: {"kind", "text", "targets", "goal"}.

    `targets` is the goal_vector input; `text` is the spoken form. Absolute
    levels use ABSOLUTE_LEVEL x model.solo_max aggregated for the goal class,
    turned into a requested change from the start state, so "high skeleton"
    means the same thing on every volume.
    """
    classes = goal_classes_for_volume(name)
    kind = _sample_kind(rng)
    if kind == "relative":
        targets, text = _sample_relative(classes, rng)
    elif kind == "compound":
        targets, text = _sample_compound(classes, rng)
    elif kind == "show_only":
        targets, text = _sample_show_only(classes, rng)
    elif kind == "absolute":
        targets, text = _sample_absolute(classes, model, start_features, rng)
    else:
        targets, text = _sample_brightness(classes, rng)
    return {"kind": kind, "text": text, "targets": targets, "goal": goal_vector(targets)}


# --- Task 2: parsed commands -> goals -----------------------------------------

_NOT_A_GOAL = ("{what} is not a goal -- apply it exactly via commands.apply_command")

_CLASS_LABEL = {"skeleton": "the skeleton", "lungs": "the lungs",
                "soft": "soft tissue", "vessels": "the vessels"}


def _check_class_supported(goal_class: str, volume) -> None:
    """Raise if `goal_class` isn't something this volume can be a goal for --
    vessels only exist as a goal on contrast scans, and a class whose labels
    aren't present on this volume at all can't be measured either."""
    if volume is None:
        return
    supported = goal_classes_for_volume(volume)
    if goal_class in supported:
        return
    if goal_class == "vessels":
        raise ValueError(f"vessels is not a goal on {volume!r} -- it has no contrast")
    raise ValueError(f"{goal_class!r} is not present on volume {volume!r}")


def _relative_command_text(goal_class: str, direction: str, strength: str) -> str:
    verb = "more" if direction == "increase" else "less"
    prefix = {"slightly": "a bit ", "moderately": "", "strongly": "a lot "}[strength]
    return f"{prefix}{verb} {_CLASS_LABEL[goal_class]}"


def _absolute_command_text(goal_class: str, level: str) -> str:
    return f"{level} opacity for {_CLASS_LABEL[goal_class]}"


def _brightness_command_text(goal_class: str, direction: str) -> str:
    verb = "brighten" if direction == "increase" else "darken"
    return f"{verb} {_CLASS_LABEL[goal_class]}"


def _show_only_command_text(shown) -> str:
    return "show only " + " and ".join(_CLASS_LABEL[c] for c in shown)


def _goal_relative(command: dict, volume) -> dict:
    goal_class = command["target"]
    _check_class_supported(goal_class, volume)
    sign = 1.0 if command["direction"] == "increase" else -1.0
    targets = {goal_class: {"vis": sign * VISIBILITY_STRENGTH[command["strength"]]}}
    text = _relative_command_text(goal_class, command["direction"], command["strength"])
    return {"kind": "relative", "text": text, "targets": targets, "goal": goal_vector(targets)}


def _goal_absolute(command: dict, model, start_features: dict, volume) -> dict:
    goal_class = command["target"]
    _check_class_supported(goal_class, volume)
    start_vis = start_features["vis"][goal_class]
    delta = _absolute_target_delta(model, goal_class, command["level"], start_vis)
    targets = {goal_class: {"vis": delta}}
    text = _absolute_command_text(goal_class, command["level"])
    return {"kind": "absolute", "text": text, "targets": targets, "goal": goal_vector(targets)}


def _goal_brightness(command: dict, volume) -> dict:
    goal_class = command["target"]
    _check_class_supported(goal_class, volume)
    sign = 1.0 if command["direction"] == "increase" else -1.0
    targets = {goal_class: {"bright": sign * BRIGHTNESS_STRENGTH[command["strength"]]}}
    text = _brightness_command_text(goal_class, command["direction"])
    return {"kind": "brightness", "text": text, "targets": targets, "goal": goal_vector(targets)}


def _goal_show_only(command: dict, volume) -> dict:
    shown = command["target"] if isinstance(command["target"], list) else [command["target"]]
    for goal_class in shown:
        _check_class_supported(goal_class, volume)
    classes = goal_classes_for_volume(volume) if volume is not None else list(GOAL_CLASSES)
    targets = {goal_class: {"vis": HIDE_STRENGTH} for goal_class in shown}
    for goal_class in classes:
        if goal_class not in shown:
            targets[goal_class] = {"vis": -HIDE_STRENGTH}
    text = _show_only_command_text(shown)
    return {"kind": "show_only", "text": text, "targets": targets, "goal": goal_vector(targets)}


def goal_from_command(command: dict, model, start_features: dict, volume: str = None) -> dict:
    """Turn a parsed command (from `commands.parse_command_rule`/`_llm`) into
    the same {"kind", "text", "targets", "goal"} shape `sample_instruction`
    produces, so downstream code cannot tell a typed instruction from a
    sampled one.

    Raises ValueError for commands that are not goals (width, centre, camera,
    reset -- those are applied exactly by `commands.apply_command`), and for
    goals the volume cannot support (vessels on a plain scan, a class the
    scan does not contain).
    """
    if "camera" in command:
        raise ValueError(_NOT_A_GOAL.format(what="a camera command"))

    if "compound" in command:
        sub_goals = [goal_from_command(sub, model, start_features, volume)
                     for sub in command["compound"]]
        targets = {}
        for sub_goal in sub_goals:
            targets.update(sub_goal["targets"])
        text = ", ".join(sub_goal["text"] for sub_goal in sub_goals)
        return {"kind": "compound", "text": text, "targets": targets, "goal": goal_vector(targets)}

    direction = command.get("direction")
    attribute = command.get("attribute")

    if direction == "reset":
        raise ValueError(_NOT_A_GOAL.format(what="reset"))

    if direction == "show_only":
        return _goal_show_only(command, volume)

    if attribute == "opacity" and direction in ("increase", "decrease"):
        return _goal_relative(command, volume)

    if attribute == "opacity" and direction == "set":
        return _goal_absolute(command, model, start_features, volume)

    if attribute == "brightness" and direction in ("increase", "decrease"):
        return _goal_brightness(command, volume)

    raise ValueError(_NOT_A_GOAL.format(what=f"{command!r}"))
