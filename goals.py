"""Turn an instruction into a measurable target.

Visibility (`visibility.for_volume(...).features(...)`) is measured over the
eight canonical anatomy classes. Promoted organs (heart, liver, kidneys and
spleen) each have their own measured class; generic `soft` is separate and is
not an aggregate of promoted organs. Vessels only stand apart from soft tissue
when the scan carries contrast, so vessels are only ever a goal on a contrast
volume (see `goal_classes_for_volume`).

A goal is a 32-value vector: four dynamic blocks, each with one value per
`GOAL_CLASSES` entry: requested visibility change, visibility-mentioned flag,
requested brightness change, and brightness-mentioned flag. An instruction
that never mentions a class should not be scored on what happens to it.
Visibility changes are scored in log10 units because visibility spans three
orders of magnitude (0.0003 to 0.5) -- a fixed absolute change means very
different things at each end, but a fixed factor does not.
Brightness changes are scored in plain units.
"""
import math

import numpy as np

import totalseg
import transfer
from anatomy import CANONICAL_CLASSES

GOAL_CLASSES = CANONICAL_CLASSES
MEASURED_FOR_GOAL = {goal_class: (goal_class,) for goal_class in GOAL_CLASSES}

# RL v2 peak order, by the class each peak is seeded for. Defined in transfer,
# because visibility.solo_max must probe at these same centres and cannot
# import goals (goals imports visibility). Keeping a second copy here is what
# let solo_max drift away from the canonical peak layout and report every
# volume's lungs as unreachable.
PEAK_CENTRES_HU = transfer.ANATOMICAL_CENTRES_HU
PEAK_INDEX = transfer.ANATOMICAL_PEAK_INDEX
PEAK_WIDTHS_HU = transfer.ANATOMICAL_WIDTHS_HU
PEAK_HEIGHTS = transfer.ANATOMICAL_HEIGHTS
PEAK_COLOURS = transfer.ANATOMICAL_COLOURS

EPSILON = 1e-3           # floor inside log10, so "invisible" is finite
KAPPA = 1.5              # brightness weight: 0.2 brightness ~= 0.3 log10 visibility
LAMBDA_KEEP = 0.3        # weight of "leave the unmentioned classes alone"
KEEP_TOLERANCE = 0.15    # unmentioned-class drift this small is free (see distance)
VISIBILITY_STRENGTH = {"slightly": 0.15, "moderately": 0.3, "strongly": 0.6}   # log10 units
BRIGHTNESS_STRENGTH = {"slightly": 0.1, "moderately": 0.2, "strongly": 0.4}
ABSOLUTE_LEVEL = {"low": 0.1, "medium": 0.4, "high": 0.8}                      # share of solo_max


def starting_params() -> np.ndarray:
    """RL v2's starting transfer function: one peak per goal class.

    The same layout `visibility.solo_max` probes at, so "what the instruction
    asks for" and "what this volume can reach" are measured in one geometry."""
    return transfer.anatomical_params()


def aggregate(features: dict) -> dict:
    """Measured classes -> goal classes.

    {"vis": {goal class: float}, "bright": {goal class: float}, "coverage": float}
    Each canonical goal class maps to its matching measured class. Legacy raw
    feature dictionaries using `organs` and `muscle` are read as generic
    `soft`; canonical models use the shared anatomy registry directly.
    """
    vis, bright = {}, {}
    for goal_class in GOAL_CLASSES:
        measured = MEASURED_FOR_GOAL[goal_class]
        # Read old synthetic feature fixtures while canonical models use only
        # registry names. Legacy organs/muscle remain soft, never promoted.
        if goal_class == "soft" and "soft" not in features["vis"]:
            sources = ("organs", "muscle")
        else:
            sources = measured
        total_vis = sum(features["vis"].get(m, 0.0) for m in sources)
        vis[goal_class] = total_vis
        if total_vis > 0.0:
            bright[goal_class] = (sum(features["vis"].get(m, 0.0) * features["bright"].get(m, 0.0)
                                   for m in sources) / total_vis)
        else:
            bright[goal_class] = 0.0
    # Not a goal class -- no instruction ever names it -- but distance() still
    # needs it to tell "hidden" apart from "hidden behind unlabeled material".
    vis["other"] = features["vis"].get("other", 0.0)
    return {"vis": vis, "bright": bright, "coverage": features["coverage"]}


def goal_vector(targets: dict) -> np.ndarray:
    """A 32-value goal with four dynamic class-aligned blocks.

    targets maps goal class -> {"vis": float, "bright": float}; a missing key
    means that aspect is not part of the instruction. Layout: d[n], m[n],
    e[n], n[n], where n = len(GOAL_CLASSES), in class order.
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
        c[goal_class] = (math.log10(current["vis"].get(goal_class, 0.0) + EPSILON)
                          - math.log10(start["vis"].get(goal_class, 0.0) + EPSILON))
        b[goal_class] = (current["bright"].get(goal_class, 0.0)
                         - start["bright"].get(goal_class, 0.0))
    c["other"] = (math.log10(current["vis"].get("other", 0.0) + EPSILON)
                  - math.log10(start["vis"].get("other", 0.0) + EPSILON))
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
    # "other" (unlabeled tissue) can never be named by an instruction, so it is
    # always the unmentioned case -- without this, a transfer function can
    # satisfy "show only X" by rendering an opaque wall of unclassified
    # material instead of X, since "other" is not a goal class and is never
    # charged by the mentioned-class terms.
    total += LAMBDA_KEEP * max(0.0, abs(c["other"]) - KEEP_TOLERANCE)
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
    "heart": ("heart",),
    "liver": ("liver",),
    "kidneys": ("kidneys", "kidney"),
    "spleen": ("spleen",),
}


# Goal classes measurable on a volume with no anatomical labelling at all --
# visibility.for_volume's intensity fallback (label_source == "intensity")
# only separates skeleton, lungs and generic soft tissue by Hounsfield value;
# promoted organs and vessels are not populated because intensity alone cannot
# identify them. Callers that need to know *why* (whether this volume actually
# has anatomical labels) should read `visibility.for_volume(name).label_source`;
# this function reports which goals are measurable either way, so policy mode
# and instruction sampling keep working on every calibrated CT.
FALLBACK_GOAL_CLASSES = ("skeleton", "lungs", "soft")


def goal_classes_for_volume(name: str) -> list:
    """Goal classes this volume supports: those with labels present, with
    `vessels` only on contrast scans (elsewhere vessels share intensities with
    soft tissue and no transfer function can single them out).

    Volumes with no TotalSegmentator anatomy labelling (anything that isn't a
    `ts_*` subject) fall back to `FALLBACK_GOAL_CLASSES` -- what the
    intensity-label fallback in `visibility.for_volume` can actually
    measure."""
    try:
        present = set(totalseg.classes_present(name))
    except ValueError:
        return list(FALLBACK_GOAL_CLASSES)
    supported = []
    for goal_class in GOAL_CLASSES:
        if goal_class == "vessels":
            if "vessels" in present and totalseg.is_contrast(name):
                supported.append(goal_class)
        elif any(m in present for m in MEASURED_FOR_GOAL[goal_class]):
            supported.append(goal_class)
    return supported


# A goal class is worth asking about only if a transfer function can actually
# make it visible. `model.solo_max` is that ceiling: the share of the image the
# class reaches under the best single-peak transfer function. Below this, the
# whole reachable range is thinner than a rater can see -- on ts_s1371 the lung
# ceiling is 0.00033, so "a bit more lungs" (x1.4) asks someone to rank a
# change of about 0.01% of the pixels. Measurable, invisible, unanswerable.
#
# 0.005 = half a percent of the image. Surveyed over the 30 selected volumes it
# excludes lungs on 26 of them (median ceiling 0.0015) and vessels on 3, while
# keeping skeleton and soft everywhere (medians 0.121 and 0.073) -- i.e. it
# removes what no rater could judge and nothing else. This dataset is
# abdomen/pelvis-heavy, so most scans hold only clipped lung bases behind an
# opaque body wall; it is a property of the selection, not of the measurement.
VISIBLE_CEILING = 0.005


def class_ceiling(model, goal_class: str) -> float:
    """The most of `goal_class` any single-peak transfer function can show,
    summed over the measured classes in `MEASURED_FOR_GOAL`; canonical classes
    are one-to-one, including generic `soft`."""
    return sum(model.solo_max(m) for m in MEASURED_FOR_GOAL[goal_class])


def reachable_goal_classes(name: str, model) -> list:
    """`goal_classes_for_volume` narrowed to the classes a render can actually
    show on this volume (ceiling >= `VISIBLE_CEILING`).

    Instruction sampling uses this rather than label presence: a class can be
    labelled in the volume and still be impossible to see, which produces
    instructions no method can satisfy and no rater can judge -- they pollute
    preference collection and drag every held-out score toward zero equally.
    """
    return [c for c in goal_classes_for_volume(name) if class_ceiling(model, c) >= VISIBLE_CEILING]


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
    classes = reachable_goal_classes(name, model)
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
    goals the volume cannot support: vessels on a plain scan, a class the scan
    does not contain, and a class the scan contains but cannot show (see
    `reachable_goal_classes` -- asking for lungs on an abdomen scan).
    """
    goal = _goal_from_command(command, model, start_features, volume)
    if volume is not None:
        for goal_class, target in goal["targets"].items():
            if target.get("vis", 0.0) <= 0.0:
                continue
            ceiling = class_ceiling(model, goal_class)
            if ceiling < VISIBLE_CEILING:
                raise ValueError(
                    f"{goal_class!r} cannot be shown on volume {volume!r} -- the most any "
                    f"transfer function reaches is {ceiling:.5f} of the image "
                    f"(below {VISIBLE_CEILING}), so the change would be invisible")
    return goal


def _goal_from_command(command: dict, model, start_features: dict, volume) -> dict:
    """`goal_from_command` without the reachability check, which the public
    entry point applies once to the finished targets so a compound command is
    checked as a whole."""
    if "camera" in command:
        raise ValueError(_NOT_A_GOAL.format(what="a camera command"))

    if "compound" in command:
        sub_goals = [_goal_from_command(sub, model, start_features, volume)
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
