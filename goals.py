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


TASK_VERSIONS = ("v1", "v2")


def starting_params(task_version: str = "v1") -> np.ndarray:
    """RL v2's starting transfer function: one peak per goal class.

    The same layout `visibility.solo_max` probes at, so "what the instruction
    asks for" and "what this volume can reach" are measured in one geometry.
    Task v2 starts with the organs visible (`transfer.VISIBLE_ORGAN_HEIGHTS`)."""
    if task_version not in TASK_VERSIONS:
        raise ValueError(f"unknown task version {task_version!r}")
    return transfer.anatomical_params(transfer.VISIBLE_ORGAN_HEIGHTS if task_version == "v2" else None)


def features(model, params, active_layers=None) -> dict:
    """Evaluate model features with optional active anatomy layers."""
    if active_layers is None:
        return model.features(params)
    try:
        return model.features(params, active_layers)
    except TypeError:
        return model.features(params)


def _goal_sources(goal_class: str, source_vis: dict) -> tuple:
    """Measured-class keys backing one goal class, in `source_vis`. Read old
    synthetic feature fixtures while canonical models use only registry
    names: legacy `organs`+`muscle` are read as generic `soft`, never
    promoted, when the canonical `soft` key itself is absent."""
    if goal_class == "soft" and "soft" not in source_vis:
        return ("organs", "muscle")
    return MEASURED_FOR_GOAL[goal_class]


def _aggregate_vis(source_vis: dict) -> dict:
    """Measured-class vis -> goal-class vis. Shared by `aggregate` (which
    also needs `bright`/`coverage`) and `aggregate_vis` (which doesn't)."""
    vis = {goal_class: sum(source_vis.get(m, 0.0) for m in _goal_sources(goal_class, source_vis))
           for goal_class in GOAL_CLASSES}
    # Not a goal class -- no instruction ever names it -- but distance() still
    # needs it to tell "hidden" apart from "hidden behind unlabeled material".
    vis["other"] = source_vis.get("other", 0.0)
    return vis


def aggregate(features: dict, active_layers=None) -> dict:
    """Measured classes -> goal classes.

    {"vis": {goal class: float}, "bright": {goal class: float}, "coverage": float}
    Each canonical goal class maps to its matching measured class. Legacy raw
    feature dictionaries using `organs` and `muscle` are read as generic
    `soft`; canonical models use the shared anatomy registry directly.
    """
    source_vis = features.get("effective_vis", features["vis"]) if active_layers is not None else features["vis"]
    source_bright = (features.get("effective_bright", features["bright"])
                     if active_layers is not None else features["bright"])
    vis = _aggregate_vis(source_vis)
    bright = {}
    for goal_class in GOAL_CLASSES:
        sources = _goal_sources(goal_class, source_vis)
        total_vis = vis[goal_class]
        if total_vis > 0.0:
            bright[goal_class] = (sum(source_vis.get(m, 0.0) * source_bright.get(m, 0.0)
                                   for m in sources) / total_vis)
        else:
            bright[goal_class] = 0.0
    coverage = (features.get("effective_coverage", features["coverage"])
                if active_layers is not None else features["coverage"])
    return {"vis": vis, "bright": bright, "coverage": coverage}


def aggregate_vis(vis: dict) -> dict:
    """Like `aggregate`, but for a vis-only measured-class dict such as
    `visibility.VisibilityModel.vis_only`'s output: no `bright`/`coverage`,
    for callers that never read them (the hindsight visibility search loop
    in `rl.oneshot_env`)."""
    return _aggregate_vis(vis)


def vis(model, params, active_layers=None) -> dict:
    """Per-class visibility only, via the model's cheap `vis_only` when it
    has one (the real `VisibilityModel`); a model without it (test stubs)
    falls back to `features()`, exactly mirroring `aggregate`'s own
    plain-vs-effective selection so the fallback can never disagree with
    what `aggregate(features(...))` would have picked."""
    if not hasattr(model, "vis_only"):
        raw = features(model, params, active_layers)
        return raw.get("effective_vis", raw["vis"]) if active_layers is not None else raw["vis"]
    if active_layers is None:
        return model.vis_only(params)
    try:
        return model.vis_only(params, active_layers)
    except TypeError:
        return model.vis_only(params)


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


def vis_target_solvable(change: float, start_vis: float) -> bool:
    """Whether a requested log10 visibility `change` can be met at all from
    `start_vis`. A decrease is bounded by the EPSILON floor (vis cannot go
    below 0); an increase of a class that is practically invisible (vis <
    EPSILON, typically an organ behind soft tissue) needs its occluders
    removed, which the keep terms charge -- even a 200-evaluation search
    rarely manages it (docs/experiments/2026-10-15-relative-diagnosis.md)."""
    if change < 0:
        return math.log10(start_vis + EPSILON) - math.log10(EPSILON) >= -change
    return start_vis >= EPSILON


def instruction_solvable(instruction: dict, start: dict):
    """For relative/compound instructions, whether every visibility target is
    `vis_target_solvable` from the aggregated `start`; None for other kinds."""
    if instruction.get("kind") not in ("relative", "compound"):
        return None
    return all(vis_target_solvable(target["vis"], start["vis"].get(goal_class, 0.0))
               for goal_class, target in instruction["targets"].items() if "vis" in target)


def distance(goal: np.ndarray, start: dict, current: dict, weights: dict = None) -> float:
    """How far a state is from the goal.

    sum m*w*|c - d| + KAPPA*sum n*w*|b - e|
      + LAMBDA_KEEP*sum (1-m)*max(0, |c| - KEEP_TOLERANCE)
      + LAMBDA_KEEP*KAPPA*sum (1-n)*max(0, |b| - KEEP_TOLERANCE)

    Moving one peak inevitably shifts what occludes what elsewhere in the
    volume, so an unmentioned class always drifts a little even when the
    instruction is followed well; a keep term that punishes any drift at
    all makes leaving the transfer function untouched score better than
    acting on the instruction. KEEP_TOLERANCE gives that small, unavoidable
    side effect for free and only penalises drift beyond it.

    `weights` (goal_class -> multiplier, default None = every class 1.0) only
    ever scales a *mentioned* class's own term -- it has no effect on a class
    the goal doesn't mention, whose treatment is entirely the keep-tolerance
    branch above. This is how `show_only` gives its shown-class group and
    hidden-class group equal total weight despite one side having far fewer
    classes than the other (see `show_only_weights`).
    """
    c, b = progress(start, current)
    n_classes = len(GOAL_CLASSES)
    d, m, e, n = goal[0:n_classes], goal[n_classes:2 * n_classes], \
        goal[2 * n_classes:3 * n_classes], goal[3 * n_classes:4 * n_classes]
    total = 0.0
    for i, goal_class in enumerate(GOAL_CLASSES):
        w = 1.0 if weights is None else weights.get(goal_class, 1.0)
        total += m[i] * w * abs(c[goal_class] - d[i]) + KAPPA * n[i] * w * abs(b[goal_class] - e[i])
        total += LAMBDA_KEEP * (1.0 - m[i]) * max(0.0, abs(c[goal_class]) - KEEP_TOLERANCE)
        total += LAMBDA_KEEP * KAPPA * (1.0 - n[i]) * max(0.0, abs(b[goal_class]) - KEEP_TOLERANCE)
    # "other" (unlabeled tissue) can never be named by an instruction, so it is
    # always the unmentioned case -- without this, a transfer function can
    # satisfy "show only X" by rendering an opaque wall of unclassified
    # material instead of X, since "other" is not a goal class and is never
    # charged by the mentioned-class terms. Task v2's occlusion-aware goals
    # (`occlusion_aware`) can exempt it with weights["other"] = 0, when that
    # unlabelled shell is what hides a target.
    other_weight = 1.0 if weights is None else weights.get("other", 1.0)
    total += LAMBDA_KEEP * other_weight * max(0.0, abs(c["other"]) - KEEP_TOLERANCE)
    return float(total)


def scoring(goal, start_features, final_features, active_layers=None, weights: dict = None) -> float:
    """Score final model features against start features using effective layers."""
    start = aggregate(start_features, active_layers)
    final = aggregate(final_features, active_layers)
    kwargs = {} if weights is None else {"weights": weights}
    return attainment(goal, start, final, **kwargs)


def attainment(goal, start, final, active_layers=None, weights: dict = None) -> float:
    """1 - distance(final) / distance(start); 1 = goal reached, < 0 = worse."""
    # Omitting the `weights` kwarg entirely when it's None (rather than always
    # passing `weights=None`) matters: several tests monkeypatch `distance`
    # with a fixed-arity stub lacking a `weights` parameter, and this function
    # calling it with an unconditional `weights=` keyword would break every
    # one of them even though no actual weighting is happening.
    kwargs = {} if weights is None else {"weights": weights}
    return 1.0 - distance(goal, start, final, **kwargs) / distance(goal, start, start, **kwargs)


def is_useless(features: dict, active_layers=None) -> bool:
    """Nothing drawn, or an opaque wall hiding everything."""
    vis = features.get("effective_vis", features["vis"]) if active_layers is not None else features["vis"]
    coverage = (features.get("effective_coverage", features["coverage"])
                if active_layers is not None else features["coverage"])
    return (coverage < 0.01 or sum(vis.values()) < 0.001)


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


def class_ceiling(model, goal_class: str, active_layers=None) -> float:
    """The most of `goal_class` any single-peak transfer function can show,
    summed over the measured classes in `MEASURED_FOR_GOAL`; canonical classes
    are one-to-one, including generic `soft`."""
    ceiling = sum(model.solo_max(m) for m in MEASURED_FOR_GOAL[goal_class])
    if active_layers is not None:
        ceiling *= active_layers.get(goal_class, {}).get("opacity", 1.0)
    return ceiling


def reachable_goal_classes(name: str, model, active_layers=None) -> list:
    """`goal_classes_for_volume` narrowed to the classes a render can actually
    show on this volume (ceiling >= `VISIBLE_CEILING`).

    Instruction sampling uses this rather than label presence: a class can be
    labelled in the volume and still be impossible to see, which produces
    instructions no method can satisfy and no rater can judge -- they pollute
    preference collection and drag every held-out score toward zero equally.
    """
    return [c for c in goal_classes_for_volume(name)
            if class_ceiling(model, c, active_layers) >= VISIBLE_CEILING]


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


def _absolute_target_delta(model, goal_class: str, level: str, start_vis: float, active_layers=None) -> float:
    """The vis change an absolute-level instruction requests: `level`'s share
    of solo_max (aggregated for the goal class) as a log10 change from
    `start_vis`, so "high skeleton" means the same thing on every volume."""
    solo = class_ceiling(model, goal_class, active_layers)
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


def _choose_class(classes: list, rng, preferred=None) -> str:
    if preferred is not None and preferred in classes:
        return preferred
    return str(rng.choice(classes))


def _sample_relative(classes: list, rng, preferred=None) -> tuple:
    goal_class = _choose_class(classes, rng, preferred)
    direction = rng.choice(("increase", "decrease"))
    strength = rng.choice(list(VISIBILITY_STRENGTH))
    sign = 1.0 if direction == "increase" else -1.0
    targets = {goal_class: {"vis": sign * VISIBILITY_STRENGTH[strength]}}
    return targets, _relative_text(goal_class, direction, strength, rng)


def _sample_compound(classes: list, rng, preferred=None) -> tuple:
    n = min(2, len(classes))
    chosen = ([preferred] if preferred in classes else [])
    remaining = [c for c in classes if c not in chosen]
    if len(chosen) < n:
        chosen.extend(list(rng.choice(remaining, size=n - len(chosen), replace=False)))
    targets, phrases = {}, []
    for goal_class in chosen:
        direction = rng.choice(("increase", "decrease"))
        strength = rng.choice(list(VISIBILITY_STRENGTH))
        sign = 1.0 if direction == "increase" else -1.0
        targets[goal_class] = {"vis": sign * VISIBILITY_STRENGTH[strength]}
        phrases.append(_relative_text(goal_class, direction, strength, rng))
    return targets, ", ".join(phrases)


SOLVABLE_DRAWS = 10   # (direction, strength) draws before giving up on a solvable target


def _draw_solvable_target(candidates: list, start_vis: dict, rng, preferred=None):
    """(class, signed log10 change) for a relative target that
    `vis_target_solvable` from `start_vis`, or None if SOLVABLE_DRAWS
    (direction, strength) draws find no class it is solvable for. Direction
    and strength are drawn first and the class is chosen among the classes
    that allow them, so the direction/strength mix stays as declared."""
    for _ in range(SOLVABLE_DRAWS):
        sign = 1.0 if rng.choice(("increase", "decrease")) == "increase" else -1.0
        strength = rng.choice(list(VISIBILITY_STRENGTH))
        change = sign * VISIBILITY_STRENGTH[strength]
        eligible = [c for c in candidates if vis_target_solvable(change, start_vis.get(c, 0.0))]
        if eligible:
            return _choose_class(eligible, rng, preferred), change, strength
    return None


def _relative_phrase(goal_class: str, change: float, strength: str, rng) -> str:
    return _relative_text(goal_class, "increase" if change > 0 else "decrease", strength, rng)


def _sample_relative_solvable(classes: list, start_vis: dict, rng, preferred=None):
    """Like `_sample_relative`, but only targets that can be met from the
    start state (docs/experiments/2026-10-15-relative-diagnosis.md); None
    when no class allows any relative change."""
    drawn = _draw_solvable_target(classes, start_vis, rng, preferred)
    if drawn is None:
        return None
    goal_class, change, strength = drawn
    return {goal_class: {"vis": change}}, _relative_phrase(goal_class, change, strength, rng)


def _sample_compound_solvable(classes: list, start_vis: dict, rng, preferred=None):
    """Like `_sample_compound`, but both targets solvable; None if two
    distinct classes with solvable targets can't be found."""
    first = _draw_solvable_target(classes, start_vis, rng, preferred)
    if first is None:
        return None
    second = _draw_solvable_target([c for c in classes if c != first[0]], start_vis, rng)
    if second is None:
        return None
    targets, phrases = {}, []
    for goal_class, change, strength in (first, second):
        targets[goal_class] = {"vis": change}
        phrases.append(_relative_phrase(goal_class, change, strength, rng))
    return targets, ", ".join(phrases)


def show_only_weights(shown: set, classes) -> dict:
    """Equal total weight for the shown group and the hidden group,
    regardless of how many classes are in each -- 1/n_shown per shown class,
    1/n_hidden per hidden class. Without this, `distance`'s flat per-class
    weighting lets a show_only goal's 6-7 hidden classes outweigh its 1-2
    shown ones purely by having more terms.

    Either group can be empty: `classes` is `reachable_goal_classes`/
    `goal_classes_for_volume`, narrowed by `active_layers` or a volume's own
    label coverage, and can shrink to as few as 1-2 classes -- in which case
    naming all of them as "shown" leaves nothing to hide (`_sample_show_only`
    and `_goal_show_only` both produce exactly this `targets` shape when
    `shown == classes`: every class at +HIDE_STRENGTH, none at -HIDE_STRENGTH).
    An empty group's weights are simply omitted rather than divided by zero.
    """
    shown = set(shown)
    hidden = [c for c in classes if c not in shown]
    weights = {}
    if shown:
        weights.update({c: 1.0 / len(shown) for c in shown})
    if hidden:
        weights.update({c: 1.0 / len(hidden) for c in hidden})
    return weights


def _sample_show_only(classes: list, rng, preferred=None) -> tuple:
    n = min(int(rng.integers(1, 3)), len(classes))
    shown = ([preferred] if preferred in classes else [])
    remaining = [c for c in classes if c not in shown]
    if len(shown) < n:
        shown.extend(list(rng.choice(remaining, size=n - len(shown), replace=False)))
    targets = {goal_class: {"vis": HIDE_STRENGTH} for goal_class in shown}
    for goal_class in classes:
        if goal_class not in shown:
            targets[goal_class] = {"vis": -HIDE_STRENGTH}
    words = [_class_word(goal_class, rng) for goal_class in shown]
    text = "show only the " + " and the ".join(words)
    return targets, text, show_only_weights(set(shown), classes)


def _sample_absolute(classes: list, model, start_features: dict, rng, preferred=None) -> tuple:
    goal_class = _choose_class(classes, rng, preferred)
    level = rng.choice(list(ABSOLUTE_LEVEL))
    delta = _absolute_target_delta(model, goal_class, level, start_features["vis"][goal_class])
    targets = {goal_class: {"vis": delta}}
    return targets, _absolute_text(goal_class, level, rng)


def _sample_brightness(classes: list, rng, preferred=None) -> tuple:
    goal_class = _choose_class(classes, rng, preferred)
    direction = rng.choice(("increase", "decrease"))
    strength = rng.choice(list(BRIGHTNESS_STRENGTH))
    sign = 1.0 if direction == "increase" else -1.0
    targets = {goal_class: {"bright": sign * BRIGHTNESS_STRENGTH[strength]}}
    return targets, _brightness_text(goal_class, direction, rng)


def sample_instruction(name, model, start_features, rng, active_layers=None,
                       class_counts=None, balance_classes=False, solvable_only=False) -> dict:
    """One instruction: {"kind", "text", "targets", "goal"}, plus "weights"
    for show_only instructions (see `show_only_weights`).

    `targets` is the goal_vector input; `text` is the spoken form. Absolute
    levels use ABSOLUTE_LEVEL x model.solo_max aggregated for the goal class,
    turned into a requested change from the start state, so "high skeleton"
    means the same thing on every volume.
    """
    classes = (reachable_goal_classes(name, model, active_layers)
               if active_layers is not None else reachable_goal_classes(name, model))
    preferred = None
    if balance_classes and class_counts:
        minimum = min(class_counts.get(goal_class, 0) for goal_class in classes)
        least_seen = [goal_class for goal_class in classes
                      if class_counts.get(goal_class, 0) == minimum]
        preferred = str(rng.choice(least_seen))
    kind = _sample_kind(rng)
    weights = None
    drawn = None
    if solvable_only and kind in ("relative", "compound"):
        # Only targets the start state allows; when there are none, an
        # absolute request stays meaningful ("make X high") where "a bit
        # more X" for an invisible X is not.
        sampler = _sample_relative_solvable if kind == "relative" else _sample_compound_solvable
        drawn = sampler(classes, start_features["vis"], rng, preferred)
        if drawn is None:
            kind = "absolute"
    if drawn is not None:
        targets, text = drawn
    elif kind == "relative":
        targets, text = _sample_relative(classes, rng, preferred)
    elif kind == "compound":
        targets, text = _sample_compound(classes, rng, preferred)
    elif kind == "show_only":
        targets, text, weights = _sample_show_only(classes, rng, preferred)
    elif kind == "absolute":
        targets, text = _sample_absolute(classes, model, start_features, rng, preferred)
    else:
        targets, text = _sample_brightness(classes, rng, preferred)
    instruction = {"kind": kind, "text": text, "targets": targets, "goal": goal_vector(targets)}
    if weights is not None:
        instruction["weights"] = weights
    return instruction


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


def _goal_absolute(command: dict, model, start_features: dict, volume, active_layers=None) -> dict:
    goal_class = command["target"]
    _check_class_supported(goal_class, volume)
    start_vis = start_features["vis"][goal_class]
    delta = _absolute_target_delta(model, goal_class, command["level"], start_vis, active_layers)
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
    return {"kind": "show_only", "text": text, "targets": targets, "goal": goal_vector(targets),
            "weights": show_only_weights(set(shown), classes)}


def goal_from_command(command: dict, model, start_features: dict, volume: str = None,
                      active_layers=None) -> dict:
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
    goal = _goal_from_command(command, model, start_features, volume, active_layers)
    if volume is not None:
        for goal_class, target in goal["targets"].items():
            if target.get("vis", 0.0) <= 0.0:
                continue
            ceiling = (class_ceiling(model, goal_class, active_layers)
                       if active_layers is not None else class_ceiling(model, goal_class))
            if ceiling < VISIBLE_CEILING:
                raise ValueError(
                    f"{goal_class!r} cannot be shown on volume {volume!r} -- the most any "
                    f"transfer function reaches is {ceiling:.5f} of the image "
                    f"(below {VISIBLE_CEILING}), so the change would be invisible")
    return goal


def _goal_from_command(command: dict, model, start_features: dict, volume, active_layers=None) -> dict:
    """`goal_from_command` without the reachability check, which the public
    entry point applies once to the finished targets so a compound command is
    checked as a whole."""
    if "camera" in command:
        raise ValueError(_NOT_A_GOAL.format(what="a camera command"))

    if "compound" in command:
        sub_goals = [_goal_from_command(sub, model, start_features, volume, active_layers)
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
        return _goal_absolute(command, model, start_features, volume, active_layers)

    if attribute == "brightness" and direction in ("increase", "decrease"):
        return _goal_brightness(command, volume)

    raise ValueError(_NOT_A_GOAL.format(what=f"{command!r}"))


# --- task v2: making hidden-organ instructions solvable ----------------------
#
# Under the v1 task about half of relative/compound val instructions cannot be
# improved by any method (docs/experiments/2026-10-17-task-v2.md): they ask to
# lower a class that is already invisible, to raise a class hidden behind other
# tissue (fading the occluder costs more keep-drift than the goal gains), or to
# move a class further than it can physically go. Task v2 keeps the metric and
# changes what an instruction asks for: well-posed sampling (solvable_only), the
# two transforms below, and a start transfer function with the organs visible
# (`transfer.VISIBLE_ORGAN_HEIGHTS`). Each is expressed through the goal vector
# and `weights`, so every caller of `distance` scores v2 without change.

OCCLUDER_GAIN = 0.15     # log10 visibility a target must gain for a class to count as its occluder


def _height_index(goal_class: str) -> int:
    return transfer.ANATOMICAL_PEAK_INDEX[goal_class] * transfer.PARAMS_PER_PEAK + 2


def find_occluders(model, params, instruction, active_layers=None) -> set:
    """Classes (and "other", the unlabelled tissue) that hide a target the
    instruction asks to make more visible.

    A class k is an occluder when removing it alone lifts a target's log10
    visibility by more than OCCLUDER_GAIN, or when -- with every unmentioned
    class removed -- putting k back drops a target by that much (joint
    occlusion: a heart behind both soft tissue and the liver). "other" is not a
    peak of its own (the skeleton/lungs/soft peaks shade it), so it counts when
    clearing the unmentioned classes both removes it and lifts a target."""
    targets = [c for c, t in instruction["targets"].items() if t.get("vis", 0.0) > 0]
    if not targets:
        return set()
    log = lambda vis, c: math.log10(vis.get(c, 0.0) + EPSILON)
    vis_of = lambda p: aggregate(model.features(p) if active_layers is None
                                 else model.features(p, active_layers), active_layers)["vis"]
    others = [k for k in GOAL_CLASSES if k not in instruction["targets"]]
    base = vis_of(params)
    found = set()
    for k in others:
        p = np.array(params, dtype=np.float64)
        p[_height_index(k)] = -1.0
        if any(log(vis_of(p), c) - log(base, c) > OCCLUDER_GAIN for c in targets):
            found.add(k)
    cleared = np.array(params, dtype=np.float64)
    for k in others:
        cleared[_height_index(k)] = -1.0
    clear_vis = vis_of(cleared)
    if (log(base, "other") - log(clear_vis, "other") > OCCLUDER_GAIN
            and any(log(clear_vis, c) - log(base, c) > OCCLUDER_GAIN for c in targets)):
        found.add("other")
    for k in others:
        p = cleared.copy()
        p[_height_index(k)] = params[_height_index(k)]
        if any(log(clear_vis, c) - log(vis_of(p), c) > OCCLUDER_GAIN for c in targets):
            found.add(k)
    return found


def occlusion_aware(instruction: dict, occluders: set) -> dict:
    """`instruction` with its occluders exempt from the keep terms: each occluding
    class becomes "mentioned with weight 0" for visibility and brightness, and an
    occluding "other" gets weights["other"] = 0. Revealing a hidden class by
    fading what is in front of it is what was asked for, not drift."""
    if not occluders:
        return instruction
    n = len(GOAL_CLASSES)
    goal = np.array(instruction["goal"], dtype=np.float64)
    weights = dict(instruction.get("weights") or {c: 1.0 for c in GOAL_CLASSES})
    for i, goal_class in enumerate(GOAL_CLASSES):
        if goal_class in occluders:
            goal[i], goal[n + i] = 0.0, 1.0
            goal[2 * n + i], goal[3 * n + i] = 0.0, 1.0
            weights[goal_class] = 0.0
    if "other" in occluders:
        weights["other"] = 0.0
    return {**instruction, "goal": goal, "weights": weights, "occluders": sorted(occluders)}


def reachable_targets(instruction: dict, model, start: dict, active_layers=None) -> dict:
    """`instruction` with every visibility target clipped to what its class can
    reach from `start`: not below the EPSILON floor ("a lot less lungs" when the
    lungs are already invisible) and not above the class's single-peak ceiling
    (show_only's +1 for a class that cannot get that bright). Occluder entries
    (weight 0) are left alone."""
    n = len(GOAL_CLASSES)
    goal = np.array(instruction["goal"], dtype=np.float64)
    floor = math.log10(EPSILON)
    for i, goal_class in enumerate(GOAL_CLASSES):
        if not goal[n + i]:
            continue
        start_vis = start["vis"].get(goal_class, 0.0)
        current = math.log10(start_vis + EPSILON)
        ceiling = math.log10(max(class_ceiling(model, goal_class, active_layers), start_vis) + EPSILON)
        goal[i] = min(max(goal[i], floor - current), ceiling - current)
    return {**instruction, "goal": goal}
