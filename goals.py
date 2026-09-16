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
      + LAMBDA_KEEP*(sum (1-m)*|c| + KAPPA*sum (1-n)*|b|)
    """
    c, b = progress(start, current)
    n_classes = len(GOAL_CLASSES)
    d, m, e, n = goal[0:n_classes], goal[n_classes:2 * n_classes], \
        goal[2 * n_classes:3 * n_classes], goal[3 * n_classes:4 * n_classes]
    total = 0.0
    for i, goal_class in enumerate(GOAL_CLASSES):
        total += m[i] * abs(c[goal_class] - d[i]) + KAPPA * n[i] * abs(b[goal_class] - e[i])
        total += LAMBDA_KEEP * ((1.0 - m[i]) * abs(c[goal_class]) + KAPPA * (1.0 - n[i]) * abs(b[goal_class]))
    return float(total)


def attainment(goal, start, final) -> float:
    """1 - distance(final) / distance(start); 1 = goal reached, < 0 = worse."""
    return 1.0 - distance(goal, start, final) / distance(goal, start, start)


def is_useless(features: dict) -> bool:
    """Nothing drawn, or an opaque wall hiding everything."""
    return (features["coverage"] < 0.01
            or sum(features["vis"].values()) < 0.001)
