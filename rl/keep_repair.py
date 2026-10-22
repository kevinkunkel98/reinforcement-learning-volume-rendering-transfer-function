"""Inference-time keep repair: undo the policy's side effects on classes the
instruction never mentioned, where undoing them costs (almost) nothing.

`goals.distance` lets every unmentioned class drift up to KEEP_TOLERANCE for
free, measured from *that command's* start. One command at a time that is
harmless; chained in the viewer, each command starting from the previous
answer, the free drift adds up (2026-10-22: after 7 commands the heart's
brightness had fallen 0.36 and rendered black, vessels +0.46 and the
unlabelled shell +0.37 in log visibility, none of them ever named).

Peak <-> class: one peak per canonical goal class, at index
`transfer.ANATOMICAL_PEAK_INDEX[c]`, i.e. peak i belongs to
`anatomy.CANONICAL_PEAK_ORDER[i]` (lungs, soft, liver, kidneys, spleen, heart,
vessels, skeleton). Centres are fixed (neither policy nor search moves them),
so `commands._find_or_create_peak`'s nearest-centre lookup lands on the same
index for every anatomical class. In label-scoped rendering a voxel labelled c
is shaded by peak c alone; unlabelled voxels by the skeleton, lungs and soft
peaks (`transfer.UNLABELLED_PEAKS`). Those three are still just their class's
peak here: reverting an unmentioned one also moves the unlabelled shell back
toward the start, which is the point, and when the shell matters to the goal
(an occluder, or "other" is charged) the attainment check below keeps it.
"""
import numpy as np

import goals
import transfer

PEAK_CLASS = {index: goal_class for goal_class, index in transfer.ANATOMICAL_PEAK_INDEX.items()}
P = transfer.PARAMS_PER_PEAK
DEFAULT_EPS = 0.01


def mentioned_classes(goal) -> set:
    """Goal classes the goal vector mentions, for visibility (m) or
    brightness (n). Task v2/v3's occluders are marked mentioned with weight 0
    (`goals.occlusion_aware`), so they count as mentioned here too."""
    goal = np.asarray(goal, dtype=np.float64)
    n = len(goals.GOAL_CLASSES)
    return {c for i, c in enumerate(goals.GOAL_CLASSES) if goal[n + i] or goal[3 * n + i]}


def keep_repair(model, goal, start_params, params, layers=None, weights=None, eps=DEFAULT_EPS,
                start_agg=None):
    """Revert unmentioned peaks of `params` to `start_params` where that
    costs at most `eps` attainment. Returns (params, n_evals).

    Every peak whose class the goal does not mention (`mentioned_classes`)
    and whose 6 values differ from the start is a candidate, tried once, the
    largest change (L2 over its 6 values) first. A revert stays if the
    attainment -- `goals.attainment(goal, start_agg, ., weights)`, the same
    score the policy and search are measured with -- is at least the
    proposal's minus `eps`. The bar is the proposal's, not the previous
    revert's, so the total loss is bounded by `eps`, not `eps` per revert.

    `start_agg` is the aggregate at `start_params`; given, it is reused,
    otherwise measured. `n_evals` counts the visibility evaluations this call
    made: the start (unless given), the proposal, one per candidate. A goal
    already met at the start (zero start distance, attainment undefined)
    returns `params` unchanged."""
    start_params = np.asarray(start_params, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64).copy()
    kwargs = {} if weights is None else {"weights": weights}
    n_evals = 0

    def measure(p):
        nonlocal n_evals
        n_evals += 1
        return goals.aggregate(goals.features(model, p, layers), layers)

    if start_agg is None:
        start_agg = measure(start_params)
    if goals.distance(goal, start_agg, start_agg, **kwargs) <= 0.0:
        return params, n_evals

    def score(p):
        return goals.attainment(goal, start_agg, measure(p), **kwargs)

    mentioned = mentioned_classes(goal)
    changes = []
    for index, goal_class in PEAK_CLASS.items():
        if goal_class in mentioned:
            continue
        span = slice(index * P, (index + 1) * P)
        size = float(np.linalg.norm(params[span] - start_params[span]))
        if size > 0.0:
            changes.append((size, index, span))
    bar = score(params) - eps
    for _size, _index, span in sorted(changes, key=lambda item: (-item[0], item[1])):
        candidate = params.copy()
        candidate[span] = start_params[span]
        if score(candidate) >= bar:
            params = candidate
    return params, n_evals
