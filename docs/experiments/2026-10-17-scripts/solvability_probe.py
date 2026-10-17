"""Which task definition makes hidden-organ instructions solvable?

For the same val episode seeds, three settings:
  base     current start TF, current metric
  visible  start TF with soft tissue dimmed and organs raised, current metric
  occl     current start TF, occlusion-aware metric: an unmentioned class whose
           fading raises a vis-increase target's log10 visibility by more than
           OCCLUDER_GAIN at the start counts as that target's occluder and is
           exempt from the keep terms (visibility and brightness) -- expressed as
           "mentioned with weight 0", so goals.distance, hill_climb and the
           policy observation work unchanged.
Reports, per setting and kind: share solvable (goals.instruction_solvable, for
relative/compound) and the cold hill_climb(BUDGET) attainment as a ceiling proxy.

  solvable relative/compound instructions drawn only where goals.instruction_solvable
           (OneShotEnv(solvable_instructions=True)): no "less X" for an invisible X
  reach    every visibility target clipped to what the class can reach (EPSILON floor,
           class_ceiling) -- e.g. show_only's -1 for a class already at the floor
Settings combine with "+", e.g. visible+occl+solvable+reach.

usage: solvability_probe.py OUT.json SETTING SEED_FROM SEED_TO [BUDGET]
"""
import json, math, os, sys
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
import goals, transfer
from rl.baselines import hill_climb
from rl.oneshot_env import OneShotEnv
import datasets

OCCLUDER_GAIN = 0.15
VISIBLE_HEIGHTS = {"soft": 0.06, "lungs": 0.05, "skeleton": 0.45,
                   "heart": 0.45, "vessels": 0.45, "liver": 0.45, "kidneys": 0.45, "spleen": 0.45}


def class_height_index(goal_class):
    return transfer.ANATOMICAL_PEAK_INDEX[goal_class] * transfer.PARAMS_PER_PEAK + 2


def occluders(model, params, instruction):
    """Unmentioned classes whose removal lifts some vis-increase target by > OCCLUDER_GAIN (log10)."""
    targets = [c for c, t in instruction["targets"].items() if t.get("vis", 0.0) > 0]
    if not targets:
        return set()
    # show_only mentions every goal class, so only "other" can be an occluder there
    # (SHOW_ONLY_OCCL=0 restores the first version, which skipped show_only).
    if instruction["kind"] == "show_only" and os.environ.get("SHOW_ONLY_OCCL", "1") != "1":
        return set()
    log = lambda vis, c: math.log10(vis[c] + goals.EPSILON)
    vis_of = lambda p: goals.aggregate(goals.features(model, p))["vis"]
    others = [k for k in goals.GOAL_CLASSES if k not in instruction["targets"]]
    base = vis_of(params)
    found = set()
    # alone: removing k by itself lifts a target
    for k in others:
        p = params.copy(); p[class_height_index(k)] = -1.0
        if any(log(vis_of(p), c) - log(base, c) > OCCLUDER_GAIN for c in targets):
            found.add(k)
    # jointly: with every other class removed, putting k back drops a target
    cleared = params.copy()
    for k in others:
        cleared[class_height_index(k)] = -1.0
    clear_vis = vis_of(cleared)
    # "other" (unlabelled tissue) is shaded by the skeleton/lungs/soft peaks and is
    # not a goal class: it occludes when clearing the rest removes it and lifts a target
    if (log(base, "other") - log(clear_vis, "other") > OCCLUDER_GAIN
            and any(log(clear_vis, c) - log(base, c) > OCCLUDER_GAIN for c in targets)):
        found.add("other")
    for k in others:
        p = cleared.copy(); p[class_height_index(k)] = params[class_height_index(k)]
        if any(log(clear_vis, c) - log(vis_of(p), c) > OCCLUDER_GAIN for c in targets):
            found.add(k)
    return found


_ORIGINAL_DISTANCE = goals.distance
EXEMPT_OTHER = {"on": False}


def _distance(goal, start, current, weights=None):
    """goals.distance, minus the "other" keep term while EXEMPT_OTHER is on."""
    kw = {} if weights is None else {"weights": weights}
    total = _ORIGINAL_DISTANCE(goal, start, current, **kw)
    if EXEMPT_OTHER["on"]:
        c, _ = goals.progress(start, current)
        total -= goals.LAMBDA_KEEP * max(0.0, abs(c["other"]) - goals.KEEP_TOLERANCE)
    return total


goals.distance = _distance


def occlusion_aware(instruction, occl):
    EXEMPT_OTHER["on"] = "other" in occl
    if not occl:
        return instruction
    n = len(goals.GOAL_CLASSES)
    goal = np.array(instruction["goal"], dtype=np.float64)
    weights = dict(instruction.get("weights") or {c: 1.0 for c in goals.GOAL_CLASSES})
    for i, c in enumerate(goals.GOAL_CLASSES):
        if c in occl:
            goal[n + i] = 1.0; goal[i] = 0.0            # vis: mentioned, weight 0
            goal[3 * n + i] = 1.0; goal[2 * n + i] = 0.0  # brightness: same
            weights[c] = 0.0
    return {**instruction, "goal": goal, "weights": weights, "occluders": sorted(occl)}


def reachable(instruction, model, start_agg):
    """Clip every visibility target to what the class can physically reach from the
    start: not below the EPSILON floor, not above its single-peak ceiling."""
    n = len(goals.GOAL_CLASSES)
    goal = np.array(instruction["goal"], dtype=np.float64)
    for i, c in enumerate(goals.GOAL_CLASSES):
        if not goal[n + i]:
            continue
        start = math.log10(start_agg["vis"][c] + goals.EPSILON)
        floor = math.log10(goals.EPSILON)
        ceiling = math.log10(max(goals.class_ceiling(model, c), start_agg["vis"][c]) + goals.EPSILON)
        goal[i] = min(max(goal[i], floor - start), ceiling - start)
    return {**instruction, "goal": goal}


if __name__ == "__main__":
    out, setting, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    budget = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    flags = set(setting.split("+"))
    if "visible" in flags:
        transfer.ANATOMICAL_HEIGHTS.update(VISIBLE_HEIGHTS)
    env = OneShotEnv(datasets.volumes_for_split("val"), policy_version="oneshot-v9",
                     action_mode="residual", reward_mode="target",
                     solvable_instructions="solvable" in flags)
    rows = []
    for seed in range(lo, hi):
        env.reset(seed=seed)
        model, inst, start = env._model, env._instruction, env._start_params.copy()
        start_agg = env._start_agg
        if "occl" in flags:
            inst = occlusion_aware(inst, occluders(model, start, inst))
        if "reach" in flags:
            inst = reachable(inst, model, start_agg)
        kw = {} if inst.get("weights") is None else {"weights": inst["weights"]}
        d0 = goals.distance(inst["goal"], start_agg, start_agg, **kw)
        found = hill_climb(model, start, inst, evaluations=budget, initial_step=0.2)
        final = goals.aggregate(goals.features(model, found))
        att = 0.0 if d0 <= 1e-9 else goals.attainment(inst["goal"], start_agg, final, **kw)
        rows.append({"seed": seed, "kind": inst["kind"], "solvable": goals.instruction_solvable(inst, start_agg),
                     "ceiling": att, "occluders": inst.get("occluders", []),
                     "start_vis": {c: start_agg["vis"][c] for c in goals.GOAL_CLASSES}})
        json.dump(rows, open(out, "w"))
