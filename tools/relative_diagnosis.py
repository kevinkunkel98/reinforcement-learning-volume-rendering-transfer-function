"""Why `relative` instructions score near zero: per-episode decomposition.

Samples `relative` instructions on the validation split (fresh seeds, no
hindsight), scores one or more one-shot checkpoints and optionally
`hill_climb`, and splits each final distance into its terms (unmet goal,
unmentioned-class visibility drift, brightness drift, "other"). Episodes are
also classed as solvable or not from the start state alone: a decrease whose
target lies below goals.EPSILON's log floor, or an increase of a class that
is practically invisible (vis < EPSILON), cannot be met by any method.

    .venv/bin/python -m tools.relative_diagnosis out/rl_v4/e3_v9_searchmix_bright_150k_seed0/best.zip \\
        --episodes 200 --hill 200 --out out/relative_diagnosis.json

See docs/experiments/2026-10-15-relative-diagnosis.md for the findings.
"""
import argparse
import collections
import json
import math
import statistics

import numpy as np
from stable_baselines3 import SAC

import goals
from rl.baselines import hill_climb
from rl.oneshot_train import build_eval_env

SEED_BASE = 50_000  # disjoint from training (0..) and validation_episodes (10_000..)


def decompose(goal, start, final) -> dict:
    """The terms of goals.distance(goal, start, final), plus the goal-class change."""
    c, b = goals.progress(start, final)
    n = len(goals.GOAL_CLASSES)
    d, m = goal[:n], goal[n:2 * n]
    parts = {"goal": 0.0, "keep_vis": 0.0, "keep_bright": 0.0}
    for i, g in enumerate(goals.GOAL_CLASSES):
        parts["goal"] += m[i] * abs(c[g] - d[i])
        parts["keep_vis"] += goals.LAMBDA_KEEP * (1 - m[i]) * max(0.0, abs(c[g]) - goals.KEEP_TOLERANCE)
        parts["keep_bright"] += goals.LAMBDA_KEEP * goals.KAPPA * max(0.0, abs(b[g]) - goals.KEEP_TOLERANCE)
    parts["other"] = goals.LAMBDA_KEEP * max(0.0, abs(c["other"]) - goals.KEEP_TOLERANCE)
    return {**parts, "c": c}


def solvable(direction: str, strength: float, start_vis: float) -> bool:
    """Whether a relative target can be met at all from this start visibility."""
    if direction == "decrease":
        floor = math.log10(goals.EPSILON) - math.log10(start_vis + goals.EPSILON)
        return abs(floor) >= strength
    return start_vis >= goals.EPSILON


def collect(checkpoints, episodes: int, hill: int) -> list:
    env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                         hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
    policies = {path: SAC.load(path, device="cpu") for path in checkpoints}
    rows, seed = [], SEED_BASE
    while len(rows) < episodes:
        obs, _ = env.reset(seed=seed)
        seed += 1
        instruction = env._instruction
        if instruction["kind"] != "relative":
            continue
        (goal_class, target), = instruction["targets"].items()
        start, model, start_params = env._start_agg, env._model, env._start_params.copy()
        direction = "increase" if target["vis"] > 0 else "decrease"
        row = {"seed": seed - 1, "volume": env._volume, "class": goal_class,
               "direction": direction, "strength": abs(target["vis"]),
               "start_vis": start["vis"][goal_class],
               "solvable": solvable(direction, abs(target["vis"]), start["vis"][goal_class]),
               "d_start": goals.distance(instruction["goal"], start, start), "arms": {}}
        finals = {}
        for path, policy in policies.items():
            action, _ = policy.predict(obs, deterministic=True)
            env.step(action)
            finals[path] = goals.aggregate(goals.features(model, env._params))
        if hill:
            finals["hill_climb"] = goals.aggregate(goals.features(
                model, hill_climb(model, start_params, instruction, evaluations=hill)))
        for name, final in finals.items():
            parts = decompose(instruction["goal"], start, final)
            total = parts["goal"] + parts["keep_vis"] + parts["keep_bright"] + parts["other"]
            row["arms"][name] = {"attainment": 1.0 - total / row["d_start"],
                                 "goal": parts["goal"], "keep_vis": parts["keep_vis"],
                                 "keep_bright": parts["keep_bright"], "other": parts["other"],
                                 "achieved_share": parts["c"][goal_class] / target["vis"]}
        rows.append(row)
    return rows


def report(rows: list) -> None:
    def line(label, subset, arm):
        values = [r["arms"][arm]["attainment"] for r in subset]
        if not values:
            return
        print(f"  {label:24s} n={len(values):4d} mean {statistics.mean(values):6.2f} "
              f"median {statistics.median(values):6.2f} >0 {sum(v > 0 for v in values) / len(values):4.0%}")
    for arm in rows[0]["arms"]:
        print(arm)
        line("all", rows, arm)
        line("solvable", [r for r in rows if r["solvable"]], arm)
        line("unsolvable", [r for r in rows if not r["solvable"]], arm)
    counts = collections.Counter(r["solvable"] for r in rows)
    print(f"solvable {counts[True]} / unsolvable {counts[False]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--hill", type=int, default=0, help="hill_climb evaluations per episode (0 = skip)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    rows = collect(args.checkpoints, args.episodes, args.hill)
    if args.out:
        with open(args.out, "w") as stream:
            json.dump(rows, stream)
    report(rows)


if __name__ == "__main__":
    main()
