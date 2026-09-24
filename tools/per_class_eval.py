"""Attainment per goal class, for one or more one-shot checkpoints.

`rl.vis_eval` reports overall and per-instruction-kind statistics, but the
question after the measurement fixes (see
docs/experiments/2026-09-16-retrain-after-measurement-fixes.md) is specifically
whether the lung instructions improved -- the broken ceiling
(`visibility.solo_max` probing at the retired band layout's fat peak) told the
policy lungs were unreachable on every volume, so that is where a difference
should show up if the fix mattered.

Scores only the policies, on the same `rl.vis_eval.fixed_episodes` set every
other evaluation uses, and skips the baselines, which is what makes this cheap
(no 200-evaluation hill climb per episode). An episode counts toward every
class its instruction mentions, so a compound naming two classes contributes to
both.

    .venv/bin/python -m tools.per_class_eval out/rl_v2/oneshot_v3_seed0/best.zip
"""
import argparse
import collections
import json
import statistics

import numpy as np

import goals
import provenance
import visibility
from rl.candidates import _apply_action, _observation_for, _predict
from rl.oneshot_env import observation_metadata
from rl.vis_eval import fixed_episodes


def _score(policy, episode, model) -> float:
    instruction = episode["instruction"]
    start_params = episode["start_params"]
    start_agg = goals.aggregate(model.features(start_params))
    observation = _observation_for(model, start_params, instruction, start_agg)
    action = _predict(policy, observation, np.random.default_rng(0), True)
    params = _apply_action(start_params, action)
    return goals.attainment(instruction["goal"], start_agg,
                            goals.aggregate(model.features(params)))


def summarise_per_class(rows: list) -> dict:
    """Summarise scores while retaining support and reachability accounting."""
    grouped = collections.defaultdict(list)
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        goal_class = row["class"]
        status = row["status"]
        counts[goal_class][status] += 1
        if status == "reachable" and row["attainment"] is not None:
            grouped[goal_class].append(float(row["attainment"]))

    report = {}
    for goal_class in sorted(set(grouped) | set(counts)):
        values = grouped[goal_class]
        report[goal_class] = {
            "median": statistics.median(values) if values else None,
            "share_positive": sum(value > 0 for value in values) / len(values) if values else None,
            "n": len(values),
            "supported": counts[goal_class]["reachable"] + counts[goal_class]["unreachable"],
            "reachable": counts[goal_class]["reachable"],
            "unsupported": counts[goal_class]["unsupported"],
            "unreachable": counts[goal_class]["unreachable"],
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    from stable_baselines3 import SAC

    episodes = fixed_episodes(args.split, args.episodes, seed=args.seed,
                              formulation="one_shot")
    models = {}
    for episode in episodes:
        if episode["volume"] not in models:
            models[episode["volume"]] = visibility.for_volume(episode["volume"])

    results = {}
    for path in args.policies:
        policy = SAC.load(path, device="cpu")
        per_class_rows = []
        overall = []
        for episode in episodes:
            model = models[episode["volume"]]
            supported_classes = set(goals.goal_classes_for_volume(episode["volume"]))
            reachable_classes = set(goals.reachable_goal_classes(episode["volume"], model))
            attainment = _score(policy, episode, model)
            for goal_class in episode["instruction"]["targets"]:
                supported = goal_class in supported_classes
                reachable = goal_class in reachable_classes
                per_class_rows.append({
                    "class": str(goal_class),
                    "status": "reachable" if reachable else "unreachable" if supported else "unsupported",
                    "attainment": attainment,
                })
            mentioned = set(episode["instruction"]["targets"])
            for goal_class in goals.GOAL_CLASSES:
                if goal_class in mentioned:
                    continue
                per_class_rows.append({
                    "class": goal_class,
                    "status": "reachable" if goal_class in reachable_classes else
                              "unreachable" if goal_class in supported_classes else "unsupported",
                    "attainment": None,
                })
            overall.append(attainment)
        results[path] = {
            "overall_median": statistics.median(overall),
            "overall_share_positive": sum(1 for a in overall if a > 0) / len(overall),
            "per_class": summarise_per_class(per_class_rows),
            "metadata": observation_metadata(),
            "provenance": provenance.IMPORT_TIME_PROVENANCE,
        }

    classes = sorted({c for r in results.values() for c in r["per_class"]})
    print(f"{'policy':44s} {'overall':>9s}  " + "  ".join(f"{c:>9s}" for c in classes))
    for path, r in results.items():
        row = f"{path[-42:]:44s} {r['overall_median']:+9.3f}  "
        row += "  ".join(
            f"{r['per_class'][c]['median']:+9.3f}" if c in r["per_class"] and r["per_class"][c]["median"] is not None else f"{'-':>9s}"
            for c in classes)
        print(row)

    print()
    print("episodes mentioning each class:",
          {c: next(r["per_class"][c]["n"] for r in results.values() if c in r["per_class"])
           for c in classes})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n[per_class_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
