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
from rl.baselines import BASELINES, hill_climb
from rl.candidates import _apply_action, _observation_for, _predict
from rl.oneshot_env import load_policy_metadata, observation_metadata
from rl.vis_eval import fixed_episodes

BASELINE_EVALUATIONS = {"expanded_hill_climb": 1000}


def baseline_name(name: str) -> str:
    if name not in BASELINE_EVALUATIONS:
        raise ValueError(f"unknown baseline: {name}")
    return name


def expanded_hill_climb(model, start_params, instruction, active_layers=None):
    """Fixed-episode baseline using larger search budget than legacy B4."""
    return hill_climb(model, start_params, instruction,
                      evaluations=BASELINE_EVALUATIONS["expanded_hill_climb"],
                      active_layers=active_layers)


def _score(policy, episode, model, active_layers=None) -> float:
    instruction = episode["instruction"]
    start_params = episode["start_params"]
    start_features = goals.features(model, start_params, active_layers)
    start_agg = goals.aggregate(start_features, active_layers)
    observation = _observation_for(model, start_params, instruction, start_agg)
    action = _predict(policy, observation, np.random.default_rng(0), True)
    params = _apply_action(start_params, action,
                           episode.get("policy_metadata", {}).get("action_mode", "absolute"))
    final_features = goals.features(model, params, active_layers)
    return goals.attainment(instruction["goal"], start_agg,
                            goals.aggregate(final_features, active_layers))


def _class_row(goal_class: str, status: str, attainment):
    return {"class": goal_class, "status": status,
            "attainment": attainment if status == "reachable" else None}


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


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--baseline", action="append",
                    choices=tuple(BASELINES) + tuple(BASELINE_EVALUATIONS), default=[])
    layers = ap.add_mutually_exclusive_group()
    layers.add_argument("--layers", help="active anatomy layers as inline JSON")
    layers.add_argument("--layers-file", help="JSON file containing active anatomy layers")
    return ap.parse_args(argv)


def main(active_layers=None, argv=None):
    args = parse_args(argv)
    if active_layers is None:
        active_layers = provenance.load_active_layers(args.layers, args.layers_file)

    from stable_baselines3 import SAC

    policy_metadata = load_policy_metadata(args.policies[0])
    episodes = fixed_episodes(args.split, args.episodes, seed=args.seed,
                              formulation="one_shot", active_layers=active_layers,
                              policy_metadata=policy_metadata)
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
            reachable_classes = set(goals.reachable_goal_classes(
                episode["volume"], model, active_layers))
            attainment = _score(policy, episode, model, active_layers)
            for goal_class in episode["instruction"]["targets"]:
                supported = goal_class in supported_classes
                reachable = goal_class in reachable_classes
                status = "reachable" if reachable else "unreachable" if supported else "unsupported"
                per_class_rows.append(_class_row(str(goal_class), status, attainment))
            mentioned = set(episode["instruction"]["targets"])
            for goal_class in goals.GOAL_CLASSES:
                if goal_class in mentioned:
                    continue
                status = ("reachable" if goal_class in reachable_classes else
                          "unreachable" if goal_class in supported_classes else "unsupported")
                per_class_rows.append(_class_row(goal_class, status, None))
            overall.append(attainment)
        results[path] = {
            "overall_median": statistics.median(overall),
            "overall_share_positive": sum(1 for a in overall if a > 0) / len(overall),
            "per_class": summarise_per_class(per_class_rows),
            "metadata": policy_metadata,
            "provenance": provenance.result_provenance(active_layers),
        }

    for name in args.baseline:
        baseline_fn = expanded_hill_climb if name == "expanded_hill_climb" else BASELINES[name]
        rows = []
        overall = []
        for episode in episodes:
            model = models[episode["volume"]]
            final_params = baseline_fn(model, episode["start_params"], episode["instruction"],
                                       active_layers=active_layers)
            start_agg = goals.aggregate(goals.features(model, episode["start_params"], active_layers), active_layers)
            final_agg = goals.aggregate(goals.features(model, final_params, active_layers), active_layers)
            attainment = goals.attainment(episode["instruction"]["goal"], start_agg, final_agg)
            overall.append(attainment)
            mentioned = set(episode["instruction"]["targets"])
            for goal_class in goals.GOAL_CLASSES:
                rows.append(_class_row(goal_class, "reachable" if goal_class in mentioned else "unsupported",
                                       attainment if goal_class in mentioned else None))
        results[name] = {"overall_median": statistics.median(overall),
                         "overall_share_positive": sum(value > 0 for value in overall) / len(overall),
                         "per_class": summarise_per_class(rows),
                         "metadata": observation_metadata(),
                         "provenance": provenance.result_provenance(active_layers)}

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
