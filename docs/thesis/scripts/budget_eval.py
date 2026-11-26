"""Attainment over the search budget on the original test split, for the
thesis budget figure: the official evaluation (`rl.evaluate.evaluate_episode`,
same fixed 300 episodes, CPU visibility as for every reported number) with
warm searches at 10 / 30 / 100 / 200 evaluations, plus the search from the
task's start at the same budgets (as docs/experiments/2026-11-17-scripts/
cold_budget.py). One checkpoint: the viewer's v3c seed 1.

    .venv/bin/python docs/thesis/scripts/budget_eval.py --range 0 50 --out out/thesis/budget_0.json
    .venv/bin/python docs/thesis/scripts/budget_eval.py --summary out/thesis/budget_*.json

At 30 and 200 the warm values must equal the v3c seed-1 test row of
docs/experiments/2026-10-19-task-v3.md (0.67 / 0.78 / 0.88 / 0.92), and
cold200 its 0.70 -- the run checks its own consistency that way.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from rl import model_based  # noqa: E402
from rl.episodes import fixed_episodes, policy_metadata  # noqa: E402
from rl.evaluate import evaluate_episode  # noqa: E402
from rl.search import hill_climb  # noqa: E402

CHECKPOINT = "models/v3c_seed1/checkpoint.pt"
BUDGETS = (10, 30, 100, 200)
KEYS = (["policy"] + [f"warm{b}" for b in BUDGETS] + ["plan"] + [f"plan_warm{b}" for b in BUDGETS]
        + [f"cold{b}" for b in BUDGETS])


def run(split, lo, hi, out):
    policy_metadata(CHECKPOINT)
    policy = model_based.ModelBasedPolicy.load(CHECKPOINT)
    rows = []
    for index, episode in enumerate(fixed_episodes(split, hi, start=lo), start=lo):
        row = evaluate_episode(policy, episode, index, plan=True, budgets=BUDGETS, cold=False)
        for budget in BUDGETS:
            params = hill_climb(episode.model, episode.start_params, episode.instruction, evaluations=budget)
            row[f"cold{budget}"] = episode.attainment(params)
        rows.append(row)
        with open(out, "w", encoding="utf-8") as stream:
            json.dump(rows, stream)
    print(f"[budget_eval] {len(rows)} episodes -> {out}")


def summary(patterns):
    rows = [r for p in patterns for f in sorted(glob.glob(p)) for r in json.load(open(f))]
    rows = sorted({r["i"]: r for r in rows}.values(), key=lambda r: r["i"])
    print(f"{len(rows)} episodes")
    for key in KEYS:
        print(f"  {key:<16} {np.median([r[key] for r in rows]):.3f}")
    kinds = sorted({r["kind"] for r in rows})
    print("by kind (policy / plan / plan_warm30 / plan_warm200 / cold200):")
    for kind in kinds:
        sub = [r for r in rows if r["kind"] == kind]
        print(f"  {kind:<11} n={len(sub):<3}" + "".join(
            f" {np.median([r[k] for r in sub]):.2f}" for k in ("policy", "plan", "plan_warm30", "plan_warm200", "cold200")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--range", nargs=2, type=int, default=(0, 300))
    ap.add_argument("--out")
    ap.add_argument("--summary", nargs="+")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
        return
    torch.set_num_threads(1)
    run(args.split, *args.range, args.out)


if __name__ == "__main__":
    main()
