"""The search from the task's start at budgets between 0 and 200, for the
budget curve (docs/figures/make_budget_curve.py), which so far had the cold
search only at 200. Same fixed episodes and the same `hill_climb` call as
`rl.evaluate`'s `cold200`; the cold search does not depend on a policy, so one
run serves every seed.

Scored with the unlabelled shell the thesis numbers were produced with
(skeleton, lungs and soft peaks; changed to skeleton + soft on 2026-11-15), so
the new points sit on the same scale as the rest of the figure. `cold200` is
recomputed as a check against the earlier evaluation.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python docs/experiments/2026-11-17-scripts/cold_budget.py --split test --range 0 30 --out out/cold_budget/test_0.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import transfer  # noqa: E402

transfer.UNLABELLED_PEAKS = ("skeleton", "lungs", "soft")  # the shell before 2026-11-15

import torch  # noqa: E402

from rl.episodes import EVAL_EPISODES, fixed_episodes  # noqa: E402
from rl.search import hill_climb  # noqa: E402

BUDGETS = (10, 30, 100, 200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--range", nargs=2, type=int, default=(0, EVAL_EPISODES))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    lo, hi = args.range
    rows = []
    for index, episode in enumerate(fixed_episodes(args.split, hi, start=lo), start=lo):
        row = {"i": index, "kind": episode.kind, "volume": episode.volume}
        for budget in BUDGETS:
            params = hill_climb(episode.model, episode.start_params, episode.instruction, evaluations=budget)
            row[f"cold{budget}"] = episode.attainment(params)
        rows.append(row)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(rows, stream)
    print(f"[cold_budget] {len(rows)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
