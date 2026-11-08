"""The official evaluation: every method on the same fixed episodes.

Per episode of `rl.episodes.fixed_episodes` (300 per split, seed 0): the policy
alone (`policy`), the policy refined by planning through its dynamics ensemble
(`plan`, no renders), each followed by a warm-started search of 30 and 200
evaluations (`warm30`, `plan_warm200`, ...), and the search from the task's
start (`cold200`). Every value is attainment; `<name>_pp` is the on-screen
change toward the goal (`goals.visible_change`, percentage points). A warm
search keeps the better of {proposal, refined}.

`--keep` also scores every answer after the viewer's keep repair
(`<name>_keep`) and `--colour` after its colour guard (`<name>_keep_colour`),
with the drift of the classes the instruction does not mention (`_drift`,
`_bdrift`).

    .venv/bin/python -m rl.evaluate models/v3c_seed1/checkpoint.pt --split test --out out/eval_test_s1.json
    .venv/bin/python -m rl.evaluate --summary out/eval_test_s*.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import goals
from rl import model_based
from rl.colour_guard import colour_guard
from rl.episodes import EVAL_EPISODES, controllable_values, fixed_episodes, policy_metadata, solo_max_log
from rl.keep_repair import keep_repair, mentioned_classes
from rl.search import hill_climb

# A search warm-started at a proposal probes smaller steps than one from the
# start (0.2): 0.1 was best on 150 val episodes.
REFINE_INITIAL_STEP = 0.1
BUDGETS = (30, 200)


def planned(policy, episode, action) -> np.ndarray:
    """`action` refined through the policy's dynamics ensemble (`model_based.plan`)."""
    instruction = episode.instruction
    return model_based.plan(policy.dynamics, instruction["goal"], instruction.get("weights"),
                            episode.model.histogram, solo_max_log(episode.model),
                            controllable_values(episode.start_params), episode.start_agg, action)


def evaluate_episode(policy, episode, index: int, plan: bool = True, budgets=BUDGETS, cold: bool = False,
                     keep: bool = False, colour: bool = False) -> dict:
    """One episode, every requested method: a row of the result file."""
    instruction, start = episode.instruction, episode.start_params
    unmentioned = [c for c in goals.GOAL_CLASSES if c not in mentioned_classes(instruction["goal"])]
    reference = goals.starting_params()
    row = {"i": index, "kind": episode.kind, "volume": episode.volume}

    def record(name, params):
        row[name], row[name + "_pp"] = episode.attainment(params), episode.visible_pp(params)
        if keep:
            c, b = goals.progress(episode.start_agg, episode.aggregate(params))
            row[name + "_drift"] = sum(abs(c[k]) for k in unmentioned + ["other"])
            row[name + "_bdrift"] = sum(abs(b[k]) for k in unmentioned)

    def repaired(name, params):
        if not keep:
            return
        started = time.perf_counter()
        fixed, evaluations = keep_repair(episode.model, instruction["goal"], start, params,
                                         weights=instruction.get("weights"), start_agg=episode.start_agg)
        row[name + "_keep_ms"] = (time.perf_counter() - started) * 1000.0
        row[name + "_keep_evals"] = evaluations
        record(name + "_keep", fixed)
        if colour:
            started = time.perf_counter()
            guarded = colour_guard(instruction["goal"], start, fixed, reference, weights=instruction.get("weights"))
            row[name + "_keep_colour_ms"] = row[name + "_keep_ms"] + (time.perf_counter() - started) * 1000.0
            row[name + "_keep_colour_evals"] = evaluations
            record(name + "_keep_colour", guarded)

    def answer_and_search(name, params):
        record(name, params)
        repaired(name, params)
        prefix = "" if name == "policy" else name + "_"
        for budget in budgets:
            warm = hill_climb(episode.model, params, instruction, evaluations=budget,
                              initial_step=REFINE_INITIAL_STEP, reference_params=start)
            better = episode.attainment(warm) > row[name]
            answer = warm if better else params
            key = f"{prefix}warm{budget}"
            row[key], row[key + "_pp"] = (episode.attainment(warm), episode.visible_pp(warm)) if better else (
                row[name], row[name + "_pp"])
            if keep:
                record(key, answer)
                repaired(key, answer)

    action, _ = policy.predict(episode.observation())
    answer_and_search("policy", episode.params_for(action))
    if plan:
        answer_and_search("plan", episode.params_for(planned(policy, episode, action)))
    if cold:
        cold_params = hill_climb(episode.model, start, instruction, evaluations=200)
        row["cold200"], row["cold200_pp"] = episode.attainment(cold_params), episode.visible_pp(cold_params)
    row["start_other"] = episode.start_agg["vis"]["other"]
    return row


def run(checkpoint: str, split: str, out: str, lo: int = 0, hi: int = EVAL_EPISODES, **options) -> list:
    """Evaluate `checkpoint` on fixed episodes `lo:hi` of `split`, writing the rows to `out` as it goes."""
    policy_metadata(checkpoint)
    policy = model_based.ModelBasedPolicy.load(checkpoint)
    episodes = fixed_episodes(split, hi)[lo:hi]
    rows = []
    for index, episode in enumerate(episodes, start=lo):
        rows.append(evaluate_episode(policy, episode, index, **options))
        with open(out, "w", encoding="utf-8") as stream:
            json.dump(rows, stream)
    return rows


# --- summary -------------------------------------------------------------------

def wilcoxon(x, y) -> dict:
    """Paired Wilcoxon signed-rank test of `x` against `y` (normal
    approximation with continuity correction; zero differences dropped, ties
    share their average rank). `{"statistic", "p_value", "n"}`."""
    diffs = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    nonzero = diffs[diffs != 0.0]
    n = nonzero.shape[0]
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}
    abs_diffs = np.abs(nonzero)
    order = np.argsort(abs_diffs, kind="stable")
    sorted_abs = abs_diffs[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    statistic = min(float(np.sum(ranks[nonzero > 0.0])), float(np.sum(ranks[nonzero < 0.0])))
    mean = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    offset = statistic - mean
    z = (offset - math.copysign(0.5, offset) if offset else 0.0) / sigma
    p_value = min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))
    return {"statistic": statistic, "p_value": p_value, "n": n}


METHODS = ("policy", "plan", "warm30", "plan_warm30", "warm200", "plan_warm200", "cold200")


def summary(paths) -> str:
    """Median attainment per method and file, and planning's paired tests."""
    lines = [f"{'file':<34}" + "".join(f"{m:>13}" for m in METHODS)]
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            rows = json.load(stream)
        medians = [f"{np.median([r[m] for r in rows]):>13.3f}" if m in rows[0] else f"{'-':>13}" for m in METHODS]
        lines.append(f"{os.path.basename(path):<34}" + "".join(medians))
        for a, b in (("plan", "policy"), ("plan_warm30", "warm30"), ("plan_warm30", "cold200")):
            if a in rows[0] and b in rows[0]:
                test = wilcoxon([r[a] for r in rows], [r[b] for r in rows])
                lines.append(f"    {a} vs {b}: p = {test['p_value']:.2g} (n = {test['n']})")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--out")
    parser.add_argument("--range", nargs=2, type=int, default=(0, EVAL_EPISODES), metavar=("LO", "HI"),
                        help="episodes lo:hi of the fixed 300, to split a run over processes")
    parser.add_argument("--no-plan", action="store_true")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--cold", action="store_true", help="also the 200-evaluation search from the start")
    parser.add_argument("--keep", action="store_true", help="also every answer after the keep repair")
    parser.add_argument("--colour", action="store_true", help="also after the colour guard (implies --keep)")
    parser.add_argument("--summary", nargs="+", metavar="RESULT", help="print medians of result files")
    args = parser.parse_args(argv)
    if args.summary:
        print(summary(args.summary))
        return
    if not args.checkpoint or not args.out:
        parser.error("a checkpoint and --out are needed (or --summary)")
    torch.set_num_threads(1)
    rows = run(args.checkpoint, args.split, args.out, *args.range, plan=not args.no_plan,
               budgets=() if args.no_search else BUDGETS, cold=args.cold,
               keep=args.keep or args.colour, colour=args.colour)
    print(f"[evaluate] {len(rows)} episodes -> {args.out}")
    print(summary([args.out]))


if __name__ == "__main__":
    main()
