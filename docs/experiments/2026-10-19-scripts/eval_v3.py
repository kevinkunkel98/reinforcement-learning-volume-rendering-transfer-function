"""Per episode on vis_eval's fixed 300 episodes (seed 0): policy alone, policy +
warm hill_climb at 30 / 200 evaluations, and (with --cold) hill_climb from the
start at 200 -- attainment and on-screen change (goals.visible_change, pp).
With --plan also the proposal refined through the checkpoint's dynamics
ensemble (rl.model_based.plan, no renders), alone and + 30 / 200.
With --keep also every answer after rl.keep_repair (`<name>_keep`, plus its
evaluations and milliseconds), and for every answer the drift of the classes
the instruction does not mention (`<name>_drift`: sum of |log10 vis change|
over them and the unlabelled tissue, `<name>_bdrift`: sum of |brightness
change|). With --keep-occ also every answer after rl.keep_repair with
include_occluders=True (`<name>_keepocc`, same extras), and the drift
(`<name>_odrift`, sum of |log10 vis change|; `<name>_obdrift`, of
|brightness change|) of the instruction's weight-0 occluders, which `_drift`
does not cover. --no-search skips the + 30 / 200
searches. Without the new flags the output is what it was.

    eval_v3.py CKPT OUT.json SPLIT LO HI [--cold] [--plan] [--keep] [--keep-occ] [--no-search]
"""
import json
import sys
import time

import torch

torch.set_num_threads(1)
sys.path.insert(0, ".")
import goals  # noqa: E402
from rl import vis_eval as ve  # noqa: E402
from rl.baselines import hill_climb  # noqa: E402
from rl.keep_repair import keep_repair, mentioned_classes, occluder_classes  # noqa: E402

ckpt, out, split, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
cold = "--cold" in sys.argv
planning = "--plan" in sys.argv
keep_occ = "--keep-occ" in sys.argv
keep = "--keep" in sys.argv or keep_occ
repairs = [(suffix, include) for suffix, include, on in (("_keep", False, "--keep" in sys.argv),
                                                          ("_keepocc", True, keep_occ)) if on]
search = "--no-search" not in sys.argv
meta = ve.load_policy_metadata(ckpt)
episodes = ve.fixed_episodes(split, 300, seed=0, formulation="one_shot", policy_metadata=meta)[lo:hi]
policy = ve._load_sac(ckpt)
rows = []
for i, ep in enumerate(episodes, start=lo):
    env, obs, info = ve._frozen_one_shot_env(ep["volume"], ep["start_params"], ep["instruction"],
                                             policy_metadata=ve._episode_policy_metadata(ep, meta))
    action, _ = policy.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    proposal, start = env._params.copy(), env._start_params.copy()

    def scored(params):
        final = goals.aggregate(ve._features(env._model, params))
        return env._attainment(final), goals.visible_change(
            ep["instruction"]["goal"], env._start_agg, final, ep["instruction"].get("weights"))

    row = {"i": i, "kind": info["kind"], "volume": ep["volume"]}
    unmentioned = [c for c in goals.GOAL_CLASSES if c not in mentioned_classes(env._instruction["goal"])]
    occluders = sorted(occluder_classes(env._instruction["goal"], env._instruction.get("weights")))

    def record(name, params):
        row[name], row[name + "_pp"] = scored(params)
        if keep:
            c, b = goals.progress(env._start_agg, goals.aggregate(ve._features(env._model, params)))
            row[name + "_drift"] = sum(abs(c[k]) for k in unmentioned + ["other"])
            row[name + "_bdrift"] = sum(abs(b[k]) for k in unmentioned)
            if keep_occ:
                row[name + "_odrift"] = sum(abs(c[k]) for k in occluders)
                row[name + "_obdrift"] = sum(abs(b[k]) for k in occluders)

    def repaired(name, params):
        for suffix, include in repairs:
            started = time.perf_counter()
            fixed, n_evals = keep_repair(env._model, env._instruction["goal"], start, params,
                                         weights=env._instruction.get("weights"), start_agg=env._start_agg,
                                         include_occluders=include)
            row[name + suffix + "_ms"] = (time.perf_counter() - started) * 1000.0
            row[name + suffix + "_evals"] = n_evals
            record(name + suffix, fixed)

    def proposal_and_search(name, params):
        record(name, params)
        repaired(name, params)
        prefix = "" if name == "policy" else name + "_"
        for budget in ((30, 200) if search else ()):
            warm = hill_climb(env._model, params, ep["instruction"], evaluations=budget,
                              initial_step=ve.REFINE_INITIAL_STEP, reference_params=start)
            att, pp = scored(warm)
            # vis_eval --refine's contract: keep the better of {proposal, refined}.
            row[f"{prefix}warm{budget}"], row[f"{prefix}warm{budget}_pp"] = (att, pp) if att > row[name] else (
                row[name], row[name + "_pp"])
            if keep:
                answer = warm if att > row[name] else params
                record(f"{prefix}warm{budget}", answer)
                repaired(f"{prefix}warm{budget}", answer)

    proposal_and_search("policy", proposal)
    if planning:
        env._params = start.copy()
        env.step(ve._planned(policy, env, action))
        proposal_and_search("plan", env._params.copy())
    if cold:
        row["cold200"], row["cold200_pp"] = scored(hill_climb(env._model, start, ep["instruction"],
                                                              evaluations=200))
    row["start_other"] = env._start_agg["vis"]["other"]
    rows.append(row)
    with open(out, "w") as stream:
        json.dump(rows, stream)
