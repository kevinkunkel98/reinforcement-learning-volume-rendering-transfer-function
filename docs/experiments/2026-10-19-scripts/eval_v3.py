"""Per episode on vis_eval's fixed 300 episodes (seed 0): policy alone, policy +
warm hill_climb at 30 / 200 evaluations, and (with --cold) hill_climb from the
start at 200 -- attainment and on-screen change (goals.visible_change, pp).
With --plan also the proposal refined through the checkpoint's dynamics
ensemble (rl.model_based.plan, no renders), alone and + 30 / 200.

    eval_v3.py CKPT OUT.json SPLIT LO HI [--cold] [--plan]
"""
import json
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, ".")
import goals  # noqa: E402
from rl import vis_eval as ve  # noqa: E402
from rl.baselines import hill_climb  # noqa: E402

ckpt, out, split, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
cold = "--cold" in sys.argv
planning = "--plan" in sys.argv
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

    def proposal_and_search(name, params):
        row[name], row[name + "_pp"] = scored(params)
        prefix = "" if name == "policy" else name + "_"
        for budget in (30, 200):
            warm = hill_climb(env._model, params, ep["instruction"], evaluations=budget,
                              initial_step=ve.REFINE_INITIAL_STEP, reference_params=start)
            att, pp = scored(warm)
            # vis_eval --refine's contract: keep the better of {proposal, refined}.
            row[f"{prefix}warm{budget}"], row[f"{prefix}warm{budget}_pp"] = (att, pp) if att > row[name] else (
                row[name], row[name + "_pp"])

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
