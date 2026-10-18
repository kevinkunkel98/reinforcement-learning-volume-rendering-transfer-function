"""Per episode: policy alone, policy + warm hill_climb at budgets B, cold hill_climb at B.
Same 300 val episodes as vis_eval (seed 0). Usage: warm_vs_cold.py ckpt out.json lo hi"""
import json, sys
import torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
import goals
from rl import vis_eval as ve
from rl.baselines import hill_climb

ckpt, out, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
split = sys.argv[5] if len(sys.argv) > 5 else "val"
BUDGETS = [10, 30, 100, 200]
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
    score = lambda p: env._attainment(goals.aggregate(ve._features(env._model, p)))
    row = {"i": i, "kind": info["kind"], "policy": info["attainment"]}
    for b in BUDGETS:
        warm = hill_climb(env._model, proposal, ep["instruction"], evaluations=b,
                          initial_step=ve.REFINE_INITIAL_STEP, reference_params=start)
        row[f"warm{b}"] = max(score(warm), row["policy"])
        row[f"cold{b}"] = score(hill_climb(env._model, start, ep["instruction"], evaluations=b))
    rows.append(row)
    json.dump(rows, open(out, "w"))
