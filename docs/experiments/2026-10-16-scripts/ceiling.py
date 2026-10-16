"""Attainment ceiling: what a big search budget reaches per instruction kind.
best of {MB proposal + hill_climb(B, step 0.1), cold hill_climb(B, step 0.2)}, measured
from the episode start. Records the trajectory at several budgets.

usage: ceiling.py OUT.json POLICY.pt SEED_FROM SEED_TO BUDGET
"""
import json, sys, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
import goals
from rl.baselines import hill_climb, apply_controllable
from rl.model_based import ModelBasedPolicy, ENV_KWARGS
from rl.oneshot_train import build_eval_env

out, pol_path, lo, hi, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
env = build_eval_env(**ENV_KWARGS, hindsight_ratio=0.0)
policy = ModelBasedPolicy.load(pol_path)
rows = []
for seed in range(lo, hi):
    obs, _ = env.reset(seed=seed)
    model, inst, start = env._model, env._instruction, env._start_params.copy()
    score = lambda p: env._attainment(goals.aggregate(goals.features(model, p)))
    ctrl = np.asarray(env._controllable_values(start))
    a, _ = policy.predict(obs)
    prop = apply_controllable(start, np.clip(ctrl + a, -1, 1))
    row = {"seed": seed, "kind": inst["kind"], "solvable": env._solvable, "policy": score(prop)}
    for b in (60, 200, budget):
        warm = score(hill_climb(model, prop, inst, evaluations=b, initial_step=0.1, reference_params=start))
        row[f"warm{b}"] = max(warm, row["policy"])
    row[f"cold{budget}"] = score(hill_climb(model, start, inst, evaluations=budget, initial_step=0.2))
    row["best"] = max(row[f"warm{budget}"], row[f"cold{budget}"])
    rows.append(row)
    json.dump(rows, open(out, "w"))
