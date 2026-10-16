"""Policy as warm start: policy + hill_climb(B) vs hill_climb(B) from the start, equal budget."""
import collections, sys, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
from stable_baselines3 import SAC
import goals
from rl.baselines import hill_climb, CONTROLLABLE, _evaluate
from rl.oneshot_train import build_eval_env

env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                     hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
policy = SAC.load(sys.argv[1], device="cpu")
BUDGETS = (10, 30)
WARM_ONLY = True

def warm_climb(model, reference, warm, instruction, evaluations, initial_step):
    """hill_climb scored against `reference` (the original start), searching from `warm`."""
    goal, weights = instruction["goal"], instruction.get("weights")
    ref_agg = goals.aggregate(goals.features(model, reference))
    best, best_d, used, step = warm.copy(), _evaluate(model, goal, ref_agg, warm, None, weights=weights), 1, initial_step
    while used < evaluations and step > 1e-3:
        improved = False
        for dim in CONTROLLABLE:
            for sign in (1.0, -1.0):
                if used >= evaluations: break
                cand = best.copy()
                for i in dim: cand[i] = float(np.clip(cand[i] + sign * step, -1, 1))
                d = _evaluate(model, goal, ref_agg, cand, None, weights=weights); used += 1
                if d < best_d: best_d, best, improved = d, cand, True
            if used >= evaluations: break
        if not improved: step /= 2.0
    return best
res = collections.defaultdict(lambda: collections.defaultdict(list))
for seed in range(50_000, 50_000 + 150):
    obs, _ = env.reset(seed=seed)
    kind, inst, model = env._instruction["kind"], env._instruction, env._model
    start_params = env._start_params.copy()
    score = lambda p: env._attainment(goals.aggregate(goals.features(model, p)))
    action, _ = policy.predict(obs, deterministic=True)
    env.step(action)
    proposal = env._params.copy()
    rows = {"policy": score(proposal)}
    w60 = warm_climb(model, start_params, proposal, inst, 60, 0.1)
    rows["warm60"] = score(w60)
    w30, c30 = warm_climb(model, start_params, proposal, inst, 30, 0.1), warm_climb(model, start_params, start_params, inst, 30, 0.1)
    rows["best(w30,c30)"] = max(score(w30), score(c30))
    for k, v in rows.items():
        res[kind][k].append(v); res["ALL"][k].append(v)
for kind, by in sorted(res.items()):
    print(f"{kind:11s} n={len(by['policy']):3d} " + "  ".join(f"{k} {np.median(v):+.3f}" for k, v in by.items()), flush=True)
