"""App number: policy proposal + hill_climb(B) measured from the episode start, for a
model-based policy (.pt) and a SAC policy (.zip), vs cold hill_climb(B), same episodes.

usage: warm_compare.py OUT.json MB.pt SAC.zip SEED_FROM SEED_TO
"""
import json, sys, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from stable_baselines3 import SAC
import goals, mb_train
from rl.baselines import hill_climb, apply_controllable
from rl.oneshot_train import build_eval_env

out, mb_path, sac_path, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                     hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
net = mb_train.mlp(97, 24, 256, 2); net.load_state_dict(torch.load(mb_path))
sac = SAC.load(sac_path, device="cpu")
BUDGETS = (30, 60)
rows = []
for seed in range(lo, hi):
    obs, _ = env.reset(seed=seed)
    model, inst, start = env._model, env._instruction, env._start_params.copy()
    score = lambda p: env._attainment(goals.aggregate(goals.features(model, p)))
    ctrl = np.asarray(env._controllable_values(start))
    with torch.no_grad():
        a_mb = torch.tanh(net(torch.as_tensor(obs).clamp(-10, 10)))[None][0].numpy()
    a_sac, _ = sac.predict(obs, deterministic=True)
    props = {"mb": apply_controllable(start, np.clip(ctrl + np.clip(a_mb, -1, 1), -1, 1)),
             "sac": apply_controllable(start, np.clip(ctrl + np.clip(a_sac, -1, 1), -1, 1)),
             "cold": start}
    row = {"seed": seed, "kind": inst["kind"]}
    for name, p in props.items():
        if name != "cold":
            row[f"{name}0"] = score(p)
        for b in BUDGETS:
            refined = hill_climb(model, p, inst, evaluations=b, initial_step=0.1, reference_params=start)
            row[f"{name}{b}"] = max(score(refined), row.get(f"{name}0", 0.0)) if name != "cold" else score(refined)
    rows.append(row)
    json.dump(rows, open(out, "w"))
