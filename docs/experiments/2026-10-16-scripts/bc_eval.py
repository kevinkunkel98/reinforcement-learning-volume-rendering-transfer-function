"""Pretrain-only actors (no RL) on different pretrain mixes, scored on real val instructions."""
import collections, sys, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
from stable_baselines3 import SAC
from rl import pretrain
from rl.oneshot_train import build_eval_env, merge_datasets, LEARNING_STARTS

env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                     hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
H = [pretrain.load_dataset(f"out/rl_v4/hindsight_v9_bright_seed{s}.npz")[0] for s in (0, 1, 2)]
S = [pretrain.load_dataset(f"out/rl_v4/search_v9_seed{s}.npz")[0] for s in (0, 1, 2)]
r = np.concatenate([s["rewards"] for s in S])
print("search reward quantiles", np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3))

def filt(d, thr):
    keep = d["rewards"] > thr
    return {k: v[keep] for k, v in d.items()}

VARIANTS = {
    "A h0+s0 (E3)": (merge_datasets(H[0], S[0]), 30),
    "B h0+s012": (merge_datasets(H[0], *S), 30),
    "C s012 only": (merge_datasets(*S), 30),
    "D h0+s012 r>0.2": (merge_datasets(H[0], *[filt(s, 0.2) for s in S]), 30),
    "E s012 only 100ep": (merge_datasets(*S), 100),
}
episodes = list(range(50_000, 50_000 + 150))
for name, (data, epochs) in VARIANTS.items():
    model = SAC("MlpPolicy", env, seed=0, verbose=0, learning_starts=LEARNING_STARTS, device="cpu")
    rep = pretrain.pretrain_actor(model, data, epochs=epochs, seed=0)
    res = collections.defaultdict(list)
    for seed in episodes:
        obs, _ = env.reset(seed=seed)
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        res[env._instruction["kind"]].append(info["attainment"]); res["ALL"].append(info["attainment"])
    print(f"{name:20s} n={rep['samples']:5d} val/zero {rep['val_mse']/rep['zero_mse']:.3f} | " +
          "  ".join(f"{k} {np.median(v):+.3f}" for k, v in sorted(res.items())), flush=True)
