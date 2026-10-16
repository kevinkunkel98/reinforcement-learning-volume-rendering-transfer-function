"""Score SAC checkpoints on the same val episodes as mb_train (seeds 50_000+), seed-average
per episode, and compare paired against model-based eval.json files.

usage: paired_eval.py OUT.json --sac a.zip b.zip ... --mb x/eval.json y/eval.json ...
"""
import argparse, collections, json, math, sys
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")


def wilcoxon(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = r[d > 0].sum()
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sd
    return math.erfc(abs(z) / math.sqrt(2))


def sac_rows(path, n):
    from stable_baselines3 import SAC
    from rl.oneshot_train import build_eval_env
    env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                         hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
    m = SAC.load(path, device="cpu")
    rows = []
    for seed in range(50_000, 50_000 + n):
        obs, _ = env.reset(seed=seed)
        a, _ = m.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(a)
        rows.append({"seed": seed, "kind": info["kind"], "attainment": info["attainment"]})
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--sac", nargs="*", default=[]); ap.add_argument("--mb", nargs="*", default=[])
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()
try:
    cache = json.load(open(args.out))
except FileNotFoundError:
    cache = {}
for p in args.sac:
    if p not in cache:
        cache[p] = sac_rows(p, args.n)
        json.dump(cache, open(args.out, "w"))
groups = {"SAC": [cache[p] for p in args.sac], "MB": [json.load(open(p))["rows"] for p in args.mb]}
avg = {}
for g, runs in groups.items():
    if runs:
        avg[g] = np.mean([[r["attainment"] for r in rows[:args.n]] for rows in runs], 0)
kinds = [r["kind"] for r in groups["MB"][0][:args.n]] if args.mb else [r["kind"] for r in groups["SAC"][0]]
assert all(r["kind"] == k for runs in groups.values() for rows in runs for r, k in zip(rows, kinds))
for g, v in avg.items():
    print(f"{g:4s} ({len(groups[g])} runs) median {np.median(v):+.3f}  " + "  ".join(
        f"{k} {np.median([x for x, kk in zip(v, kinds) if kk == k]):+.3f}" for k in sorted(set(kinds))))
if len(avg) == 2:
    print(f"MB ahead on {np.mean(avg['MB'] > avg['SAC']):.0%} of episodes, paired Wilcoxon p = {wilcoxon(avg['MB'], avg['SAC']):.2g}")
