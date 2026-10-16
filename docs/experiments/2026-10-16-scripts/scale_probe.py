"""Does scaling the policy's residual action by k change attainment?
k > 1 helping => policy is timid (undershoots along a good direction)."""
import sys, collections, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, ".")
from stable_baselines3 import SAC
from rl.oneshot_train import build_eval_env

SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 'mask', 'mask0.5')
import transfer
env = build_eval_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                     hindsight_ratio=0.0, balance_classes=False, hindsight_brightness_share=0.0)
for path in sys.argv[1:]:
    policy = SAC.load(path, device="cpu")
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for seed in range(50_000, 50_000 + 150):
        obs, _ = env.reset(seed=seed)
        kind = env._instruction["kind"]
        action, _ = policy.predict(obs, deterministic=True)
        mask = np.zeros_like(action)
        for c in env._instruction["targets"]:
            i = transfer.ANATOMICAL_PEAK_INDEX[c]; mask[3*i:3*i+3] = 1
        for k in SCALES:
            env.reset(seed=seed)
            a = action * mask if k == "mask" else action * mask * 0.5 if k == "mask0.5" else k * action
            _, _, _, _, info = env.step(np.clip(a, -1, 1))
            res[kind][k].append(info["attainment"])
            res["ALL"][k].append(info["attainment"])
    print(path)
    for kind, by_k in sorted(res.items()):
        print("  %-11s n=%3d " % (kind, len(by_k[1.0])) +
              "  ".join("k=%s %+.3f" % (k, np.median(v)) for k, v in by_k.items()), flush=True)
