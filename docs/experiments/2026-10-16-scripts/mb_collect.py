"""Model-based RL data: for many training-split starts, render ACTIONS random TF changes
and record what they did to the aggregated features; also sample INSTR real
instructions per start for policy training.

usage: mb_collect.py OUT N_STARTS SEED_BASE WORKERS [ACTIONS] [POLICY_ZIP]
With POLICY_ZIP, half the actions are that policy's (closed-form residual) proposal plus noise.
"""
import sys, os, multiprocessing as mp
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTIONS, INSTR = 20, 8


def state_vec(e, agg):
    import goals, math
    lv = [math.log10(agg["vis"].get(c, 0.0) + goals.EPSILON) for c in list(goals.GOAL_CLASSES) + ["other"]]
    br = [agg["bright"][c] for c in goals.GOAL_CLASSES]
    return np.array(lv + br + [agg["coverage"]], np.float32)


def work(args):
    seeds, n_actions, policy_zip = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch; torch.set_num_threads(1)
    import goals
    from rl.baselines import CONTROLLABLE, apply_controllable
    from rl.oneshot_train import build_env
    e = build_env(policy_version="oneshot-v9", action_mode="residual", reward_mode="target",
                  hindsight_ratio=0.0, balance_classes=True, hindsight_brightness_share=0.0)
    policy = None
    if policy_zip and policy_zip.endswith(".pt"):
        import mb_train
        net = mb_train.mlp(97, len(CONTROLLABLE), 256, 2)
        net.load_state_dict(torch.load(policy_zip))

        class _P:
            def predict(self, o, deterministic=False):
                with torch.no_grad():
                    return torch.tanh(net(torch.as_tensor(o, dtype=torch.float32).clamp(-10, 10)))[None][0].numpy(), None
        policy = _P()
    elif policy_zip:
        from stable_baselines3 import SAC
        policy = SAC.load(policy_zip, device="cpu")
    G = len(CONTROLLABLE)
    out = {k: [] for k in ("hist", "solo", "ctrl", "s0", "act", "s1", "useless", "goal", "weights", "kind")}
    for seed in seeds:
        obs, _ = e.reset(seed=seed)
        rng = np.random.default_rng(seed)
        ctrl = np.asarray(e._controllable_values(e._start_params))
        acts = []
        for j in range(n_actions):
            r = rng.random()
            if policy is not None and j < n_actions // 2:
                a, _ = policy.predict(obs, deterministic=False)
                a = np.clip(a + rng.normal(0, rng.uniform(0, 0.3), G), -1, 1)
            elif r < 0.4:
                a = rng.uniform(-1, 1, G) * rng.uniform(0.05, 0.8)
            elif r < 0.9:
                a = np.zeros(G)
                idx = rng.choice(G, rng.integers(1, 5), replace=False)
                a[idx] = rng.uniform(-1, 1, len(idx))
            else:
                a = np.zeros(G)
            a = np.clip(ctrl + a, -1, 1) - ctrl          # residual actually applied
            raw = goals.features(e._model, apply_controllable(e._start_params, ctrl + a))
            acts.append((a, state_vec(e, goals.aggregate(raw)), goals.is_useless(raw)))
        goals_, weights_, kinds = [], [], []
        for _ in range(INSTR):
            inst = goals.sample_instruction(e._volume, e._model, e._start_agg, rng, None,
                                            e._class_counts, True)
            for c in inst["targets"]:
                e._class_counts[c] = e._class_counts.get(c, 0) + 1
            w = inst.get("weights") or {}
            goals_.append(inst["goal"]); kinds.append(inst["kind"])
            weights_.append([w.get(c, 1.0) for c in goals.GOAL_CLASSES])
        out["hist"].append(np.asarray(e._model.histogram, np.float32))
        out["solo"].append(np.asarray(e._solo_max_log(e._volume, e._model), np.float32))
        out["ctrl"].append(ctrl.astype(np.float32))
        out["s0"].append(state_vec(e, e._start_agg))
        out["act"].append(np.array([a for a, _, _ in acts], np.float32))
        out["s1"].append(np.array([s for _, s, _ in acts], np.float32))
        out["useless"].append(np.array([u for _, _, u in acts], bool))
        out["goal"].append(np.array(goals_, np.float32)); out["weights"].append(np.array(weights_, np.float32))
        out["kind"].append(kinds)
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    out, n, base, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    n_actions = int(sys.argv[5]) if len(sys.argv) > 5 else ACTIONS
    policy_zip = sys.argv[6] if len(sys.argv) > 6 else None
    seeds = list(range(base, base + n))
    chunks = [(seeds[i::workers * 4], n_actions, policy_zip) for i in range(workers * 4)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(work, chunks)
    parts = [p for p in parts if len(p["s0"])]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(out, **data)
    print({k: v.shape for k, v in data.items()})
