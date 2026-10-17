"""Closed-loop model-based policy: K steps, trained by backprop through a K-step rollout in
the learned dynamics. At step t the policy sees the current state and the *remaining*
goal (instruction minus progress so far; unmentioned classes drifted past
KEEP_TOLERANCE become "undo" targets) in the one-shot observation layout, so it can
start from a one-shot checkpoint. Loss = final normalised goal distance measured from
the episode start (+ small weight on intermediate steps). Evaluated in the real env:
each step re-renders and re-observes.

usage: mb_multistep.py OUT DYNAMICS_AND_INIT.pt DATA1,DATA2,... K [steps] [seed]
"""
import collections, json, os, sys
import numpy as np, torch
sys.path.insert(0, ".")
import goals
from rl import model_based as mb
from rl.baselines import apply_controllable

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))
N = mb.N_CLASSES
TOL = goals.KEEP_TOLERANCE


CLAMP = os.environ.get("CLAMP_GOAL", "0") == "1"
LOG_FLOOR = float(np.log10(goals.EPSILON))
LOG_CEIL = float(np.log10(1.0 + goals.EPSILON))


TASK_VERSION = os.environ.get("TASK_VERSION", "v1")


def remaining_goal_t(goal, delta, state=None, weights=None):
    """Torch twin of the closed-loop goal rebasing; `delta` is cumulative change from the start.
    With CLAMP_GOAL=1 and `state`, a remaining visibility change is clipped to what the current
    log10 visibility can still move (not below the EPSILON floor, not above full visibility)."""
    c, b = delta[:, :N], delta[:, N + 1:2 * N + 1]
    d, m, e, mb_ = goal[:, :N], goal[:, N:2 * N], goal[:, 2 * N:3 * N], goal[:, 3 * N:]
    drifted = (m == 0) & (c.abs() > TOL)
    new_d = torch.where(m > 0, d - c, torch.where(drifted, -c, torch.zeros_like(d)))
    if weights is not None:
        # task v2 occluders ("mentioned with weight 0") stay free: no undo target
        free = weights[:, :N] == 0
        new_d = torch.where(free, torch.zeros_like(d), new_d)
    if CLAMP and state is not None:
        current = state[:, :N]
        new_d = torch.maximum(torch.minimum(new_d, LOG_CEIL - current), LOG_FLOOR - current)
    new_m = torch.where(drifted, torch.ones_like(m), m)
    new_e = torch.where(mb_ > 0, e - b, torch.zeros_like(e))
    return torch.cat([new_d, new_m, new_e, mb_], 1)


def remaining_goal_np(goal, start_agg, cur_agg, weights=None):
    cur = torch.as_tensor(mb.state_vector(cur_agg))[None]
    delta = cur - torch.as_tensor(mb.state_vector(start_agg))[None]
    w = None
    if weights:
        w = torch.tensor([[weights.get(c, 1.0) for c in goals.GOAL_CLASSES]], dtype=torch.float32)
    return remaining_goal_t(torch.as_tensor(np.asarray(goal), dtype=torch.float32)[None], delta, cur, w)[0].numpy()


def train(policy, ensemble, data, train_starts, K, steps, seed, batch=256, lr=1e-4, inter_w=0.2):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = mb._tensor
    goal_all, w_all = T(data["goal"]), T(data["weights"])
    hist, solo, ctrl0, s0 = (T(data[k]) for k in ("hist", "solo", "ctrl", "s0"))
    opt = torch.optim.Adam(policy.net.parameters(), lr=lr)
    policy.net.train()
    per = data["goal"].shape[1]
    for step in range(steps):
        si = torch.as_tensor(rng.choice(train_starts, batch)); gi = torch.as_tensor(rng.integers(0, per, batch))
        goal, w = goal_all[si, gi], w_all[si, gi]
        d0 = mb.goal_distance(goal, w, torch.zeros(batch, mb.STATE_SIZE)); keep = d0 > 1e-6
        ctrl, state = ctrl0[si], s0[si]
        loss, disagree_sum, useless_sum = 0.0, 0.0, 0.0
        for t in range(K):
            delta = state - s0[si]
            obs = mb.policy_observation(remaining_goal_t(goal, delta, state, w), hist[si], state, solo[si], ctrl)
            a = (ctrl + policy.action_tensor(obs)).clamp(-1, 1) - ctrl
            ctx = torch.cat([hist[si], solo[si], ctrl, state], 1)
            pred = ensemble(ctx, a)
            mean = pred.mean(0)
            state = state + mean[:, :mb.STATE_SIZE]
            ctrl = ctrl + a
            dist = mb.goal_distance(goal, w, state - s0[si]) / d0.clamp_min(1e-6)
            weight = 1.0 if t == K - 1 else inter_w
            loss = loss + weight * dist.clamp(max=3.0)
            disagree_sum = disagree_sum + pred[:, :, :mb.STATE_SIZE].std(0).mean(1)
            useless_sum = useless_sum + torch.sigmoid(mean[:, mb.STATE_SIZE])
        total = (loss + disagree_sum + useless_sum)[keep].mean()
        opt.zero_grad(); total.backward(); opt.step()
        if step % 1000 == 999:
            print(f"step {step+1}: model final attainment median {float((1 - dist[keep]).median().detach()):+.3f}", flush=True)
    policy.net.eval()
    return policy


def evaluate_closed_loop(policy, K, episodes=200):
    from rl.oneshot_env import OneShotEnv, build_observation
    import datasets
    env = OneShotEnv(datasets.volumes_for_split("val"), **mb.ENV_KWARGS, task_version=TASK_VERSION)
    res = collections.defaultdict(lambda: collections.defaultdict(list)); rows = []
    for seed in range(mb.VALIDATION_SEED_BASE, mb.VALIDATION_SEED_BASE + episodes):
        env.reset(seed=seed)
        model, goal, start_agg = env._model, env._instruction["goal"], env._start_agg
        solo = env._solo_max_log(env._volume, model)
        params, agg = env._start_params.copy(), start_agg
        row = {"seed": seed, "kind": env._instruction["kind"], "solvable": env._solvable}
        for t in range(1, K + 1):
            ctrl = np.asarray(env._controllable_values(params))
            obs = build_observation(remaining_goal_np(goal, start_agg, agg, env._instruction.get("weights")),
                                    model.histogram, agg, solo, ctrl)
            a, _ = policy.predict(np.clip(obs, -10, 10))
            params = apply_controllable(params, np.clip(ctrl + a, -1, 1))
            agg = goals.aggregate(goals.features(model, params))
            row[f"t{t}"] = env._attainment(agg)
            res[row["kind"]][f"t{t}"].append(row[f"t{t}"]); res["ALL"][f"t{t}"].append(row[f"t{t}"])
        rows.append(row)
    summary = {k: {t: float(np.median(v)) for t, v in by.items()} for k, by in res.items()}
    for k in sorted(summary):
        print(f"  {k:10s} " + "  ".join(f"{t} {v:+.3f}" for t, v in sorted(summary[k].items())), flush=True)
    return summary, rows


if __name__ == "__main__":
    out, init, data_paths, K = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    data = mb.load_data(data_paths)
    train_starts, _ = mb.split_starts(len(data["s0"]))
    ensemble = mb.load_dynamics(init)
    for p in ensemble.parameters():
        p.requires_grad_(False)
    policy = mb.ModelBasedPolicy.load(init)
    print("before (one-shot policy, closed loop):", flush=True)
    evaluate_closed_loop(policy, K, episodes=int(os.environ.get("EVAL_BEFORE", "0")) or 1) if os.environ.get("EVAL_BEFORE") else None
    train(policy, ensemble, data, train_starts, K, steps, seed)
    print(f"after {K}-step training:", flush=True)
    summary, rows = evaluate_closed_loop(policy, K)
    os.makedirs(out, exist_ok=True)
    policy.metadata = {"closed_loop_steps": K, "init": init, "train_steps": steps, "seed": seed,
                       "task_version": TASK_VERSION}
    policy.save(out, ensemble, {"closed_loop": summary})
    json.dump({"summary": summary, "rows": rows}, open(f"{out}/eval_closed_loop.json", "w"), indent=1)
