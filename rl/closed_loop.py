"""Closed-loop fine-tuning of a model-based policy (`rl.model_based`).

The policy acts K times. Before each step it sees the current state and the
*remaining* goal -- the instruction minus the progress so far, with unmentioned
classes that drifted past KEEP_TOLERANCE turned into "undo" targets -- in the
one-shot observation layout, so a one-shot checkpoint is the starting point and
step 1 is still a plain one-shot policy. Training backpropagates the goal
distance through a K-step rollout in the frozen dynamics ensemble (final step
weight 1, earlier steps `intermediate_weight`).

On task v2 the first step of a K=2 fine-tune is the best pure policy so far
(docs/experiments/2026-10-17-task-v2.md); evaluation re-renders after every step.

    .venv/bin/python -m rl.closed_loop --init out/rl_mb_v2/r2_seed0/checkpoint.pt \\
        --data out/rl_mb_v2/data_r0b.npz,out/rl_mb_v2/data_r1.npz,out/rl_mb_v2/data_r2.npz \\
        --out out/rl_mb_v2/cl2_r2_seed0 --task-version v2
"""
import argparse
import collections
import json
import os

import numpy as np
import torch

import goals
from rl import model_based as mb
from rl.baselines import apply_controllable

N = mb.N_CLASSES


def remaining_goal(goal, delta, weights=None):
    """Batched: what is left of `goal` (B x 4n) after the cumulative change `delta`
    (B x STATE_SIZE) from the start. Mentioned targets shrink by the progress;
    an unmentioned class that drifted past KEEP_TOLERANCE gets an undo target;
    classes with weight 0 (task v2 occluders) stay free."""
    tol = goals.KEEP_TOLERANCE
    c, b = delta[:, :N], delta[:, N + 1:2 * N + 1]
    d, m, e, mb_ = goal[:, :N], goal[:, N:2 * N], goal[:, 2 * N:3 * N], goal[:, 3 * N:]
    drifted = (m == 0) & (c.abs() > tol)
    new_d = torch.where(m > 0, d - c, torch.where(drifted, -c, torch.zeros_like(d)))
    if weights is not None:
        new_d = torch.where(weights[:, :N] == 0, torch.zeros_like(d), new_d)
    new_m = torch.where(drifted, torch.ones_like(m), m)
    new_e = torch.where(mb_ > 0, e - b, torch.zeros_like(e))
    return torch.cat([new_d, new_m, new_e, mb_], 1)


def remaining_goal_for(goal, start_agg, current_agg, weights=None) -> np.ndarray:
    """`remaining_goal` for one real episode."""
    delta = torch.as_tensor(mb.state_vector(current_agg) - mb.state_vector(start_agg))[None]
    w = None
    if weights:
        w = torch.tensor([[weights.get(c, 1.0) for c in goals.GOAL_CLASSES]], dtype=torch.float32)
    return remaining_goal(torch.as_tensor(np.asarray(goal), dtype=torch.float32)[None], delta, w)[0].numpy()


def finetune(policy, ensemble, data, train_starts, steps: int = 20_000, horizon: int = 2, seed: int = 0,
             batch: int = 256, learning_rate: float = 1e-4, intermediate_weight: float = 0.2,
             balance_kinds: bool = False, log=print):
    """Train `policy` in place through `horizon`-step rollouts in the frozen `ensemble`.
    `balance_kinds` draws every instruction kind equally often instead of in the
    grammar's mix (absolute and brightness are 10 % each)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = mb._tensor
    goal_all, weights_all = T(data["goal"]), T(data["weights"])
    hist, solo, ctrl0, s0 = (T(data[k]) for k in ("hist", "solo", "ctrl", "s0"))
    optimizer = torch.optim.Adam(policy.net.parameters(), lr=learning_rate)
    policy.net.train()
    per_start = data["goal"].shape[1]
    pairs = np.array([(s, i) for s in train_starts for i in range(per_start)])
    p = None
    if balance_kinds:
        kinds = data["kind"][pairs[:, 0], pairs[:, 1]]
        names, inverse, counts = np.unique(kinds, return_inverse=True, return_counts=True)
        p = 1.0 / counts[inverse]
        p /= p.sum()
    model_attainment = None
    for step in range(steps):
        chosen = pairs[rng.choice(len(pairs), batch, p=p)]
        starts, pick = torch.as_tensor(chosen[:, 0]), torch.as_tensor(chosen[:, 1])
        goal, weights = goal_all[starts, pick], weights_all[starts, pick]
        start_distance = mb.goal_distance(goal, weights, torch.zeros(batch, mb.STATE_SIZE))
        keep = start_distance > 1e-6
        ctrl, state = ctrl0[starts], s0[starts]
        loss = penalty = 0.0
        for t in range(horizon):
            obs = mb.policy_observation(remaining_goal(goal, state - s0[starts], weights),
                                        hist[starts], state, solo[starts], ctrl)
            action = (ctrl + policy.action_tensor(obs)).clamp(-1, 1) - ctrl
            pred = ensemble(torch.cat([hist[starts], solo[starts], ctrl, state], 1), action)
            mean = pred.mean(0)
            state, ctrl = state + mean[:, :mb.STATE_SIZE], ctrl + action
            normalised = mb.goal_distance(goal, weights, state - s0[starts]) / start_distance.clamp_min(1e-6)
            loss = loss + (1.0 if t == horizon - 1 else intermediate_weight) * normalised.clamp(max=3.0)
            penalty = penalty + pred[:, :, :mb.STATE_SIZE].std(0).mean(1) + torch.sigmoid(mean[:, mb.STATE_SIZE])
        total = (loss + penalty)[keep].mean()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        if step % 1000 == 999:
            model_attainment = float((1 - normalised[keep]).median().detach())
            log(f"[closed_loop] step {step + 1}: model-predicted final attainment {model_attainment:+.3f}")
    policy.net.eval()
    return policy, {"model_predicted_median": model_attainment}


def evaluate(policy, horizon: int = 2, episodes: int = 200, seed_base: int = mb.VALIDATION_SEED_BASE,
             split: str = "val", task_version: str = "v1", volume_ids=None, model_for_volume=None) -> dict:
    """The policy alone on real renders, re-rendering and re-observing after each step.
    Medians are per kind and per step ("t1" is the one-shot score)."""
    import datasets
    from rl.oneshot_env import OneShotEnv, build_observation
    ids = volume_ids if volume_ids is not None else datasets.volumes_for_split(split)
    kwargs = {} if model_for_volume is None else {"model_for_volume": model_for_volume}
    env = OneShotEnv(ids, **kwargs, **mb.ENV_KWARGS, task_version=task_version)
    by_kind, rows = collections.defaultdict(lambda: collections.defaultdict(list)), []
    for seed in range(seed_base, seed_base + episodes):
        env.reset(seed=seed)
        model, instruction, start_agg = env._model, env._instruction, env._start_agg
        solo = env._solo_max_log(env._volume, model)
        params, agg = env._start_params.copy(), start_agg
        row = {"seed": seed, "kind": instruction["kind"], "solvable": env._solvable}
        for t in range(1, horizon + 1):
            ctrl = np.asarray(env._controllable_values(params))
            goal = remaining_goal_for(instruction["goal"], start_agg, agg, instruction.get("weights"))
            action, _ = policy.predict(build_observation(goal, model.histogram, agg, solo, ctrl))
            params = apply_controllable(params, np.clip(ctrl + action, -1, 1))
            agg = goals.aggregate(goals.features(model, params))
            row[f"t{t}"] = env._attainment(agg)
            by_kind[row["kind"]][f"t{t}"].append(row[f"t{t}"])
            by_kind["ALL"][f"t{t}"].append(row[f"t{t}"])
        rows.append(row)
    medians = {k: {t: float(np.median(v)) for t, v in steps.items()} for k, steps in sorted(by_kind.items())}
    return {"medians": medians, "rows": rows}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", required=True, help="model-based checkpoint.pt (policy and dynamics)")
    parser.add_argument("--data", required=True, help="comma-separated .npz files from `rl.model_based collect`")
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--intermediate-weight", type=float, default=0.2)
    parser.add_argument("--balance-kinds", action="store_true",
                        help="draw every instruction kind equally often")
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--task-version", choices=goals.TASK_VERSIONS, default="v1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    data = mb.load_data(args.data.split(","))
    train_starts, _ = mb.split_starts(len(data["s0"]))
    ensemble = mb.load_dynamics(args.init)
    for parameter in ensemble.parameters():
        parameter.requires_grad_(False)
    policy = mb.ModelBasedPolicy.load(args.init)
    policy, report = finetune(policy, ensemble, data, train_starts, steps=args.steps, horizon=args.horizon,
                              seed=args.seed, learning_rate=args.learning_rate,
                              intermediate_weight=args.intermediate_weight,
                              balance_kinds=args.balance_kinds)
    policy.metadata = {"init": args.init, "data": args.data.split(","), "closed_loop_horizon": args.horizon,
                       "train_steps": args.steps, "seed": args.seed, "learning_rate": args.learning_rate,
                       "intermediate_weight": args.intermediate_weight,
                       "balance_kinds": args.balance_kinds, "task_version": args.task_version}
    os.makedirs(args.out, exist_ok=True)
    if args.eval_episodes:
        result = evaluate(policy, horizon=args.horizon, episodes=args.eval_episodes,
                          task_version=args.task_version)
        report["val"] = result["medians"]
        for kind, steps in result["medians"].items():
            print(f"[closed_loop] {kind:10s} " + "  ".join(f"{t} {v:+.3f}" for t, v in steps.items()))
        with open(os.path.join(args.out, "eval_closed_loop.json"), "w") as stream:
            json.dump(result, stream, indent=1)
    print(f"[closed_loop] wrote {policy.save(args.out, ensemble, report)}")


if __name__ == "__main__":
    main()
