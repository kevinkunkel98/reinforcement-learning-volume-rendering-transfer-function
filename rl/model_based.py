"""Model-based RL for the one-shot task: learn what a transfer-function change
does to the measured visibility, then train the policy through that model with
the exact goal metric.

Why not SAC: model-free SAC on `OneShotEnv` (E3-E9, docs/experiments/) stays at
a median attainment of about 0. Its critic has to learn instruction x physics
jointly from a scalar reward. Here the two are split:

1. **Dynamics.** An ensemble of MLPs maps (volume histogram, solo_max ceilings,
   start controllable values, start state, residual action) to the change in
   `state_vector` -- per-class log10 visibility (plus "other"), brightness and
   coverage -- and a "useless" logit. The renderer's physics does not depend on
   the instruction, so random TF changes on training-split starts teach it
   (`collect`).
2. **Policy.** An MLP with the one-shot observation layout
   (`rl.oneshot_env.build_observation`) and residual actions, trained by
   backpropagating `goal_distance` -- a torch twin of `goals.distance` -- through
   the frozen ensemble (SVG(0)/Dreamer-style actor training with a known
   reward). An ensemble-disagreement penalty keeps it where the model is sure.
3. **Dyna.** Rendering the policy's own proposals and refitting the model
   closes most of the model-exploitation gap (`collect --policy`).

`ModelBasedPolicy` has SB3's `predict` and space attributes, so `policy.py`
(the viewer) and `rl.vis_eval` run it like a SAC checkpoint.

    .venv/bin/python -m rl.model_based collect --out out/rl_mb/data_r0.npz --starts 3000
    .venv/bin/python -m rl.model_based train --data out/rl_mb/data_r0.npz --out out/rl_mb/r0_seed0
    .venv/bin/python -m rl.model_based collect --out out/rl_mb/data_r1.npz --starts 1500 \\
        --seed-base 2000000 --policy out/rl_mb/r0_seed0/checkpoint.pt
    .venv/bin/python -m rl.model_based train --data out/rl_mb/data_r0.npz,out/rl_mb/data_r1.npz \\
        --out out/rl_mb/r1_seed0
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn

import goals
from rl.baselines import CONTROLLABLE, apply_controllable
from rl.oneshot_env import (ACTION_SIZE, OBSERVATION_BOUND, OBSERVATION_SIZE, V9_POLICY_VERSION,
                            observation_metadata)

N_CLASSES = len(goals.GOAL_CLASSES)
HISTOGRAM_SIZE = 16
# log10 visibility per goal class + "other", brightness per goal class, coverage
STATE_SIZE = (N_CLASSES + 1) + N_CLASSES + 1
CONTEXT_SIZE = HISTOGRAM_SIZE + N_CLASSES + ACTION_SIZE + STATE_SIZE
CHECKPOINT_NAME = "checkpoint.pt"
METADATA_NAME = "metadata.json"
ENV_KWARGS = {"policy_version": V9_POLICY_VERSION, "action_mode": "residual", "reward_mode": "target"}

# Random actions for the dynamics data: dense small moves, sparse large moves of
# 1-4 groups (one organ's peak, say), and no-ops -- the policy's proposals fall
# in all three regimes.
DENSE_SHARE, SPARSE_SHARE = 0.4, 0.5
ACTIONS_PER_START = 20
INSTRUCTIONS_PER_START = 8
POLICY_NOISE_MAX = 0.3
VALIDATION_SEED_BASE = 50_000


def state_vector(agg: dict) -> np.ndarray:
    """An aggregated feature dict (`goals.aggregate`) as `STATE_SIZE` floats,
    visibility in the log10 units `goals.progress` measures change in."""
    log_vis = [math.log10(agg["vis"].get(c, 0.0) + goals.EPSILON)
               for c in list(goals.GOAL_CLASSES) + ["other"]]
    bright = [agg["bright"][c] for c in goals.GOAL_CLASSES]
    return np.array(log_vis + bright + [agg["coverage"]], dtype=np.float32)


def mlp(inputs: int, outputs: int, hidden: int = 256, depth: int = 3) -> nn.Sequential:
    layers, width = [], inputs
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    return nn.Sequential(*layers, nn.Linear(width, outputs))


class DynamicsEnsemble(nn.Module):
    """K MLPs: (context, residual action) -> (delta state, useless logit), K x B x (STATE_SIZE + 1)."""

    def __init__(self, members: int = 5):
        super().__init__()
        self.nets = nn.ModuleList(mlp(CONTEXT_SIZE + ACTION_SIZE, STATE_SIZE + 1) for _ in range(members))
        self.register_buffer("mean", torch.zeros(CONTEXT_SIZE + ACTION_SIZE))
        self.register_buffer("scale", torch.ones(CONTEXT_SIZE + ACTION_SIZE))

    def forward(self, context, action):
        x = (torch.cat([context, action], -1) - self.mean) / self.scale
        return torch.stack([net(x) for net in self.nets])


def goal_distance(goal, weights, delta):
    """`goals.distance` in torch, batched: `goal` B x 4n, `weights` B x n or
    B x (n + 1) (1.0 where the instruction has none; the optional last column is
    weights["other"], task v2's occlusion exemption), `delta` B x STATE_SIZE (the
    change from the start state, so `delta = 0` is the start's own distance)."""
    n = N_CLASSES
    other_weight = weights[:, n] if weights.shape[1] > n else 1.0
    weights = weights[:, :n]
    c, c_other, b = delta[:, :n], delta[:, n], delta[:, n + 1:2 * n + 1]
    d, m, e, mb = goal[:, :n], goal[:, n:2 * n], goal[:, 2 * n:3 * n], goal[:, 3 * n:4 * n]
    total = (m * weights * (c - d).abs()).sum(1) + goals.KAPPA * (mb * weights * (b - e).abs()).sum(1)
    total = total + goals.LAMBDA_KEEP * ((1 - m) * torch.relu(c.abs() - goals.KEEP_TOLERANCE)).sum(1)
    total = total + goals.LAMBDA_KEEP * goals.KAPPA * (
        (1 - mb) * torch.relu(b.abs() - goals.KEEP_TOLERANCE)).sum(1)
    return total + goals.LAMBDA_KEEP * other_weight * torch.relu(c_other.abs() - goals.KEEP_TOLERANCE)


def policy_observation(goal, histogram, start_state, solo_max_log, controllable):
    """`rl.oneshot_env.build_observation`'s layout from batched tensors."""
    n = N_CLASSES
    obs = torch.cat([goal, histogram, start_state[:, :n], start_state[:, n + 1:2 * n + 1],
                     solo_max_log, controllable, start_state[:, -1:]], -1)
    return obs.clamp(-OBSERVATION_BOUND, OBSERVATION_BOUND)


class _Space:
    def __init__(self, shape):
        self.shape = shape


class ModelBasedPolicy:
    """The trained actor behind SB3's interface (`predict`, `observation_space`,
    `action_space`): observation -> residual action in [-1, 1]."""

    def __init__(self, net: nn.Module, metadata: dict = None):
        self.net = net.eval()
        self.metadata = dict(metadata or {})
        self.observation_space = _Space((OBSERVATION_SIZE,))
        self.action_space = _Space((ACTION_SIZE,))

    @staticmethod
    def new_net(hidden: int = 256, depth: int = 2) -> nn.Module:
        return mlp(OBSERVATION_SIZE, ACTION_SIZE, hidden=hidden, depth=depth)

    def action_tensor(self, observation):
        return torch.tanh(self.net(observation.clamp(-OBSERVATION_BOUND, OBSERVATION_BOUND)))

    def predict(self, observation, deterministic: bool = True):
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32))
        with torch.no_grad():
            action = self.action_tensor(obs.reshape(-1, OBSERVATION_SIZE)).numpy()
        return (action[0] if np.ndim(observation) == 1 else action), None

    def save(self, out_dir: str, dynamics: DynamicsEnsemble = None, report: dict = None) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, CHECKPOINT_NAME)
        torch.save({"policy": self.net.state_dict(),
                    "arch": {"hidden": self.net[0].out_features, "depth": (len(self.net) - 1) // 2},
                    "dynamics": None if dynamics is None else dynamics.state_dict(),
                    "metadata": self.metadata}, path)
        # rl.oneshot_env.load_policy_metadata reads this, as for a SAC run.
        with open(os.path.join(out_dir, METADATA_NAME), "w") as stream:
            task_version = self.metadata.get("task_version", "v1")
            json.dump({**observation_metadata(**ENV_KWARGS, task_version=task_version), "algorithm": "model_based",
                       **self.metadata, **({"report": report} if report else {})}, stream, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ModelBasedPolicy":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        net = cls.new_net(**payload.get("arch", {}))
        net.load_state_dict(payload["policy"])
        return cls(net, payload.get("metadata"))


def load_dynamics(path: str) -> DynamicsEnsemble:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ensemble = DynamicsEnsemble()
    ensemble.load_state_dict(payload["dynamics"])
    return ensemble.eval()


# --- data -------------------------------------------------------------------

def _collect_chunk(args):
    seeds, actions_per_start, policy_path, volume_ids, start_variety, task_version = args
    torch.set_num_threads(1)
    from rl.oneshot_train import build_env
    env = build_env(volume_ids=volume_ids, **ENV_KWARGS, balance_classes=True, start_variety=start_variety,
                    task_version=task_version)
    policy = ModelBasedPolicy.load(policy_path) if policy_path else None
    return collect_starts(env, seeds, actions_per_start, policy)


def collect_starts(env, seeds, actions_per_start: int = ACTIONS_PER_START, policy=None) -> dict:
    """For each seeded start of `env` (a `OneShotEnv`): render `actions_per_start`
    TF changes and record the resulting `state_vector`, and sample
    `INSTRUCTIONS_PER_START` real instructions for policy training. With
    `policy`, the first half of the actions are its proposals plus noise
    (Dyna: data where the policy actually acts)."""
    out = {k: [] for k in ("hist", "solo", "ctrl", "s0", "act", "s1", "useless", "goal", "weights", "kind",
                           "volume")}
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        ctrl = np.asarray(env._controllable_values(env._start_params))
        actions, states, useless = [], [], []
        for j in range(actions_per_start):
            draw = rng.random()
            if policy is not None and j < actions_per_start // 2:
                action, _ = policy.predict(obs)
                action = action + rng.normal(0.0, rng.uniform(0.0, POLICY_NOISE_MAX), ACTION_SIZE)
            elif draw < DENSE_SHARE:
                action = rng.uniform(-1, 1, ACTION_SIZE) * rng.uniform(0.05, 0.8)
            elif draw < DENSE_SHARE + SPARSE_SHARE:
                action = np.zeros(ACTION_SIZE)
                picked = rng.choice(ACTION_SIZE, rng.integers(1, 5), replace=False)
                action[picked] = rng.uniform(-1, 1, len(picked))
            else:
                action = np.zeros(ACTION_SIZE)
            action = np.clip(ctrl + np.clip(action, -1, 1), -1, 1) - ctrl   # the residual actually applied
            raw = goals.features(env._model, apply_controllable(env._start_params, ctrl + action))
            actions.append(action)
            states.append(state_vector(goals.aggregate(raw)))
            useless.append(goals.is_useless(raw))
        goal_rows, weight_rows, kinds = [], [], []
        for _ in range(INSTRUCTIONS_PER_START):
            instruction = env.sample_real_instruction(rng, env._volume, env._model, env._start_params,
                                                      env._start_agg)
            weights = instruction.get("weights") or {}
            goal_rows.append(instruction["goal"])
            weight_rows.append([weights.get(c, 1.0) for c in goals.GOAL_CLASSES] + [weights.get("other", 1.0)])
            kinds.append(instruction["kind"])
        out["hist"].append(np.asarray(env._model.histogram, np.float32))
        out["solo"].append(np.asarray(env._solo_max_log(env._volume, env._model), np.float32))
        out["ctrl"].append(ctrl.astype(np.float32))
        out["s0"].append(state_vector(env._start_agg))
        out["act"].append(np.asarray(actions, np.float32))
        out["s1"].append(np.asarray(states, np.float32))
        out["useless"].append(np.asarray(useless, bool))
        out["goal"].append(np.asarray(goal_rows, np.float32))
        out["weights"].append(np.asarray(weight_rows, np.float32))
        out["kind"].append(kinds)
        out["volume"].append(env._volume)
    return {k: np.array(v) for k, v in out.items()}


def collect(seeds, workers: int = 3, actions_per_start: int = ACTIONS_PER_START,
            policy_path: str = None, volume_ids=None, start_variety: float = 0.0,
            task_version: str = "v1") -> dict:
    """`collect_starts` over `workers` processes. `start_variety` is
    `OneShotEnv`'s: that share of starts dims soft tissue and raises the organs,
    so the model also sees states where organs are visible."""
    seeds = list(seeds)
    chunks = [(seeds[i::workers * 4], actions_per_start, policy_path, volume_ids, start_variety, task_version)
              for i in range(workers * 4)]
    chunks = [chunk for chunk in chunks if chunk[0]]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_collect_chunk, chunks)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load_data(paths) -> dict:
    """Concatenate `collect` outputs; keys only some files carry (older files
    lack "volume") are dropped, and weights are padded to the n + 1 column
    layout ("other" = 1.0) when files differ."""
    parts = [dict(np.load(p, allow_pickle=True)) for p in paths]
    if len({p["weights"].shape[-1] for p in parts}) > 1:
        for p in parts:
            if p["weights"].shape[-1] == N_CLASSES:
                p["weights"] = np.concatenate([p["weights"], np.ones(p["weights"].shape[:-1] + (1,), np.float32)], -1)
    keys = set.intersection(*(set(p) for p in parts))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k in keys}


# --- training -----------------------------------------------------------------

def _tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _context(data) -> torch.Tensor:
    return _tensor(np.concatenate([data["hist"], data["solo"], data["ctrl"], data["s0"]], 1))


def _transitions(data, starts, context):
    per_start = data["act"].shape[1]
    return (context[starts].repeat_interleave(per_start, 0),
            _tensor(data["act"][starts].reshape(-1, ACTION_SIZE)),
            _tensor((data["s1"][starts] - data["s0"][starts][:, None]).reshape(-1, STATE_SIZE)),
            _tensor(data["useless"][starts].reshape(-1).astype(np.float32)))


def split_starts(n_starts: int, validation_share: float = 0.1):
    """A fixed (seed-independent) held-out share of starts for the model's own validation."""
    perm = np.random.default_rng(0).permutation(n_starts)
    cut = max(1, int(n_starts * validation_share)) if n_starts > 1 else 0
    return perm[cut:], perm[:cut]


def train_dynamics(data, train_starts, val_starts, epochs: int = 60, seed: int = 0, log=print):
    torch.manual_seed(seed)
    context = _context(data)
    ctx, act, delta, useless = _transitions(data, train_starts, context)
    ensemble = DynamicsEnsemble()
    inputs = torch.cat([ctx, act], -1)
    ensemble.mean.copy_(inputs.mean(0))
    ensemble.scale.copy_(inputs.std(0) + 1e-3)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    report = {}
    for epoch in range(epochs):
        ensemble.train()
        for k in range(len(ensemble.nets)):
            order = torch.randint(0, len(act), (len(act),))      # bootstrap per member
            for batch in torch.split(order, 512):
                pred = ensemble(ctx[batch], act[batch])[k]
                loss = (nn.functional.smooth_l1_loss(pred[:, :STATE_SIZE], delta[batch], beta=0.1)
                        + 0.1 * nn.functional.binary_cross_entropy_with_logits(pred[:, STATE_SIZE], useless[batch]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        schedule.step()
        if len(val_starts) and (epoch % 10 == 9 or epoch == epochs - 1):
            report = dynamics_report(ensemble, data, val_starts, context)
            log(f"[model_based] dynamics epoch {epoch + 1}: {report}")
    ensemble.eval()
    for parameter in ensemble.parameters():
        parameter.requires_grad_(False)
    return ensemble, report


def dynamics_report(ensemble, data, val_starts, context=None) -> dict:
    """Held-out fit, and -- the number that matters -- how well the model ranks
    real attainment for (start, action, instruction) triples it never saw."""
    context = _context(data) if context is None else context
    ctx, act, delta, _ = _transitions(data, val_starts, context)
    per_start = data["act"].shape[1]
    with torch.no_grad():
        pred = ensemble(ctx, act).mean(0)[:, :STATE_SIZE]
        r2 = 1 - ((pred - delta) ** 2).mean(0) / delta.var(0).clamp_min(1e-9)
        pick = torch.randint(0, data["goal"].shape[1], (len(act),), generator=torch.Generator().manual_seed(0))
        rows = torch.arange(len(act))
        goal = _tensor(data["goal"][val_starts]).repeat_interleave(per_start, 0)[rows, pick]
        weights = _tensor(data["weights"][val_starts]).repeat_interleave(per_start, 0)[rows, pick]
        start = goal_distance(goal, weights, torch.zeros_like(delta))
        keep = start > 1e-6
        real = (1 - goal_distance(goal, weights, delta) / start.clamp_min(1e-6))[keep].numpy()
        guess = (1 - goal_distance(goal, weights, pred) / start.clamp_min(1e-6))[keep].numpy()
    rank = lambda v: np.argsort(np.argsort(v))
    return {"r2_log_vis": round(float(r2[:N_CLASSES + 1].mean()), 3),
            "r2_bright": round(float(r2[N_CLASSES + 1:2 * N_CLASSES + 1].mean()), 3),
            "attainment_spearman": round(float(np.corrcoef(rank(real), rank(guess))[0, 1]), 3),
            "attainment_mae": round(float(np.abs(np.clip(real, -2, 1) - np.clip(guess, -2, 1)).mean()), 3)}


def train_policy(ensemble, data, train_starts, steps: int = 6000, beta: float = 1.0,
                 useless_weight: float = 1.0, seed: int = 0, batch: int = 256, hidden: int = 256,
                 depth: int = 2, learning_rate: float = 3e-4, cosine: bool = False, log=print):
    """Fit the actor by gradient descent on the model-predicted normalised goal
    distance (1 - attainment, capped at 3 so hopeless instructions do not
    dominate), plus `beta` x ensemble disagreement and `useless_weight` x the
    predicted probability of a useless render."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    context, goal_all, weights_all = _context(data), _tensor(data["goal"]), _tensor(data["weights"])
    hist, solo, ctrl, s0 = (_tensor(data[k]) for k in ("hist", "solo", "ctrl", "s0"))
    net = ModelBasedPolicy.new_net(hidden, depth)
    policy = ModelBasedPolicy(net.train())
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps) if cosine else None
    per_start = data["goal"].shape[1]
    model_attainment = None
    for step in range(steps):
        starts = torch.as_tensor(rng.choice(train_starts, batch))
        pick = torch.as_tensor(rng.integers(0, per_start, batch))
        goal, weights = goal_all[starts, pick], weights_all[starts, pick]
        start = goal_distance(goal, weights, torch.zeros(batch, STATE_SIZE))
        keep = start > 1e-6
        obs = policy_observation(goal, hist[starts], s0[starts], solo[starts], ctrl[starts])
        action = (ctrl[starts] + policy.action_tensor(obs)).clamp(-1, 1) - ctrl[starts]
        pred = ensemble(context[starts], action)
        mean = pred.mean(0)
        normalised = goal_distance(goal, weights, mean[:, :STATE_SIZE]) / start.clamp_min(1e-6)
        disagreement = pred[:, :, :STATE_SIZE].std(0).mean(1)
        loss = (normalised.clamp(max=3.0) + beta * disagreement
                + useless_weight * torch.sigmoid(mean[:, STATE_SIZE]))[keep].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if schedule is not None:
            schedule.step()
        if step % 1000 == 999:
            model_attainment = float((1 - normalised[keep]).median().detach())
            log(f"[model_based] policy step {step + 1}: model-predicted median attainment "
                f"{model_attainment:+.3f}, disagreement {float(disagreement.mean().detach()):.3f}")
    net.eval()
    return policy, {"model_predicted_median": model_attainment}


def evaluate(policy, episodes: int = 200, seed_base: int = VALIDATION_SEED_BASE, split: str = "val",
             volume_ids=None, model_for_volume=None, task_version: str = "v1") -> dict:
    """The policy alone on real renders: one `OneShotEnv` step per episode."""
    import datasets
    from rl.oneshot_env import OneShotEnv
    ids = volume_ids if volume_ids is not None else datasets.volumes_for_split(split)
    kwargs = {} if model_for_volume is None else {"model_for_volume": model_for_volume}
    env = OneShotEnv(ids, **kwargs, **ENV_KWARGS, task_version=task_version)
    rows, by_kind = [], collections.defaultdict(list)
    for seed in range(seed_base, seed_base + episodes):
        obs, _ = env.reset(seed=seed)
        action, _ = policy.predict(obs)
        _, _, _, _, info = env.step(action)
        rows.append({"seed": seed, "kind": info["kind"], "attainment": info["attainment"],
                     "solvable": info["solvable"], "useless": info["useless"]})
        by_kind[info["kind"]].append(info["attainment"])
        by_kind["ALL"].append(info["attainment"])
    return {"medians": {k: float(np.median(v)) for k, v in sorted(by_kind.items())},
            "useless_share": float(np.mean([r["useless"] for r in rows])), "rows": rows}


def run_training(data_paths, out: str, seed: int = 0, model_epochs: int = 60, policy_steps: int = 6000,
                 beta: float = 1.0, useless_weight: float = 1.0, eval_episodes: int = 200,
                 dynamics_path: str = None, hidden: int = 256, depth: int = 2,
                 learning_rate: float = 3e-4, cosine: bool = False, task_version: str = "v1",
                 log=print) -> dict:
    """Fit the dynamics ensemble (or reuse the one in `dynamics_path`, a
    checkpoint.pt trained on the same data), then the policy through it."""
    data = load_data(data_paths)
    train_starts, val_starts = split_starts(len(data["s0"]))
    t0 = time.time()
    if dynamics_path:
        ensemble = load_dynamics(dynamics_path)
        for parameter in ensemble.parameters():
            parameter.requires_grad_(False)
        dyn_report = {"reused": dynamics_path, **dynamics_report(ensemble, data, val_starts)}
        log(f"[model_based] reused dynamics: {dyn_report}")
    else:
        ensemble, dyn_report = train_dynamics(data, train_starts, val_starts, epochs=model_epochs,
                                              seed=seed, log=log)
    policy, pol_report = train_policy(ensemble, data, train_starts, steps=policy_steps, beta=beta,
                                      useless_weight=useless_weight, seed=seed, hidden=hidden,
                                      depth=depth, learning_rate=learning_rate, cosine=cosine, log=log)
    policy.metadata = {"seed": seed, "data": list(data_paths), "starts": int(len(data["s0"])),
                       "model_epochs": model_epochs, "policy_steps": policy_steps, "beta": beta,
                       "useless_weight": useless_weight, "hidden": hidden, "depth": depth,
                       "learning_rate": learning_rate, "cosine": cosine, "dynamics": dynamics_path,
                       "task_version": task_version}
    report = {"dynamics": dyn_report, "policy": pol_report, "train_seconds": round(time.time() - t0)}
    if eval_episodes:
        result = evaluate(policy, episodes=eval_episodes, task_version=task_version)
        report["val"] = {"medians": result["medians"], "useless_share": result["useless_share"]}
        log(f"[model_based] real val medians: {result['medians']}")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "eval_val.json"), "w") as stream:
            json.dump(result, stream, indent=1)
    path = policy.save(out, ensemble, report)
    log(f"[model_based] wrote {path}")
    return {"policy": policy, "dynamics": ensemble, "report": report, "path": path}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="render random (and policy) TF changes on training-split starts")
    c.add_argument("--out", required=True)
    c.add_argument("--starts", type=int, default=3000)
    c.add_argument("--seed-base", type=int, default=1_000_000)
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--actions", type=int, default=ACTIONS_PER_START)
    c.add_argument("--policy", default=None, help="checkpoint.pt whose proposals fill half the actions (Dyna)")
    c.add_argument("--splits", default="train",
                   help="comma-separated volume splits to draw starts from, e.g. train,train_extra")
    c.add_argument("--task-version", choices=goals.TASK_VERSIONS, default="v1")
    c.add_argument("--start-variety", type=float, default=0.0,
                   help="share of starts with soft tissue dimmed and organs raised (OneShotEnv)")
    t = sub.add_parser("train", help="fit the dynamics ensemble, then the policy through it")
    t.add_argument("--data", required=True, help="comma-separated .npz files from `collect`")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--model-epochs", type=int, default=60)
    t.add_argument("--policy-steps", type=int, default=6000)
    t.add_argument("--beta", type=float, default=1.0)
    t.add_argument("--useless-weight", type=float, default=1.0)
    t.add_argument("--eval-episodes", type=int, default=200)
    t.add_argument("--threads", type=int, default=3)
    t.add_argument("--dynamics", default=None, help="reuse this checkpoint.pt's dynamics ensemble")
    t.add_argument("--policy-hidden", type=int, default=256)
    t.add_argument("--policy-depth", type=int, default=2)
    t.add_argument("--policy-lr", type=float, default=3e-4)
    t.add_argument("--policy-cosine", action="store_true", help="cosine-anneal the policy learning rate")
    t.add_argument("--task-version", choices=goals.TASK_VERSIONS, default="v1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "collect":
        import datasets
        volume_ids = [name for split in args.splits.split(",") for name in datasets.volumes_for_split(split)]
        data = collect(range(args.seed_base, args.seed_base + args.starts), workers=args.workers,
                       volume_ids=volume_ids,
                       actions_per_start=args.actions, policy_path=args.policy,
                       start_variety=args.start_variety, task_version=args.task_version)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, **data)
        print(f"[model_based] wrote {args.out}: {len(data['s0'])} starts x {data['act'].shape[1]} actions")
    else:
        torch.set_num_threads(args.threads)
        run_training(args.data.split(","), args.out, seed=args.seed, model_epochs=args.model_epochs,
                     policy_steps=args.policy_steps, beta=args.beta, useless_weight=args.useless_weight,
                     eval_episodes=args.eval_episodes, dynamics_path=args.dynamics,
                     hidden=args.policy_hidden, depth=args.policy_depth, learning_rate=args.policy_lr,
                     cosine=args.policy_cosine, task_version=args.task_version)


if __name__ == "__main__":
    main()
