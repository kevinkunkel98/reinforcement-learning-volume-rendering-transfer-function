"""Behaviour-cloning warm start for one-shot SAC, from hindsight episodes.

A one-shot episode is a contextual bandit: SAC has to learn a reward model
(the critic) from scratch before the actor gets a useful gradient, and on the
eight-class task 50k steps were not enough for that. Hindsight episodes
(`OneShotEnv._sample_hindsight_goal`) come with the action that produced their
target, so they are free supervised data: fit the actor to those actions
first, then let SAC fine-tune on the real reward.

Two uses of the same samples:

- `pretrain_actor`: regress the actor's deterministic action onto the oracle
  action (mean squared error through SB3's own tanh-squashed actor).
- `prefill_replay_buffer`: add the oracle transitions, with their true
  reward, to SAC's replay buffer, so the critic's first updates see good
  actions and not only the random warm-up.

Collection is the slow part (every episode renders its start, up to
`HINDSIGHT_MAX_TRIES` candidate targets and the oracle's result), so it runs
across processes and the result can be cached with `save_dataset`.
"""
import json
import multiprocessing

import numpy as np
import torch

from rl.oneshot_env import OneShotEnv

# Collection seeds live far from training (SB3's own) and validation
# (`rl.oneshot_train.VALIDATION_SEED_BASE`, 10_000 + i) seeds.
PRETRAIN_SEED_BASE = 1_000_000
SEED_STRIDE = 100_000
# atanh(0.97) ~ 2.1: targets at the edge of [-1, 1] would ask the squashed
# actor for an unbounded pre-tanh mean.
TARGET_CLIP = 0.97


def episode_seed(seed: int, index: int) -> int:
    return PRETRAIN_SEED_BASE + seed * SEED_STRIDE + index


def _collect_chunk(job) -> list:
    volume_ids, contract, seed, indices = job
    torch.set_num_threads(1)
    env = OneShotEnv(list(volume_ids), hindsight_ratio=1.0, **contract)
    rows = []
    for index in indices:
        obs, _ = env.reset(seed=episode_seed(seed, index))
        action = env.hindsight_policy_action()
        _obs, reward, *_ = env.step(action)
        rows.append((index, obs, action, float(reward)))
    return rows


def collect_hindsight(volume_ids, contract: dict, count: int, seed: int = 0,
                      workers: int = 1) -> dict:
    """`count` hindsight episodes on `volume_ids` under the env `contract`
    (policy_version/action_mode/reward_mode): {"obs", "actions", "rewards"}.
    Each episode is seeded by its index alone, so the result does not depend
    on `workers`."""
    indices = list(range(count))
    chunks = [indices[i::workers] for i in range(workers)] if workers > 1 else [indices]
    jobs = [(tuple(volume_ids), dict(contract), seed, chunk) for chunk in chunks if chunk]
    if len(jobs) == 1:
        rows = _collect_chunk(jobs[0])
    else:
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            rows = [row for chunk in pool.map(_collect_chunk, jobs) for row in chunk]
    rows.sort(key=lambda row: row[0])
    return {"obs": np.stack([row[1] for row in rows]).astype(np.float32),
            "actions": np.stack([row[2] for row in rows]).astype(np.float32),
            "rewards": np.asarray([row[3] for row in rows], dtype=np.float32)}


def save_dataset(data: dict, path: str, metadata: dict) -> None:
    np.savez_compressed(path, **data, metadata=np.array(json.dumps(metadata)))


def load_dataset(path: str) -> tuple:
    with np.load(path) as stored:
        data = {key: stored[key] for key in ("obs", "actions", "rewards")}
        metadata = json.loads(str(stored["metadata"]))
    return data, metadata


def pretrain_actor(model, data: dict, epochs: int = 30, batch_size: int = 256,
                   learning_rate: float = 1e-3, val_fraction: float = 0.1,
                   seed: int = 0) -> dict:
    """Fit `model.actor`'s deterministic action to `data["actions"]`.
    Returns train/validation MSE and the MSE of always predicting 0 (doing
    nothing, in residual mode) on the same validation split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data["obs"]))
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    obs = model.policy.obs_to_tensor(data["obs"])[0].float()
    targets = torch.as_tensor(np.clip(data["actions"], -TARGET_CLIP, TARGET_CLIP),
                              dtype=torch.float32, device=model.device)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)

    model.actor.set_training_mode(True)
    train_loss = float("nan")
    for _epoch in range(epochs):
        shuffled = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(shuffled), batch_size):
            batch = torch.as_tensor(shuffled[start:start + batch_size], device=model.device)
            predicted = model.actor(obs[batch], deterministic=True)
            loss = torch.nn.functional.mse_loss(predicted, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))
    model.actor.set_training_mode(False)

    with torch.no_grad():
        val_targets = targets[torch.as_tensor(val_idx, device=model.device)]
        predicted = model.actor(obs[torch.as_tensor(val_idx, device=model.device)], deterministic=True)
        val_loss = float(torch.nn.functional.mse_loss(predicted, val_targets))
        zero_loss = float((val_targets ** 2).mean())
    return {"train_mse": train_loss, "val_mse": val_loss, "zero_mse": zero_loss,
            "samples": int(len(order)), "epochs": epochs}


def prefill_replay_buffer(model, data: dict) -> None:
    """Add every sample as a terminal one-step transition with its reward."""
    for obs, action, reward in zip(data["obs"], data["actions"], data["rewards"]):
        model.replay_buffer.add(obs[None], obs[None], action[None],
                                np.asarray([reward], dtype=np.float32),
                                np.asarray([True]), [{}])
