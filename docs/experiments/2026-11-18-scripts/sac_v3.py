"""Model-free SAC on task v3, for a fair comparison with the model-based policy.

The only SAC number so far (<= 0.09) is from task v1 (2026-10-16). This trains
SAC on exactly the task the thesis numbers use:

- the same kind of data as the model-based pipeline: 7500 seeded starts over
  train + train_extra (90 patients) with 8 class-balanced instructions each
  (`rl.model_based.collect_starts` draws the same), built once in parallel
  (`pool`), since sampling an instruction costs ~1 s and would otherwise
  dominate training; the same observation (`Episode.observation`) and
  residual action (24 controllable groups) as the model-based policy;
- reward = attainment of the answer, clipped to [-1, 1] (unbounded below
  otherwise; the evaluation reports medians, which the clip does not move);
- the shell the reported numbers were scored with (skeleton, lungs and soft
  peaks; changed on 2026-11-15);
- budget: the model-based pipeline rendered 7500 starts x 20 actions =
  150k real evaluations (data_r0..r3), so a checkpoint at 150k steps is the
  equal-budget SAC and the run continues to 300k (twice the budget).

Evaluation uses `rl.evaluate.evaluate_episode` on the fixed episodes with
planning off (SAC has no model): the SAC answer alone and a warm search of 30
and 200 evaluations from it, the same rows as "policy, then search".

    .venv/bin/python docs/experiments/2026-11-18-scripts/sac_v3.py pool --out out/sac_v3/pool.pkl --workers 10
    .venv/bin/python docs/experiments/2026-11-18-scripts/sac_v3.py train --pool out/sac_v3/pool.pkl --seed 0 --out out/sac_v3/seed0
    .venv/bin/python docs/experiments/2026-11-18-scripts/sac_v3.py eval out/sac_v3/seed0/steps_150000.zip --split test --out out/sac_v3/seed0_150k_test.json
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import transfer  # noqa: E402

transfer.UNLABELLED_PEAKS = ("skeleton", "lungs", "soft")  # the shell of the reported numbers

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

import scans  # noqa: E402
import goals  # noqa: E402
import visibility  # noqa: E402
from rl.episodes import ACTION_SIZE, OBSERVATION_BOUND, OBSERVATION_SIZE, Episode, EpisodeSampler, fixed_episodes  # noqa: E402
from rl.evaluate import evaluate_episode  # noqa: E402

TRAIN_SPLITS = ("train", "train_extra")
EQUAL_BUDGET = 150_000      # real evaluations the model-based pipeline used
SEED_BASE = 20_000_000      # far from every seed range the pipeline and the evaluation use
POOL_STARTS = 7500          # as data_r0..r3 (3000 + 3 x 1500)
INSTRUCTIONS_PER_START = 8  # as rl.model_based.INSTRUCTIONS_PER_START
REWARD_CLIP = (-1.0, 1.0)


def _pool_chunk(seeds) -> list:
    torch.set_num_threads(1)
    sampler = EpisodeSampler([v for s in TRAIN_SPLITS for v in scans.split_names(s)], balance_classes=True)
    out = []
    for seed in seeds:
        episode = sampler.reset(int(seed))
        rng = np.random.default_rng(int(seed))
        for _ in range(INSTRUCTIONS_PER_START):
            instruction = sampler.sample_instruction(rng, episode.volume, episode.model, episode.start_params,
                                                     episode.start_agg)
            out.append({"volume": episode.volume, "start_params": episode.start_params,
                        "start_agg": episode.start_agg, "instruction": instruction})
    return out


def build_pool(args):
    seeds = [SEED_BASE + i for i in range(POOL_STARTS)]
    chunks = [seeds[i::args.workers * 4] for i in range(args.workers * 4)]
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_pool_chunk, chunks)
    episodes = [e for part in parts for e in part]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as stream:
        pickle.dump(episodes, stream)
    kinds = {}
    for e in episodes:
        kinds[e["instruction"]["kind"]] = kinds.get(e["instruction"]["kind"], 0) + 1
    print(f"[pool] {len(episodes)} episodes in {time.time() - started:.0f} s -> {args.out}; kinds {kinds}", flush=True)


_MODELS = {}   # every training volume's visibility model, kept (~6 MB each): `visibility.for_volume`
              # re-reads the volume to check its version on every call (~50 ms), and its LRU holds 4


def episode_from(row) -> Episode:
    """An `Episode` from a pool row without re-measuring its start (one render saved per step)."""
    episode = Episode.__new__(Episode)
    episode.volume, episode.instruction = row["volume"], row["instruction"]
    if row["volume"] not in _MODELS:
        _MODELS[row["volume"]] = visibility.for_volume(row["volume"])
    episode.model = _MODELS[row["volume"]]
    episode.start_params = np.asarray(row["start_params"], dtype=np.float64)
    episode.start_agg = row["start_agg"]
    episode.start_distance = goals.distance(episode.instruction["goal"], episode.start_agg, episode.start_agg,
                                            episode.instruction.get("weights"))
    return episode


class OneShotV3(gym.Env):
    """One step per episode: observe the start and the instruction, answer
    with a residual action, get the answer's attainment. Episodes are drawn
    uniformly from the pool."""

    def __init__(self, pool: list, seed: int):
        self.pool = pool
        self.observation_space = gym.spaces.Box(-OBSERVATION_BOUND, OBSERVATION_BOUND, (OBSERVATION_SIZE,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_SIZE,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.episode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = episode_from(self.pool[int(self.rng.integers(len(self.pool)))])
        return self.episode.observation(), {}

    def step(self, action):
        attainment = self.episode.attainment(self.episode.params_for(action))
        reward = float(np.clip(attainment, *REWARD_CLIP))
        return self.episode.observation(), reward, True, False, {"attainment": attainment, "kind": self.episode.kind}


class Progress(BaseCallback):
    """Checkpoints at fixed step counts and a log line with the recent median attainment."""

    def __init__(self, out: str, checkpoints, every: int = 2000):
        super().__init__()
        self.out, self.checkpoints, self.every = out, set(checkpoints), every
        self.recent, self.started = [], time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "attainment" in info:
                self.recent.append(info["attainment"])
        if self.num_timesteps in self.checkpoints:
            self.model.save(os.path.join(self.out, f"steps_{self.num_timesteps}.zip"))
        if self.num_timesteps % self.every == 0:
            window = self.recent[-self.every:]
            rate = self.num_timesteps / (time.time() - self.started)
            print(f"[sac] {self.num_timesteps} steps, train median attainment {np.median(window):.3f} "
                  f"(last {len(window)}), {rate:.1f} steps/s", flush=True)
        return True


def train(args):
    os.makedirs(args.out, exist_ok=False)
    torch.set_num_threads(args.threads)
    with open(args.pool, "rb") as stream:
        pool = pickle.load(stream)
    env = OneShotV3(pool, args.seed)
    # gamma is irrelevant (every episode is one step); the rest are SB3 defaults
    # except a longer random warm-up, as in the task v1 runs.
    model = SAC("MlpPolicy", env, seed=args.seed, learning_starts=1000, buffer_size=args.steps,
                batch_size=256, gamma=0.0, verbose=0, device="cpu")
    checkpoints = sorted({c for c in {EQUAL_BUDGET, args.steps} | set(range(50_000, args.steps + 1, 50_000)) if c <= args.steps})
    with open(os.path.join(args.out, "config.json"), "w") as stream:
        json.dump({"steps": args.steps, "seed": args.seed, "equal_budget": EQUAL_BUDGET,
                   "reward_clip": REWARD_CLIP, "shell": transfer.UNLABELLED_PEAKS,
                   "train_splits": TRAIN_SPLITS, "pool": args.pool, "pool_episodes": len(pool),
                   "checkpoints": checkpoints}, stream, indent=1)
    model.learn(args.steps, callback=Progress(args.out, checkpoints))
    print(f"[sac] done -> {args.out}", flush=True)


class SacPolicy:
    """`rl.evaluate.evaluate_episode`'s policy interface for an SB3 SAC model (no dynamics: no planning)."""
    dynamics = None

    def __init__(self, path):
        self.model = SAC.load(path, device="cpu")

    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action, None


def evaluate(args):
    torch.set_num_threads(1)
    policy = SacPolicy(args.checkpoint)
    lo, hi = args.range
    rows = []
    for index, episode in enumerate(fixed_episodes(args.split, hi, start=lo), start=lo):
        rows.append(evaluate_episode(policy, episode, index, plan=False, budgets=(30, 200)))
        with open(args.out, "w") as stream:
            json.dump(rows, stream)
    print(f"[sac-eval] {len(rows)} episodes -> {args.out}; median policy "
          f"{np.median([r['policy'] for r in rows]):.3f}, warm30 {np.median([r['warm30'] for r in rows]):.3f}, "
          f"warm200 {np.median([r['warm200'] for r in rows]):.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--pool", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=2 * EQUAL_BUDGET)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("checkpoint")
    e.add_argument("--split", default="test")
    e.add_argument("--range", nargs=2, type=int, default=(0, 300))
    e.add_argument("--out", required=True)
    args = parser.parse_args()
    {"pool": build_pool, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
