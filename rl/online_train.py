"""Online-learning counterpart to rl/train.py -- trains a fresh SAC agent on
a single TFEnv, periodically re-evaluating against the same held-out
episodes rl/eval.py uses so the task metric's improvement over training is
directly observable, not just inferred from SB3's own reward curve.

Unlike rl/train.py (4 parallel envs, one 200k-step batch, evaluated once at
the end by rl/eval.py), this trains one env continuously and checkpoints
its held-out performance every `eval_interval` steps -- same underlying SAC
mechanics (SB3's own model.learn() drives the actual online interaction and
replay-buffer/gradient-update bookkeeping), just interleaved with
visibility. Writes to a separate model path and log directory from the
offline run, so neither one clobbers the other.
"""
import argparse
import csv
import math
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_logger

from rl.env import TFEnv
from rl.eval import EVAL_SEED, N_EPISODES, _build_episodes, _run_hill_climb, _run_policy, _steps_to_90pct

MODEL_PATH = "out/rl_models/sac_tf_agent_online.zip"
LOG_DIR = "out/rl_logs"


def online_train(total_timesteps: int = 50_000, eval_interval: int = 5_000,
                  seed: int = 0, model_path: str = MODEL_PATH) -> SAC:
    run_dir = os.path.join(LOG_DIR, f"online_run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = TFEnv(seed=seed)
    model = SAC("MlpPolicy", env, seed=seed)
    model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))

    # Same held-out set rl/eval.py checks the offline policy against, so
    # this run's numbers are directly comparable to the documented offline
    # table. Hill-climb trajectories are deterministic per episode, so
    # they're computed once here rather than once per checkpoint.
    episodes = _build_episodes(N_EPISODES, EVAL_SEED)
    hill_climb_fractions = [_run_hill_climb(ep) for ep in episodes]

    eval_log_path = os.path.join(run_dir, "eval_progress.csv")
    with open(eval_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["timesteps", "mean_final_mass_fraction", "mean_steps_to_90pct"])

    n_chunks = math.ceil(total_timesteps / eval_interval)
    timesteps_done = 0
    for _ in range(n_chunks):
        chunk = min(eval_interval, total_timesteps - timesteps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        timesteps_done += chunk

        finals, steps_to_90 = [], []
        for ep, hf in zip(episodes, hill_climb_fractions):
            pf = _run_policy(model, ep)
            combined = pf + hf
            best = max(combined) if ep["direction"] == "increase" else min(combined)
            finals.append(pf[-1])
            steps_to_90.append(_steps_to_90pct(pf, pf[0], best))

        row = [timesteps_done, float(np.mean(finals)), float(np.mean(steps_to_90))]
        with open(eval_log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[online_train] timesteps={timesteps_done} "
              f"mean_final_mass_fraction={row[1]:.4f} mean_steps_to_90pct={row[2]:.2f}")

    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train the TF SAC agent online, with periodic held-out evaluation.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    online_train(args.timesteps, eval_interval=args.eval_interval,
                 seed=args.seed, model_path=args.out)


if __name__ == "__main__":
    main()
