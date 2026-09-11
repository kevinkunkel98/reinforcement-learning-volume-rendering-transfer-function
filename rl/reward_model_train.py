"""Trains a fresh SAC agent against the learned reward model
(rl/reward_model.py) via RewardModelTFEnv, instead of the automatic
mass_fraction metric -- mirrors rl/online_train.py's chunked-learn()-plus-
checkpoint-eval shape, at a much smaller scale since every step now renders
(see docs/superpowers/specs/2026-09-09-rlhf-reward-model-design.md)."""
import argparse
import csv
import math
import os

import numpy as np
from PIL import Image
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_logger

from camera import DEFAULT_CAMERA
from datasets import load_dataset
from rl.eval import _build_episodes
from rl.reward_model import load_reward_model
from rl.reward_model_env import RewardModelTFEnv

import render

MODEL_PATH = "out/rl_models/sac_tf_reward_model_agent.zip"
LOG_DIR = "out/rl_logs"
EVAL_N_EPISODES = 5
EVAL_SEED = 12345
N_GALLERY_EPISODES = 3
MAX_STEPS = 20


def _seed_episode(env, episode):
    env.params = episode["params"].copy()
    env.target_tissue = episode["target_tissue"]
    env.direction = episode["direction"]
    env.peak_idx = episode["peak_idx"]
    env.step_count = 0
    env._prev_features = env._render_features(env.params)
    return env._obs()


def _run_policy_with_logging(model, env, episode):
    obs = _seed_episode(env, episode)
    reward_scores, automatic_rewards = [], []
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_scores.append(reward)
        automatic_rewards.append(info["automatic_mass_fraction_reward"])
    return reward_scores, automatic_rewards


def reward_model_train(total_timesteps: int = 2000, eval_interval: int = 500, seed: int = 0,
                        reward_model_path: str = "out/rl_models/reward_model.pt",
                        model_path: str = MODEL_PATH) -> SAC:
    run_dir = os.path.join(LOG_DIR, f"reward_model_run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    volume, spacing = load_dataset("synthetic")
    reward_model = load_reward_model(reward_model_path)
    env = RewardModelTFEnv(volume, spacing, reward_model, seed=seed)
    model = SAC("MlpPolicy", env, seed=seed)
    model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))

    eval_episodes = _build_episodes(EVAL_N_EPISODES, EVAL_SEED)
    eval_env = RewardModelTFEnv(volume, spacing, reward_model, seed=seed + 1)

    eval_log_path = os.path.join(run_dir, "eval_progress.csv")
    with open(eval_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["timesteps", "mean_reward_model_score", "mean_automatic_mass_fraction_reward"])

    n_chunks = math.ceil(total_timesteps / eval_interval)
    timesteps_done = 0
    for _ in range(n_chunks):
        chunk = min(eval_interval, total_timesteps - timesteps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        timesteps_done += chunk

        all_reward_scores, all_automatic_rewards = [], []
        for ep in eval_episodes:
            reward_scores, automatic_rewards = _run_policy_with_logging(model, eval_env, ep)
            all_reward_scores.extend(reward_scores)
            all_automatic_rewards.extend(automatic_rewards)

        row = [timesteps_done, float(np.mean(all_reward_scores)), float(np.mean(all_automatic_rewards))]
        with open(eval_log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[reward_model_train] timesteps={timesteps_done} "
              f"mean_reward_model_score={row[1]:.4f} mean_automatic_reward={row[2]:.4f}")

    gallery_dir = os.path.join(run_dir, "gallery")
    os.makedirs(gallery_dir, exist_ok=True)
    for i, ep in enumerate(eval_episodes[:N_GALLERY_EPISODES]):
        before_win = render.render(volume, ep["params"], spacing, dict(DEFAULT_CAMERA))
        Image.fromarray(render.grab(before_win)).save(os.path.join(gallery_dir, f"{i}_before.png"))

        obs = _seed_episode(eval_env, ep)
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = eval_env.step(action)
        after_win = render.render(volume, eval_env.params, spacing, dict(DEFAULT_CAMERA))
        Image.fromarray(render.grab(after_win)).save(os.path.join(gallery_dir, f"{i}_after.png"))

    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SAC against the learned RLHF reward model.")
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-model", type=str, default="out/rl_models/reward_model.pt")
    parser.add_argument("--out", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    reward_model_train(args.timesteps, eval_interval=args.eval_interval, seed=args.seed,
                        reward_model_path=args.reward_model, model_path=args.out)


if __name__ == "__main__":
    main()
