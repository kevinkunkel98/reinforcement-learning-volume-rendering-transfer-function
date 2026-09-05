"""Train a SAC agent on TFEnv against the opacity_mass-derived reward."""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rl.env import TFEnv

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
LOG_DIR = "out/rl_logs"


def _make_env():
    return Monitor(TFEnv())


def train(total_timesteps: int, n_envs: int = 4, seed: int = 0) -> SAC:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    vec_env = make_vec_env(_make_env, n_envs=n_envs, seed=seed)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_PATH)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the TF SAC agent.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.timesteps, n_envs=args.n_envs, seed=args.seed)


if __name__ == "__main__":
    main()
