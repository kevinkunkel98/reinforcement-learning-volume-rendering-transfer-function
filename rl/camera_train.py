"""Train a SAC agent on CameraViewpointEnv against the centroid-alignment reward."""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as configure_logger

from datasets import load_dataset
from rl.camera_env import CameraViewpointEnv

MODEL_PATH = "out/rl_models/sac_camera_agent.zip"
LOG_DIR = "out/rl_camera_logs"


def train(total_timesteps: int, dataset: str = "ct_skull", n_envs: int = 4,
          seed: int = 0, model_path: str = MODEL_PATH) -> SAC:
    volume, spacing = load_dataset(dataset)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    run_dir = os.path.join(LOG_DIR, f"run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    vec_env = make_vec_env(lambda: CameraViewpointEnv(volume, spacing), n_envs=n_envs, seed=seed)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, seed=seed)
    model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the camera-viewpoint SAC agent.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--dataset", type=str, default="ct_skull")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=MODEL_PATH,
                         help="Path to save the trained model (default: %(default)s)")
    args = parser.parse_args()
    train(args.timesteps, dataset=args.dataset, n_envs=args.n_envs, seed=args.seed, model_path=args.out)


if __name__ == "__main__":
    main()
