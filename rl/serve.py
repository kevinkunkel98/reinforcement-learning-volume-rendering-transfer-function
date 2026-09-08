"""Runs the trained opacity SAC policy against live params -- the serving
counterpart to rl/eval.py's offline policy-vs-hill-climb comparison."""
import os

import numpy as np
from stable_baselines3 import SAC

from rl.env import TFEnv

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
_model_cache = {}


def _load_model(model_path: str = MODEL_PATH):
    if model_path not in _model_cache:
        if not os.path.exists(model_path):
            raise ValueError(
                f"no trained model at {model_path!r} -- run `python -m rl.train` first"
            )
        _model_cache[model_path] = SAC.load(model_path)
    return _model_cache[model_path]


def run_policy(params: np.ndarray, target_tissue: str, direction: str,
               peak_idx: int, steps: int, model_path: str = MODEL_PATH) -> np.ndarray:
    """Rolls the trained policy forward `steps` times from `params`, seeded
    with the same (tissue, direction, peak_idx) shape rl/eval.py's
    _run_policy uses. Raises ValueError if no trained model exists at
    `model_path`. Does not mutate the input `params` array."""
    model = _load_model(model_path)
    env = TFEnv()
    env.params = params.copy()
    env.target_tissue = target_tissue
    env.direction = direction
    env.peak_idx = peak_idx
    env.step_count = 0
    obs = env._obs()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
    return env.params
