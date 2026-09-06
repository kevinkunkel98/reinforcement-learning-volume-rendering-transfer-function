"""Compare the trained camera-viewpoint SAC policy against a hill-climbing
baseline over the same held-out (tissue, start-angle) episodes."""
import json
import os

import numpy as np
from stable_baselines3 import SAC

from datasets import load_dataset
from rl.camera_env import ELEVATION_RANGE, MAX_DELTA_DEGREES, MAX_STEPS, CameraViewpointEnv, _view_direction
from rl.eval import _steps_to_90pct
from search import resize_step

MODEL_PATH = "out/rl_models/sac_camera_agent.zip"
RESULTS_PATH = "out/rl_camera_eval_results.json"
N_EPISODES = 20
EVAL_SEED = 12345


def _build_episodes(volume, spacing, n_episodes, seed):
    env = CameraViewpointEnv(volume, spacing, seed=seed)
    episodes = []
    for _ in range(n_episodes):
        env.reset()
        episodes.append({
            "azimuth": env.azimuth, "elevation": env.elevation,
            "target_tissue": env.target_tissue, "target_direction": env.target_direction,
        })
    return episodes


def _run_policy(model, volume, spacing, episode):
    env = CameraViewpointEnv(volume, spacing)
    env.azimuth = episode["azimuth"]
    env.elevation = episode["elevation"]
    env.target_tissue = episode["target_tissue"]
    env.target_direction = episode["target_direction"]
    env.step_count = 0
    alignments = [env._alignment()]
    obs = env._obs()
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        alignments.append(info["alignment_after"])
    return alignments


def _run_hill_climb(episode):
    azimuth, elevation = episode["azimuth"], episode["elevation"]
    target_direction = episode["target_direction"]

    def alignment(az, el):
        return float(np.dot(_view_direction(az, el), target_direction))

    step = MAX_DELTA_DEGREES
    alignments = [alignment(azimuth, elevation)]
    for _ in range(MAX_STEPS):
        current = alignments[-1]
        best, best_az, best_el = current, azimuth, elevation
        for d_az, d_el in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            cand_az = (azimuth + d_az) % 360.0
            cand_el = float(np.clip(elevation + d_el, *ELEVATION_RANGE))
            score = alignment(cand_az, cand_el)
            if score > best:
                best, best_az, best_el = score, cand_az, cand_el
        accepted = best > current
        if accepted:
            azimuth, elevation = best_az, best_el
        step = resize_step(step, accepted, max_step=MAX_DELTA_DEGREES)
        alignments.append(alignment(azimuth, elevation))
    return alignments


def evaluate(model_path: str = MODEL_PATH, dataset: str = "ct_skull",
             n_episodes: int = N_EPISODES, seed: int = EVAL_SEED,
             results_path: str = RESULTS_PATH) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path!r} -- run `python -m rl.camera_train` first."
        )
    volume, spacing = load_dataset(dataset)
    model = SAC.load(model_path)
    episodes = _build_episodes(volume, spacing, n_episodes, seed)

    policy_finals, policy_steps = [], []
    hillclimb_finals, hillclimb_steps = [], []
    for ep in episodes:
        pf = _run_policy(model, volume, spacing, ep)
        hf = _run_hill_climb(ep)
        best = max(pf + hf)
        policy_finals.append(pf[-1])
        hillclimb_finals.append(hf[-1])
        policy_steps.append(_steps_to_90pct(pf, pf[0], best))
        hillclimb_steps.append(_steps_to_90pct(hf, hf[0], best))

    results = {
        "n_episodes": n_episodes,
        "dataset": dataset,
        "policy": {
            "mean_final_alignment": float(np.mean(policy_finals)),
            "mean_steps_to_90pct": float(np.mean(policy_steps)),
        },
        "hill_climb": {
            "mean_final_alignment": float(np.mean(hillclimb_finals)),
            "mean_steps_to_90pct": float(np.mean(hillclimb_steps)),
        },
    }
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
