"""Compare the trained SAC policy against the hill-climbing baseline over
the same held-out (tissue, direction, start-params) episodes."""
import json

import numpy as np
from stable_baselines3 import SAC

from rl.env import MAX_DELTA, MAX_STEPS, TFEnv
from search import propose_step, resize_step
from transfer import mass_fraction

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
RESULTS_PATH = "out/rl_eval_results.json"
N_EPISODES = 20
EVAL_SEED = 12345


def _build_episodes(n_episodes: int, seed: int) -> list:
    env = TFEnv(seed=seed)
    episodes = []
    for _ in range(n_episodes):
        env.reset()
        episodes.append({
            "params": env.params.copy(),
            "target_tissue": env.target_tissue,
            "direction": env.direction,
            "peak_idx": env.peak_idx,
        })
    return episodes


def _run_policy(model, episode: dict) -> list:
    env = TFEnv()
    env.params = episode["params"].copy()
    env.target_tissue = episode["target_tissue"]
    env.direction = episode["direction"]
    env.peak_idx = episode["peak_idx"]
    env.step_count = 0
    fractions = [mass_fraction(env.params, env.target_tissue)]
    obs = env._obs()
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        fractions.append(info["mass_fraction_after"])
    return fractions


def _run_hill_climb(episode: dict) -> list:
    params = episode["params"].copy()
    peak_idx = episode["peak_idx"]
    tissue = episode["target_tissue"]
    sign = 1.0 if episode["direction"] == "increase" else -1.0
    step = MAX_DELTA
    fractions = [mass_fraction(params, tissue)]
    for _ in range(MAX_STEPS):
        proposed = propose_step(params, peak_idx, sign=sign, step=step)
        before = fractions[-1]
        after = mass_fraction(proposed, tissue)
        accepted = sign * (after - before) > 0.0
        if accepted:
            params = proposed
        step = resize_step(step, accepted)
        fractions.append(mass_fraction(params, tissue))
    return fractions


def _steps_to_90pct(fractions: list, start: float, best: float) -> int:
    target = start + 0.9 * (best - start)
    for i, f in enumerate(fractions):
        if (best >= start and f >= target) or (best < start and f <= target):
            return i
    return len(fractions) - 1


def evaluate(model_path: str = MODEL_PATH, n_episodes: int = N_EPISODES, seed: int = EVAL_SEED) -> dict:
    model = SAC.load(model_path)
    episodes = _build_episodes(n_episodes, seed)

    policy_finals, policy_steps = [], []
    hillclimb_finals, hillclimb_steps = [], []
    for ep in episodes:
        pf = _run_policy(model, ep)
        hf = _run_hill_climb(ep)
        combined = pf + hf
        best = max(combined) if ep["direction"] == "increase" else min(combined)
        policy_finals.append(pf[-1])
        hillclimb_finals.append(hf[-1])
        policy_steps.append(_steps_to_90pct(pf, pf[0], best))
        hillclimb_steps.append(_steps_to_90pct(hf, hf[0], best))

    results = {
        "n_episodes": n_episodes,
        "policy": {
            "mean_final_mass_fraction": float(np.mean(policy_finals)),
            "mean_steps_to_90pct": float(np.mean(policy_steps)),
        },
        "hill_climb": {
            "mean_final_mass_fraction": float(np.mean(hillclimb_finals)),
            "mean_steps_to_90pct": float(np.mean(hillclimb_steps)),
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
