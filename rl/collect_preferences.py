"""Collects human better/worse preference labels over rendered transfer-
function before/after pairs, for training the reward model in
rl/reward_model.py. Reuses the same base-case sampling rl/eval.py's offline
comparison uses, and the same console judging prompt the hill-climb search
flow already uses (evaluate.human) -- run this yourself, interactively, it
cannot be delegated to a background agent."""
import argparse
import datetime
import os

import numpy as np
from PIL import Image

from camera import DEFAULT_CAMERA
from datasets import load_dataset
from evaluate import human, jsonl_append
from rl.env import MAX_DELTA
from rl.eval import _build_episodes
from search import propose_step

import render

PREFERENCES_PATH = "out/rlhf_preferences.jsonl"
LABELING_DIR = "out/rlhf_labeling"


def collect_preferences(n_pairs: int = 50, rater_id: str = "default", seed: int = 0,
                         out_path: str = PREFERENCES_PATH) -> None:
    volume, spacing = load_dataset("synthetic")
    camera = dict(DEFAULT_CAMERA)
    cases = _build_episodes(n_pairs, seed)
    rng = np.random.default_rng(seed)

    os.makedirs(LABELING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for i, case in enumerate(cases):
        before_params = case["params"]
        action = float(rng.uniform(-1.0, 1.0))
        delta = action * MAX_DELTA
        after_params = propose_step(before_params, case["peak_idx"], sign=1.0, step=delta)

        before_win = render.render(volume, before_params, spacing, camera)
        before_rgb = render.grab(before_win)
        before_features = render.features(before_rgb)
        after_win = render.render(volume, after_params, spacing, camera)
        after_rgb = render.grab(after_win)
        after_features = render.features(after_rgb)

        before_path = os.path.join(LABELING_DIR, f"{i}_before.png")
        after_path = os.path.join(LABELING_DIR, f"{i}_after.png")
        Image.fromarray(before_rgb).save(before_path)
        Image.fromarray(after_rgb).save(after_path)

        print(f"[{i + 1}/{n_pairs}] target={case['target_tissue']} direction={case['direction']}")
        label = human(before_path, after_path)

        jsonl_append(out_path, {
            "timestamp": datetime.datetime.now().isoformat(),
            "rater_id": rater_id,
            "target_tissue": case["target_tissue"],
            "direction": case["direction"],
            "delta": delta,
            "before_features": before_features,
            "after_features": after_features,
            "label": label,
        })


def main():
    parser = argparse.ArgumentParser(description="Collect human preference labels for the reward model.")
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--rater-id", type=str, default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=PREFERENCES_PATH)
    args = parser.parse_args()
    collect_preferences(args.n_pairs, rater_id=args.rater_id, seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
