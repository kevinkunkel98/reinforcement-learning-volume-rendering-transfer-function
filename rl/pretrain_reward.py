"""Synthetic objective pretraining for goal-conditioned reward models."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

import render
from datasets import load_dataset
from rl.env import DIRECTIONS, TISSUES
from rl.reward_model import RewardModel, _pair_arrays, bradley_terry_loss, save_reward_model
from transfer import TOTAL_PARAMS, default_params, mass_fraction


DEFAULT_PAIR_COUNT = 20_000
DEFAULT_MEMBERS = 5
FEATURE_KEYS = ("mean", "std", "coverage", "entropy")


def sample_start_params(rng: np.random.Generator) -> np.ndarray:
    """Sample a valid normalized transfer-function vector near its default."""
    return np.clip(default_params() + rng.normal(0.0, 0.2, TOTAL_PARAMS), -1.0, 1.0)


def sample_delta(rng: np.random.Generator) -> np.ndarray:
    """Sample bounded normalized changes, preserving valid TF coordinates."""
    return rng.uniform(-0.2, 0.2, TOTAL_PARAMS)


def sample_goal(rng: np.random.Generator) -> tuple[str, str]:
    return TISSUES[int(rng.integers(len(TISSUES)))], DIRECTIONS[int(rng.integers(2))]


def _feature_dict(features: dict) -> dict:
    return {key: float(features[key]) for key in FEATURE_KEYS}


def render_features(renderer, params: np.ndarray) -> dict:
    """Render params through seam, then extract scalar image features."""
    return _feature_dict(render.features(renderer(params)))


def _default_renderer():
    volume, spacing = load_dataset("synthetic")

    def renderer(params):
        window = render.render(volume, params, spacing)
        return render.grab(window)

    return renderer


def _objective(before: np.ndarray, after: np.ndarray, target: str, direction: str) -> float:
    sign = 1.0 if direction == "increase" else -1.0
    return sign * (mass_fraction(after, target) - mass_fraction(before, target))


def generate_synthetic_pairs(pair_count: int = DEFAULT_PAIR_COUNT, *, seed: int = 0,
                             renderer=None, render_features_fn=None) -> list[dict]:
    """Generate weighted canonical pair records from two random deltas."""
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    rng = np.random.default_rng(seed)
    renderer = renderer or _default_renderer()
    feature_extractor = render_features_fn or render_features
    rows = []
    for _ in range(pair_count):
        start = sample_start_params(rng)
        target, direction = sample_goal(rng)
        after_a = np.clip(start + sample_delta(rng), -1.0, 1.0)
        after_b = np.clip(start + sample_delta(rng), -1.0, 1.0)
        before_features = feature_extractor(renderer, start)
        after_a_features = feature_extractor(renderer, after_a)
        after_b_features = feature_extractor(renderer, after_b)
        objective_a = _objective(start, after_a, target, direction)
        objective_b = _objective(start, after_b, target, direction)
        # TODO: replace mass_fraction objective with rendered visibility metric when available.
        label = 1 if objective_a >= objective_b else -1
        rows.append({
            "source": "synthetic",
            "command": {"attribute": "opacity", "target": target, "direction": direction},
            "target_tissue": target,
            "direction": direction,
            "weight": 1.0,
            "label": label,
            "objective_a": float(objective_a),
            "objective_b": float(objective_b),
            "observation_a": {
                "before_features": before_features,
                "after_features": after_a_features,
            },
            "observation_b": {
                "before_features": before_features,
                "after_features": after_b_features,
            },
        })
    return rows


def _train_member(records: list[dict], *, seed: int, epochs: int, learning_rate: float):
    torch.manual_seed(seed)
    preferred, other, weights = _pair_arrays(records)
    model = RewardModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = bradley_terry_loss(model(preferred), model(other), weights)
        loss.backward()
        optimizer.step()
    return model


def pretrain(pair_count: int = DEFAULT_PAIR_COUNT, *, epochs: int = 200, seed: int = 0,
             members: int = DEFAULT_MEMBERS, learning_rate: float = 1e-2,
             output_path="out/rl_models/reward_pretrained.pt", renderer=None,
             render_features_fn=None) -> list[Path]:
    """Train seeded ensemble and save member plus aggregate artifacts."""
    if members < 1:
        raise ValueError("members must be positive")
    records = generate_synthetic_pairs(
        pair_count, seed=seed, renderer=renderer,
        render_features_fn=render_features_fn,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    member_paths = []
    for index in range(members):
        member_path = output.with_name(f"{output.stem}_member{index}{output.suffix}")
        model = _train_member(records, seed=seed + index, epochs=epochs,
                              learning_rate=learning_rate)
        save_reward_model(model, member_path)
        member_paths.append(member_path)
    torch.save({
        "member_paths": [str(path) for path in member_paths],
        "seeds": [seed + index for index in range(members)],
        "pair_count": pair_count,
    }, output)
    return member_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--members", type=int, default=DEFAULT_MEMBERS)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--out", default="out/rl_models/reward_pretrained.pt")
    args = parser.parse_args()
    pretrain(args.pairs, epochs=args.epochs, seed=args.seed, members=args.members,
             learning_rate=args.learning_rate, output_path=args.out)


if __name__ == "__main__":
    main()
