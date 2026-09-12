"""Goal-conditioned reward model learned from preference judgments."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from rl.env import TISSUES

FEATURE_KEYS = ("mean", "std", "coverage", "entropy")
INPUT_DIM = len(FEATURE_KEYS) * 3 + len(TISSUES) + 1


def encode_target(cmd: dict) -> np.ndarray:
    """Encode command goal; future versions can swap this for text embeddings."""
    target = cmd.get("target_tissue", cmd.get("target"))
    direction = cmd.get("direction")
    if target not in TISSUES:
        raise ValueError(f"invalid target tissue: {target!r}")
    if direction not in ("increase", "decrease"):
        raise ValueError(f"invalid direction: {direction!r}")
    onehot = [1.0 if tissue == target else 0.0 for tissue in TISSUES]
    direction_flag = 1.0 if direction == "increase" else -1.0
    return np.asarray(onehot + [direction_flag], dtype=np.float32)


def featurize(before_features: dict, after_features: dict, target_tissue: str,
              direction: str) -> np.ndarray:
    """Return after, before, delta, goal one-hot, and direction features."""
    after = [float(after_features[key]) for key in FEATURE_KEYS]
    before = [float(before_features[key]) for key in FEATURE_KEYS]
    if not np.isfinite(after + before).all():
        raise ValueError("features must be finite")
    delta = [a - b for a, b in zip(after, before)]
    goal = encode_target({"target_tissue": target_tissue, "direction": direction})
    return np.concatenate([
        np.asarray(after + before + delta, dtype=np.float32), goal
    ]).astype(np.float32)


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def bradley_terry_loss(preferred: torch.Tensor, other: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
    """Weighted negative log likelihood that preferred scores beat others."""
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("weights must be finite and positive")
    losses = -torch.nn.functional.logsigmoid(preferred - other)
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(losses.dtype).eps)


def _observation(record: dict, prefix: str = ""):
    if prefix:
        observation = record.get(prefix) or record.get(f"{prefix}_observation")
        if observation and "features" in observation:
            observation = observation["features"]
        if observation:
            return observation
        features = record.get(f"{prefix}_features")
    else:
        features = record.get("features")
    return features


def _record_features(record: dict, prefix: str):
    observation = _observation(record, prefix)
    if observation is None:
        raise KeyError(f"missing {prefix} features")
    return observation


def _pair_observation_features(observation: dict, target: str, direction: str):
    before = observation.get("before_features", observation.get("before"))
    after = observation.get("after_features", observation.get("after"))
    return featurize(before, after, target, direction)


def _pair_arrays(records):
    preferred, other, weights = [], [], []
    for record in records:
        target = record.get("target_tissue", record.get("target"))
        direction = record["direction"]
        weight = float(record.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weights must be finite and positive")
        if "label" in record and (isinstance(record["label"], bool) or
                                   record["label"] not in (-1, 1)):
            raise ValueError("label must be -1 or 1")
        if "observation_a" in record and "observation_b" in record:
            first = _pair_observation_features(record["observation_a"], target, direction)
            second = _pair_observation_features(record["observation_b"], target, direction)
            if record.get("label") == -1:
                first, second = second, first
            preferred.append(first)
            other.append(second)
        elif "preferred" in record or "chosen" in record:
            left = record.get("preferred", record.get("chosen"))
            right = record.get("rejected", record.get("other"))
            left_before = left.get("before_features", left.get("before"))
            left_after = left.get("after_features", left.get("after"))
            right_before = right.get("before_features", right.get("before"))
            right_after = right.get("after_features", right.get("after"))
            preferred.append(featurize(left_before, left_after, target, direction))
            other.append(featurize(right_before, right_after, target, direction))
        else:
            features = featurize(_record_features(record, "before"),
                                 _record_features(record, "after"), target, direction)
            zero = np.zeros(INPUT_DIM, dtype=np.float32)
            if record.get("label", 1) == 1:
                preferred.append(features)
                other.append(zero)
            else:
                preferred.append(zero)
                other.append(features)
        weights.append(weight)
    return (torch.from_numpy(np.stack(preferred)), torch.from_numpy(np.stack(other)),
            torch.tensor(weights, dtype=torch.float32))


def save_reward_model(model: RewardModel, model_path) -> None:
    directory = os.path.dirname(os.fspath(model_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def train_reward_model(preferences_path: str, model_path: str, epochs: int = 200,
                       lr: float = 1e-2, seed: int = 0) -> RewardModel:
    with open(preferences_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError("preferences file contains no records")

    torch.manual_seed(seed)
    preferred, other, weights = _pair_arrays(records)
    if len(preferred) < 2:
        raise ValueError("at least two preference records are required for training")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(preferred))
    n_val = max(1, len(idx) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    model = RewardModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = bradley_terry_loss(model(preferred[train_idx]), model(other[train_idx]), weights[train_idx])
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = (model(preferred[train_idx]) > model(other[train_idx])).float().mean().item()
        val_acc = (model(preferred[val_idx]) > model(other[val_idx])).float().mean().item()
    print(f"[reward_model] train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
          f"(n_train={len(train_idx)}, n_val={len(val_idx)})")
    save_reward_model(model, model_path)
    return model


def load_reward_model(model_path: str) -> RewardModel:
    model = RewardModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict_reward(model: RewardModel, before_features: dict, after_features: dict,
                   target_tissue: str, direction: str) -> float:
    x = torch.from_numpy(featurize(before_features, after_features, target_tissue, direction)).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).item()
    return float(2.0 * torch.sigmoid(torch.tensor(logit)).item() - 1.0)


def ensemble_reward(values) -> float:
    values = np.asarray(values, dtype=np.float32)
    return float(values.mean() - values.std())


def main():
    parser = argparse.ArgumentParser(description="Train the goal-conditioned reward model.")
    parser.add_argument("--preferences", type=str, default="out/rlhf_preferences.jsonl")
    parser.add_argument("--out", type=str, default="out/rl_models/reward_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_reward_model(args.preferences, args.out, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
