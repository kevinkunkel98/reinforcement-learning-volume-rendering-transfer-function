"""Small reward model learned from human better/worse judgments of rendered
before/after transfer-function pairs -- used in place of the automatic
mass_fraction metric as the RL reward. See rl/reward_model_env.py for where
this gets used, and rl/collect_preferences.py for where the training data
comes from."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from rl.env import TISSUES

FEATURE_KEYS = ("mean", "std", "coverage", "entropy")
INPUT_DIM = len(FEATURE_KEYS) + len(TISSUES) + 1  # 10


def featurize(before_features: dict, after_features: dict, target_tissue: str, direction: str) -> np.ndarray:
    delta = [after_features[k] - before_features[k] for k in FEATURE_KEYS]
    onehot = [1.0 if t == target_tissue else 0.0 for t in TISSUES]
    direction_flag = 1.0 if direction == "increase" else -1.0
    return np.array(delta + onehot + [direction_flag], dtype=np.float32)


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reward_model(preferences_path: str, model_path: str, epochs: int = 200,
                        lr: float = 1e-2, seed: int = 0) -> RewardModel:
    with open(preferences_path) as f:
        records = [json.loads(line) for line in f]

    X = np.stack([
        featurize(r["before_features"], r["after_features"], r["target_tissue"], r["direction"])
        for r in records
    ])
    y = np.array([1.0 if r["label"] == 1 else 0.0 for r in records], dtype=np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    model = RewardModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = ((model(X_train) > 0).float() == y_train).float().mean().item()
        val_acc = ((model(X_val) > 0).float() == y_val).float().mean().item()
    print(f"[reward_model] train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
          f"(n_train={len(train_idx)}, n_val={len(val_idx)})")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model


def load_reward_model(model_path: str) -> RewardModel:
    model = RewardModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_reward(model: RewardModel, before_features: dict, after_features: dict,
                    target_tissue: str, direction: str) -> float:
    x = torch.from_numpy(featurize(before_features, after_features, target_tissue, direction)).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).item()
    return float(2.0 / (1.0 + np.exp(-logit)) - 1.0)


def main():
    parser = argparse.ArgumentParser(description="Train the RLHF reward model from logged preferences.")
    parser.add_argument("--preferences", type=str, default="out/rlhf_preferences.jsonl")
    parser.add_argument("--out", type=str, default="out/rl_models/reward_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_reward_model(args.preferences, args.out, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
