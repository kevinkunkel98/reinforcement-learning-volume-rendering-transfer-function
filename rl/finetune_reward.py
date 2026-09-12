"""Fine-tune pretrained reward-model members on canonical human preference pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl.reward_model import (
    _pair_arrays,
    bradley_terry_loss,
    load_reward_model,
    save_reward_model,
)


DEFAULT_OUTPUT = "out/rl_models/reward_finetuned.pt"


def split_pairs(records: list[dict], *, seed: int = 0,
                test_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Return a deterministic, disjoint train/test split without mutating records."""
    if not records:
        raise ValueError("preference records cannot be empty")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    test_count = max(1, int(round(len(records) * test_fraction)))
    indices = np.random.default_rng(seed).permutation(len(records))
    test_indices = set(int(index) for index in indices[:test_count])
    train = [record for index, record in enumerate(records) if index not in test_indices]
    test = [record for index, record in enumerate(records) if index in test_indices]
    return train, test


def _load_records(preferences_path) -> list[dict]:
    with open(preferences_path) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError("preferences file contains no records")
    return records


def _train_member(model, records: list[dict], *, seed: int, epochs: int,
                  learning_rate: float):
    torch.manual_seed(seed)
    preferred, other, weights = _pair_arrays(records)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = bradley_terry_loss(model(preferred), model(other), weights)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def finetune(preferences_path, pretrained_path, *, output_path=DEFAULT_OUTPUT,
             epochs: int = 200, split_seed: int = 0, test_fraction: float = 0.2,
             pretrain_lr: float = 1e-2) -> list[Path]:
    """Fine-tune every pretrained member using only the fixed train split."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if pretrain_lr <= 0:
        raise ValueError("pretrain_lr must be positive")
    preferences = _load_records(preferences_path)
    train_records, test_records = split_pairs(
        preferences, seed=split_seed, test_fraction=test_fraction
    )
    pretrained = Path(pretrained_path)
    output = Path(output_path)
    aggregate = torch.load(pretrained, map_location="cpu")
    member_names = aggregate.get("member_paths") if isinstance(aggregate, dict) else None
    seeds = aggregate.get("seeds") if isinstance(aggregate, dict) else None
    if not member_names or not seeds or len(member_names) != len(seeds):
        raise ValueError("pretrained checkpoint missing member_paths or seeds")
    member_paths = [output.with_name(f"{output.stem}_member{index}{output.suffix}")
                    for index in range(len(member_names))]
    if output.resolve() == pretrained.resolve() or any(
            path.resolve() == pretrained.resolve() for path in member_paths):
        raise ValueError("fine-tuned artifacts must not overwrite pretrained artifacts")
    if output.exists() or any(path.exists() for path in member_paths):
        raise FileExistsError(f"fine-tuning artifact already exists: {output}")

    learning_rate = pretrain_lr / 10.0
    for index, member_name in enumerate(member_names):
        member_path = Path(member_name)
        if not member_path.is_absolute():
            member_path = pretrained.parent / member_path
        model = load_reward_model(member_path)
        model = _train_member(model, train_records, seed=int(seeds[index]),
                              epochs=epochs, learning_rate=learning_rate)
        save_reward_model(model, member_paths[index])

    train_ids = [record["id"] for record in train_records if "id" in record]
    test_ids = [record["id"] for record in test_records if "id" in record]
    metadata = {
        "seed": int(seeds[0]),
        "split_seed": split_seed,
        "train_count": len(train_records),
        "test_count": len(test_records),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "learning_rate": learning_rate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "member_paths": [str(path) for path in member_paths],
        "seeds": [int(seed) for seed in seeds],
        "pretrained_path": str(pretrained),
        "metadata": metadata,
    }, output)
    return member_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--pretrained", default="out/rl_models/reward_pretrained.pt")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    args = parser.parse_args()
    finetune(args.preferences, args.pretrained, output_path=args.out,
             epochs=args.epochs, split_seed=args.split_seed,
             test_fraction=args.test_fraction, pretrain_lr=args.pretrain_lr)


if __name__ == "__main__":
    main()
