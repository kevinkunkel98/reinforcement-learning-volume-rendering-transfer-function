"""Evaluate learned reward models without mixing evaluation with training data.

The default workflow scores one persisted held-out split with both checkpoint
families, writes objective/human disagreements for audit, and optionally runs
an independent blind comparison. Blind verdicts are deliberately written to a
different file and are never passed to pair extraction or fine-tuning.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from rl.reward_model import ensemble_reward, load_reward_model, predict_reward


def _label_from_scores(first: float, second: float) -> int:
    return 1 if first > second else -1


def _model_predictor(models) -> Callable[[dict], int]:
    if not isinstance(models, (list, tuple)):
        models = [models]

    def predict(row: dict) -> int:
        target = row.get("target_tissue", row.get("target"))
        direction = row["direction"]
        values = []
        for model in models:
            values.append(predict_reward(
                model, row["observation_a"]["before_features"],
                row["observation_a"]["after_features"], target, direction,
            ))
        first = ensemble_reward(values)
        values = [predict_reward(
            model, row["observation_b"]["before_features"],
            row["observation_b"]["after_features"], target, direction,
        ) for model in models]
        return _label_from_scores(first, ensemble_reward(values))

    return predict


def _accuracy(rows: list[dict], predictor: Callable[[dict], int]) -> dict:
    predictions = [int(predictor(row)) for row in rows]
    correct = [prediction == int(row["label"]) for prediction, row in zip(predictions, rows)]
    by_source = {}
    for source in sorted({row.get("source", "unknown") for row in rows}):
        source_correct = [ok for ok, row in zip(correct, rows)
                          if row.get("source", "unknown") == source]
        by_source[source] = float(np.mean(source_correct)) if source_correct else None
    return {
        "count": len(rows),
        "overall_accuracy": float(np.mean(correct)) if correct else None,
        "by_source": by_source,
    }


def evaluate_reward_only(
    rows: Iterable[dict], *, pretrained_predictor: Callable[[dict], int],
    finetuned_predictor: Callable[[dict], int],
) -> dict:
    """Compare two predictors on exactly the same held-out row objects."""
    rows = list(rows)
    result = {
        "test_row_ids": [row.get("id", index) for index, row in enumerate(rows)],
        "pretrained": _accuracy(rows, pretrained_predictor),
        "finetuned": _accuracy(rows, finetuned_predictor),
    }
    return result


def find_disagreements(
    rows: Iterable[dict], *, human_predictor: Callable[[dict], int],
    objective_predictor: Callable[[dict], int],
) -> list[dict]:
    """Return auditable objective/human disagreements, including image paths."""
    disagreements = []
    for index, row in enumerate(rows):
        human = int(human_predictor(row))
        objective_value = objective_predictor(row)
        if objective_value is None:
            continue
        objective = int(objective_value)
        if human == objective:
            continue
        def images(observation):
            return {
                "before": observation.get("before_image"),
                "after": observation.get("after_image"),
            }
        disagreements.append({
            "id": row.get("id", index),
            "source": row.get("source", "unknown"),
            "target_tissue": row.get("target_tissue", row.get("target")),
            "direction": row["direction"],
            "human_label": human,
            "objective_label": objective,
            "observation_a_images": images(row["observation_a"]),
            "observation_b_images": images(row["observation_b"]),
        })
    return disagreements


def policy_objective_sanity_check(
    episodes: Iterable[dict], *, policy_runner: Callable[[dict], list[float]],
    baseline_runner: Callable[[dict], list[float]],
) -> dict:
    """Compare final objective values without rendering or changing training."""
    episodes = list(episodes)
    policy = [policy_runner(episode)[-1] for episode in episodes]
    baseline = [baseline_runner(episode)[-1] for episode in episodes]
    return {
        "n_episodes": len(episodes),
        "policy_mean_final_objective": float(np.mean(policy)) if policy else None,
        "baseline_mean_final_objective": float(np.mean(baseline)) if baseline else None,
    }


def evaluate_rl_policy_against_hill_climb(model_path: str, *, n_episodes: int = 20,
                                          seed: int = 12345) -> dict:
    """Run the existing ``rl.eval`` policy/baseline runners as an objective check."""
    from stable_baselines3 import SAC
    from rl.eval import _build_episodes, _run_hill_climb, _run_policy

    model = SAC.load(model_path)
    episodes = _build_episodes(n_episodes, seed)
    return policy_objective_sanity_check(
        episodes,
        policy_runner=lambda episode: _run_policy(model, episode),
        baseline_runner=_run_hill_climb,
    )


def collect_blind_head_to_head(
    episodes: Iterable[dict], *, render_policy: Callable[[dict], str],
    render_baseline: Callable[[dict], str], seed: int = 0,
    verdicts: Iterable[str] | None = None, verdict_path=None,
) -> list[dict]:
    """Create randomized A/B records and append supplied human verdicts.

    ``verdicts`` is a seam for a UI or manual collector. This function does
    not call extraction, training, or any code that consumes verdicts.
    """
    rng = random.Random(seed)
    verdict_iter = iter(verdicts) if verdicts is not None else iter(())
    results = []
    for index, episode in enumerate(episodes):
        policy_path = render_policy(episode)
        baseline_path = render_baseline(episode)
        policy_first = bool(rng.randrange(2))
        if policy_first:
            order, a_path, b_path = "policy_baseline", policy_path, baseline_path
        else:
            order, a_path, b_path = "baseline_policy", baseline_path, policy_path
        raw_verdict = next(verdict_iter, None)
        verdict = None if raw_verdict is None else str(raw_verdict).lower()
        if verdict in ("better", "a"):
            verdict = "A"
        elif verdict in ("worse", "b"):
            verdict = "B"
        elif verdict in ("policy", "baseline"):
            verdict = "A" if (verdict == "policy") == policy_first else "B"
        if verdict not in (None, "A", "B", "tie"):
            raise ValueError("verdict must be better, worse, tie, policy, or baseline")
        results.append({
            "id": episode.get("id", index),
            "display_order": order,
            "a_image": a_path,
            "b_image": b_path,
            "verdict": verdict,
        })
    if verdict_path is not None:
        path = Path(verdict_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
    return results


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_members(aggregate_path) -> list:
    checkpoint = torch_load(aggregate_path)
    names = checkpoint.get("member_paths") if isinstance(checkpoint, dict) else None
    if not names:
        raise ValueError(f"checkpoint missing member_paths: {aggregate_path}")
    return [load_reward_model(_resolve_member_path(path, aggregate_path))
            for path in names]


def _resolve_member_path(member_name, aggregate_path) -> Path:
    """Resolve stored member paths from absolute, cwd, or artifact layouts."""
    member = Path(member_name)
    if member.is_absolute() and member.exists():
        return member
    aggregate = Path(aggregate_path)
    candidates = []
    if member.is_absolute():
        candidates.append(member)
    else:
        candidates.extend((member, Path.cwd() / member,
                           aggregate.parent / member,
                           aggregate.parent / member.name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"reward member not found: {member_name!r}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def torch_load(path):
    # Local wrapper keeps command-line loading easy to replace in tests.
    import torch
    return torch.load(path, map_location="cpu")


def _row_hash(record: dict, ordinal: int) -> str:
    payload = json.dumps(
        {"ordinal": ordinal, "record": record}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_fixed_test_rows(rows: Iterable[dict], metadata: dict) -> list[dict]:
    """Validate persisted split identity and return its test rows.

    Missing or inconsistent metadata is fatal. Evaluation never regenerates a
    different split when the persisted split cannot be proven intact.
    """
    rows = list(rows)
    required = ("train_count", "test_count", "train_row_hashes", "test_row_hashes")
    if not isinstance(metadata, dict) or any(key not in metadata for key in required):
        raise ValueError("fixed split metadata is missing required metadata")
    train_hashes = list(metadata["train_row_hashes"])
    test_hashes = list(metadata["test_row_hashes"])
    if (metadata["train_count"] != len(train_hashes) or
            metadata["test_count"] != len(test_hashes)):
        raise ValueError("fixed split metadata counts do not match metadata hashes")
    if set(train_hashes) & set(test_hashes):
        raise ValueError("fixed split metadata has overlapping train/test hashes")
    known = {_row_hash(row, ordinal): row for ordinal, row in enumerate(rows)}
    expected = set(train_hashes) | set(test_hashes)
    if len(expected) != len(rows) or set(known) != expected:
        raise ValueError("fixed split metadata hashes do not match preference rows")
    return [known[row_hash] for row_hash in test_hashes]


def evaluate_files(preferences_path, pretrained_path, finetuned_path, *, test_rows=None,
                   disagreements_path=None, split_seed=0, test_fraction=0.2) -> dict:
    """Evaluate persisted artifacts; callers may pass persisted split rows."""
    rows = _read_jsonl(preferences_path)
    if test_rows is None:
        metadata = torch_load(finetuned_path).get("metadata")
        test_rows = select_fixed_test_rows(rows, metadata)
    pretrained = _model_predictor(_load_members(pretrained_path))
    finetuned = _model_predictor(_load_members(finetuned_path))
    result = evaluate_reward_only(rows=test_rows, pretrained_predictor=pretrained,
                                  finetuned_predictor=finetuned)
    def objective(row):
        if "objective_a" not in row or "objective_b" not in row:
            return None
        return _label_from_scores(row["objective_a"], row["objective_b"])
    disagreements = find_disagreements(test_rows, human_predictor=lambda row: row["label"],
                                       objective_predictor=objective)
    result["disagreement_count"] = len(disagreements)
    if disagreements_path is not None:
        path = Path(disagreements_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                              for row in disagreements), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--pretrained", default="out/rl_models/reward_pretrained.pt")
    parser.add_argument("--finetuned", default="out/rl_models/reward_finetuned.pt")
    parser.add_argument("--disagreements", default="out/reward_disagreements.jsonl")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = evaluate_files(args.preferences, args.pretrained, args.finetuned,
                            disagreements_path=args.disagreements,
                            split_seed=args.split_seed, test_fraction=args.test_fraction)
    rendered = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
