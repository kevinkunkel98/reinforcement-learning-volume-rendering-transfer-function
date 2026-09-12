import json
from pathlib import Path
import numpy as np
import pytest
import torch

from rl import pretrain_reward
from rl import finetune_reward
from rl import eval_reward
from rl.env import TISSUES


FEATURES = {"mean": 1.0, "std": 2.0, "coverage": 0.3, "entropy": 1.5}


def test_generate_pairs_uses_two_deltas_and_objective_label(monkeypatch):
    rendered = []

    def fake_features(renderer, params):
        rendered.append(np.asarray(params).copy())
        return {
            "mean": float(params[0]),
            "std": 0.0,
            "coverage": 1.0,
            "entropy": 0.0,
        }

    monkeypatch.setattr(pretrain_reward, "render_features", fake_features)
    rows = pretrain_reward.generate_synthetic_pairs(
        4, seed=3, renderer=lambda params: params
    )

    assert len(rows) == 4
    assert all(row["source"] == "synthetic" for row in rows)
    assert all(row["target_tissue"] in TISSUES for row in rows)
    assert all(row["direction"] in ("increase", "decrease") for row in rows)
    assert all(row["command"] == {
        "attribute": "opacity",
        "target": row["target_tissue"],
        "direction": row["direction"],
    } for row in rows)
    assert all(row["weight"] == 1.0 for row in rows)
    assert all(row["label"] in (-1, 1) for row in rows)
    assert all(set(row["observation_a"]) >= {"before_features", "after_features"}
               for row in rows)
    assert all(set(row["observation_b"]) >= {"before_features", "after_features"}
               for row in rows)
    assert len(rendered) == 12  # one shared start plus two after states per pair


def test_generate_pairs_labels_unequal_increase_objectives(monkeypatch):
    monkeypatch.setattr(
        pretrain_reward,
        "sample_start_params",
        lambda rng: np.zeros(24, dtype=np.float64),
    )
    deltas = iter((np.full(24, 0.5), np.full(24, 0.75)))
    monkeypatch.setattr(pretrain_reward, "sample_delta", lambda rng: next(deltas))
    monkeypatch.setattr(pretrain_reward, "render_features", lambda *args: FEATURES)
    monkeypatch.setattr(
        pretrain_reward,
        "mass_fraction",
        lambda params, tissue: float(params[0]),
    )
    monkeypatch.setattr(pretrain_reward, "sample_goal", lambda rng: ("bone", "increase"))
    rows = pretrain_reward.generate_synthetic_pairs(1, seed=0, renderer=lambda p: p)
    row = rows[0]
    assert row["objective_a"] == 0.5
    assert row["objective_b"] == 0.75
    assert row["label"] == -1


def test_generate_pairs_labels_unequal_decrease_objectives(monkeypatch):
    monkeypatch.setattr(
        pretrain_reward,
        "sample_start_params",
        lambda rng: np.zeros(24, dtype=np.float64),
    )
    deltas = iter((np.full(24, 0.5), np.full(24, 0.75)))
    monkeypatch.setattr(pretrain_reward, "sample_delta", lambda rng: next(deltas))
    monkeypatch.setattr(pretrain_reward, "render_features", lambda *args: FEATURES)
    monkeypatch.setattr(
        pretrain_reward,
        "mass_fraction",
        lambda params, tissue: float(params[0]),
    )
    monkeypatch.setattr(pretrain_reward, "sample_goal", lambda rng: ("bone", "decrease"))
    row = pretrain_reward.generate_synthetic_pairs(
        1, seed=0, renderer=lambda p: p
    )[0]
    assert row["objective_a"] == -0.5
    assert row["objective_b"] == -0.75
    assert row["label"] == 1


def test_pretrain_saves_aggregate_and_distinct_seeded_members(tmp_path):
    out = tmp_path / "reward_pretrained.pt"
    paths = pretrain_reward.pretrain(
        pair_count=4,
        epochs=1,
        seed=7,
        members=2,
        learning_rate=1e-3,
        output_path=out,
        renderer=lambda params: params,
        render_features_fn=lambda renderer, params: {
            "mean": float(params[0]), "std": 0.0, "coverage": 1.0, "entropy": 0.0
        },
    )

    assert out.exists()
    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    assert len({str(path) for path in paths + [out]}) == 3
    checkpoint = torch.load(out, map_location="cpu")
    assert checkpoint["member_paths"] == [str(path) for path in paths]
    assert checkpoint["seeds"] == [7, 8]


def test_pretrain_refuses_existing_artifacts(tmp_path):
    out = tmp_path / "reward_pretrained.pt"
    out.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        pretrain_reward.pretrain(
            pair_count=1,
            epochs=1,
            members=1,
            output_path=out,
            renderer=lambda params: params,
            render_features_fn=lambda renderer, params: FEATURES,
        )


def test_pretrain_refuses_existing_member_artifact(tmp_path):
    out = tmp_path / "reward_pretrained.pt"
    out.with_name("reward_pretrained_member0.pt").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        pretrain_reward.pretrain(
            pair_count=1,
            epochs=1,
            members=1,
            output_path=out,
            renderer=lambda params: params,
            render_features_fn=lambda renderer, params: FEATURES,
        )


def test_generate_pairs_resamples_duplicates_and_ties(monkeypatch):
    monkeypatch.setattr(pretrain_reward, "sample_start_params", lambda rng: np.zeros(24))
    deltas = iter((
        np.zeros(24), np.zeros(24),  # duplicate candidates
        np.full(24, 0.25), np.full(24, 0.25),  # tied objectives
        np.full(24, 0.25), np.full(24, 0.5),  # accepted pair
    ))
    monkeypatch.setattr(pretrain_reward, "sample_delta", lambda rng: next(deltas))
    monkeypatch.setattr(pretrain_reward, "sample_goal", lambda rng: ("bone", "increase"))
    monkeypatch.setattr(pretrain_reward, "render_features", lambda renderer, params: FEATURES)
    monkeypatch.setattr(pretrain_reward, "mass_fraction", lambda params, tissue: float(params[0]))

    rows = pretrain_reward.generate_synthetic_pairs(1, seed=0, renderer=lambda p: p)

    assert len(rows) == 1
    assert rows[0]["objective_a"] != rows[0]["objective_b"]


def test_fixed_split_is_seeded_and_disjoint():
    rows = [{"id": i} for i in range(10)]

    first = finetune_reward.split_pairs(rows, seed=11, test_fraction=0.2)
    second = finetune_reward.split_pairs(rows, seed=11, test_fraction=0.2)

    assert first == second
    assert {row["id"] for row in first[0]}.isdisjoint({row["id"] for row in first[1]})
    assert sorted(row["id"] for row in first[0] + first[1]) == list(range(10))


@pytest.mark.parametrize("records,test_fraction", [([{"id": 1}], 0.2), ([{"id": i} for i in range(4)], 0.99)])
def test_fixed_split_rejects_empty_partition(records, test_fraction):
    with pytest.raises(ValueError, match="empty"):
        finetune_reward.split_pairs(records, seed=0, test_fraction=test_fraction)


def test_pretrain_rejects_invalid_epochs_and_learning_rate(tmp_path):
    for kwargs in ({"epochs": 0}, {"learning_rate": 0}, {"learning_rate": -1}):
        with pytest.raises(ValueError):
            pretrain_reward.pretrain(
                pair_count=1, members=1, output_path=tmp_path / f"{len(kwargs)}.pt",
                renderer=lambda params: params,
                render_features_fn=lambda renderer, params: FEATURES,
                **kwargs,
            )


def test_finetune_saves_separate_members_and_split_metadata(tmp_path):
    pretrained = tmp_path / "reward_pretrained.pt"
    pretrain_reward.pretrain(
        pair_count=4,
        epochs=1,
        seed=7,
        members=2,
        learning_rate=1e-3,
        output_path=pretrained,
        renderer=lambda params: params,
        render_features_fn=lambda renderer, params: FEATURES,
    )
    preferences = tmp_path / "pairs.jsonl"
    records = [
        {
            "id": index,
            "source": "human",
            "target_tissue": "bone",
            "direction": "increase",
            "label": 1,
            "weight": 1.0,
            "observation_a": {"before_features": FEATURES, "after_features": FEATURES},
            "observation_b": {"before_features": FEATURES, "after_features": FEATURES},
        }
        for index in range(6)
    ]
    preferences.write_text("".join(f"{json.dumps(row)}\n" for row in records))

    output = tmp_path / "reward_finetuned.pt"
    paths = finetune_reward.finetune(
        preferences,
        pretrained,
        output_path=output,
        epochs=1,
        split_seed=11,
        test_fraction=1 / 3,
        pretrain_lr=1e-3,
    )

    assert output.exists()
    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    assert pretrained not in [output, *paths]
    checkpoint = torch.load(output, map_location="cpu")
    assert checkpoint["member_paths"] == [str(path) for path in paths]
    assert checkpoint["metadata"] == {
        "seed": 7,
        "split_seed": 11,
        "train_count": 4,
        "test_count": 2,
        "test_fraction": 1 / 3,
        "train_ids": [0, 2, 4, 5],
        "test_ids": [1, 3],
        "learning_rate": 1e-4,
    } | {
        "train_row_hashes": checkpoint["metadata"]["train_row_hashes"],
        "test_row_hashes": checkpoint["metadata"]["test_row_hashes"],
    }
    assert len(set(checkpoint["metadata"]["train_row_hashes"] +
                   checkpoint["metadata"]["test_row_hashes"])) == 6
    assert checkpoint["pretrained_path"] == str(pretrained)


def test_finetune_resolves_default_relative_member_paths_and_hashes_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pretrained = "out/rl_models/reward_pretrained.pt"
    pretrain_reward.pretrain(
        pair_count=4,
        epochs=1,
        seed=7,
        members=1,
        learning_rate=1e-3,
        output_path=pretrained,
        renderer=lambda params: params,
        render_features_fn=lambda renderer, params: FEATURES,
    )
    records = [
        {
            "source": "human",
            "target_tissue": "bone",
            "direction": "increase",
            "label": 1,
            "weight": 1.0,
            "observation_a": {"before_features": FEATURES, "after_features": FEATURES},
            "observation_b": {"before_features": FEATURES, "after_features": FEATURES},
        }
        for _ in range(5)
    ]
    preferences = tmp_path / "pairs.jsonl"
    preferences.write_text("".join(f"{json.dumps(row)}\n" for row in records))

    output = "out/rl_models/reward_finetuned.pt"
    finetune_reward.finetune(
        preferences,
        pretrained,
        output_path=output,
        epochs=1,
        split_seed=3,
        test_fraction=0.4,
        pretrain_lr=1e-3,
    )

    checkpoint = torch.load(output, map_location="cpu")
    metadata = checkpoint["metadata"]
    assert metadata["test_fraction"] == 0.4
    assert len(metadata["train_row_hashes"]) == metadata["train_count"]
    assert len(metadata["test_row_hashes"]) == metadata["test_count"]
    assert set(metadata["train_row_hashes"]).isdisjoint(metadata["test_row_hashes"])
    assert all(len(row_hash) == 64 for row_hash in metadata["train_row_hashes"] + metadata["test_row_hashes"])


def test_finetune_does_not_publish_partial_artifacts(tmp_path, monkeypatch):
    pretrained = tmp_path / "reward_pretrained.pt"
    pretrain_reward.pretrain(
        pair_count=4, epochs=1, seed=7, members=2, learning_rate=1e-3,
        output_path=pretrained, renderer=lambda params: params,
        render_features_fn=lambda renderer, params: FEATURES,
    )
    preferences = tmp_path / "pairs.jsonl"
    records = [{
        "target_tissue": "bone", "direction": "increase", "label": 1,
        "observation_a": {"before_features": FEATURES, "after_features": FEATURES},
        "observation_b": {"before_features": FEATURES, "after_features": FEATURES},
    } for _ in range(4)]
    preferences.write_text("".join(f"{json.dumps(row)}\n" for row in records))
    monkeypatch.setattr(finetune_reward, "save_reward_model", lambda *args: (_ for _ in ()).throw(RuntimeError("save failed")))

    output = tmp_path / "reward_finetuned.pt"
    with pytest.raises(RuntimeError, match="save failed"):
        finetune_reward.finetune(preferences, pretrained, output_path=output, epochs=1)

    assert not output.exists()
    assert not list(tmp_path.glob("reward_finetuned_member*.pt"))


def test_finetune_cleans_published_members_if_aggregate_publish_fails(tmp_path, monkeypatch):
    pretrained = tmp_path / "reward_pretrained.pt"
    pretrain_reward.pretrain(
        pair_count=4, epochs=1, seed=7, members=2, learning_rate=1e-3,
        output_path=pretrained, renderer=lambda params: params,
        render_features_fn=lambda renderer, params: FEATURES,
    )
    preferences = tmp_path / "pairs.jsonl"
    record = {
        "target_tissue": "bone", "direction": "increase", "label": 1,
        "observation_a": {"before_features": FEATURES, "after_features": FEATURES},
        "observation_b": {"before_features": FEATURES, "after_features": FEATURES},
    }
    preferences.write_text("".join(f"{json.dumps(record)}\n" for _ in range(4)))
    original_replace = finetune_reward.Path.replace

    def fail_aggregate_replace(path, target):
        if Path(target).name == "reward_finetuned.pt":
            raise RuntimeError("publish failed")
        return original_replace(path, target)

    monkeypatch.setattr(finetune_reward.Path, "replace", fail_aggregate_replace)
    output = tmp_path / "reward_finetuned.pt"
    with pytest.raises(RuntimeError, match="publish failed"):
        finetune_reward.finetune(preferences, pretrained, output_path=output, epochs=1)

    assert not output.exists()
    assert not list(tmp_path.glob("reward_finetuned_member*.pt"))


def test_finetune_rolls_back_members_published_before_aggregate_failure(tmp_path, monkeypatch):
    pretrained = tmp_path / "reward_pretrained.pt"
    pretrain_reward.pretrain(
        pair_count=4, epochs=1, seed=7, members=2, learning_rate=1e-3,
        output_path=pretrained, renderer=lambda params: params,
        render_features_fn=lambda renderer, params: FEATURES,
    )
    preferences = tmp_path / "pairs.jsonl"
    record = {
        "target_tissue": "bone", "direction": "increase", "label": 1,
        "observation_a": {"before_features": FEATURES, "after_features": FEATURES},
        "observation_b": {"before_features": FEATURES, "after_features": FEATURES},
    }
    preferences.write_text("".join(f"{json.dumps(record)}\n" for _ in range(4)))
    original_replace = finetune_reward.Path.replace

    def fail_after_members(path, target):
        if Path(target).name == "reward_finetuned.pt":
            raise RuntimeError("publish failed")
        return original_replace(path, target)

    monkeypatch.setattr(finetune_reward.Path, "replace", fail_after_members)
    output = tmp_path / "reward_finetuned.pt"
    with pytest.raises(RuntimeError, match="publish failed"):
        finetune_reward.finetune(preferences, pretrained, output_path=output, epochs=1)

    assert not output.exists()
    assert not list(tmp_path.glob("reward_finetuned_member*.pt"))


def test_finetune_rejects_output_path_matching_pretrained_member(tmp_path):
    pretrained = tmp_path / "reward_pretrained.pt"
    pretrain_reward.pretrain(
        pair_count=4, epochs=1, seed=7, members=1, learning_rate=1e-3,
        output_path=pretrained, renderer=lambda params: params,
        render_features_fn=lambda renderer, params: FEATURES,
    )
    preferences = tmp_path / "pairs.jsonl"
    record = {
        "target_tissue": "bone", "direction": "increase", "label": 1,
        "observation_a": {"before_features": FEATURES, "after_features": FEATURES},
        "observation_b": {"before_features": FEATURES, "after_features": FEATURES},
    }
    preferences.write_text("".join(f"{json.dumps(record)}\n" for _ in range(4)))
    member_path = tmp_path / "reward_pretrained_member0.pt"

    with pytest.raises(ValueError, match="overwrite pretrained"):
        finetune_reward.finetune(
            preferences, pretrained, output_path=member_path, epochs=1
        )


def _evaluation_row(row_id, source="human", label=1):
    return {
        "id": row_id,
        "source": source,
        "target_tissue": "bone",
        "direction": "increase",
        "label": label,
        "objective_a": 0.8,
        "objective_b": 0.2,
        "observation_a": {
            "before_features": FEATURES,
            "after_features": FEATURES,
            "before_image": f"before-{row_id}.png",
            "after_image": f"after-{row_id}.png",
        },
        "observation_b": {
            "before_features": FEATURES,
            "after_features": FEATURES,
            "before_image": f"other-before-{row_id}.png",
            "after_image": f"other-after-{row_id}.png",
        },
    }


def test_reward_only_accuracy_reports_same_rows_and_source_breakdown():
    rows = [_evaluation_row(1, "branch"), _evaluation_row(2, "thumb", -1)]
    result = eval_reward.evaluate_reward_only(
        rows,
        pretrained_predictor=lambda row: row["label"],
        finetuned_predictor=lambda row: 1,
    )

    assert result["test_row_ids"] == [1, 2]
    assert result["pretrained"]["overall_accuracy"] == 1.0
    assert result["pretrained"]["by_source"] == {"branch": 1.0, "thumb": 1.0}
    assert result["finetuned"]["overall_accuracy"] == 0.5
    assert result["finetuned"]["by_source"]["thumb"] == 0.0


def test_disagreements_preserve_audit_images_and_goal_context():
    row = _evaluation_row(7, "episode", label=1)
    records = eval_reward.find_disagreements(
        [row],
        human_predictor=lambda row: row["label"],
        objective_predictor=lambda row: -1,
    )

    assert records == [{
        "id": 7,
        "source": "episode",
        "target_tissue": "bone",
        "direction": "increase",
        "human_label": 1,
        "objective_label": -1,
        "observation_a_images": {
            "before": "before-7.png", "after": "after-7.png"
        },
        "observation_b_images": {
            "before": "other-before-7.png", "after": "other-after-7.png"
        },
    }]


def test_missing_objective_labels_are_excluded_from_disagreements():
    row = _evaluation_row(8, "episode", label=1)
    row.pop("objective_a")
    row.pop("objective_b")

    records = eval_reward.find_disagreements(
        [row], human_predictor=lambda item: item["label"],
        objective_predictor=lambda item: None,
    )

    assert records == []


def test_policy_sanity_check_compares_objective_finals_without_rendering():
    episodes = [{"id": 1}, {"id": 2}]
    result = eval_reward.policy_objective_sanity_check(
        episodes,
        policy_runner=lambda episode: [0.1, 0.4],
        baseline_runner=lambda episode: [0.1, 0.2],
    )

    assert result == {
        "n_episodes": 2,
        "policy_mean_final_objective": 0.4,
        "baseline_mean_final_objective": 0.2,
    }


def test_blind_head_to_head_randomizes_display_and_writes_separate_verdicts(tmp_path):
    verdict_path = tmp_path / "verdicts.jsonl"
    result = eval_reward.collect_blind_head_to_head(
        [{"id": 1}, {"id": 2}],
        render_policy=lambda episode: f"policy-{episode['id']}.png",
        render_baseline=lambda episode: f"baseline-{episode['id']}.png",
        seed=4,
        verdicts=["A", "tie"],
        verdict_path=verdict_path,
    )

    assert len(result) == 2
    assert {row["display_order"] for row in result} == {"policy_baseline", "baseline_policy"}
    assert all("verdict" in row for row in result)
    assert all("verdict" not in row or row["verdict"] in ("A", "B", "tie") for row in result)
    written = [json.loads(line) for line in verdict_path.read_text().splitlines()]
    assert written == result


def test_blind_verdicts_normalize_better_worse_relative_to_display_and_append(tmp_path):
    verdict_path = tmp_path / "verdicts.jsonl"
    verdict_path.write_text('{"previous": true}\n')

    result = eval_reward.collect_blind_head_to_head(
        [{"id": 1}, {"id": 2}],
        render_policy=lambda episode: f"policy-{episode['id']}.png",
        render_baseline=lambda episode: f"baseline-{episode['id']}.png",
        seed=4,
        verdicts=["better", "worse"],
        verdict_path=verdict_path,
    )

    assert [row["verdict"] for row in result] == ["A", "B"]
    written = verdict_path.read_text().splitlines()
    assert written[0] == '{"previous": true}'
    assert [json.loads(line)["verdict"] for line in written[1:]] == ["A", "B"]


def test_evaluation_resolves_member_paths_from_default_artifact_layout(tmp_path):
    aggregate = tmp_path / "out" / "rl_models" / "reward_pretrained.pt"
    member = aggregate.with_name("reward_pretrained_member0.pt")
    member.parent.mkdir(parents=True)
    member.write_bytes(b"member")

    resolved = eval_reward._resolve_member_path(
        "out/rl_models/reward_pretrained_member0.pt", aggregate
    )

    assert resolved == member


def test_finetune_resolves_stored_out_path_outside_original_cwd(tmp_path):
    project = tmp_path / "project"
    aggregate = project / "out" / "rl_models" / "reward_pretrained.pt"
    member = aggregate.with_name("reward_pretrained_member0.pt")
    member.parent.mkdir(parents=True)
    member.write_bytes(b"member")

    resolved = finetune_reward._resolve_member_path(
        "out/rl_models/reward_pretrained_member0.pt", aggregate
    )

    assert resolved == member


def test_fixed_split_metadata_is_required_and_validated():
    rows = [_evaluation_row(index) for index in range(4)]
    train, test = finetune_reward.split_pairs(rows, seed=11, test_fraction=0.5)
    metadata = {
        "train_count": len(train),
        "test_count": len(test),
        "train_row_hashes": finetune_reward._hash_subset(rows, train),
        "test_row_hashes": finetune_reward._hash_subset(rows, test),
    }

    assert eval_reward.select_fixed_test_rows(rows, metadata) == test
    with pytest.raises(ValueError, match="metadata"):
        eval_reward.select_fixed_test_rows(rows, {})


@pytest.mark.parametrize("mutation", ["count", "hash", "overlap"])
def test_fixed_split_metadata_rejects_tampering(mutation):
    rows = [_evaluation_row(index) for index in range(4)]
    train, test = finetune_reward.split_pairs(rows, seed=11, test_fraction=0.5)
    metadata = {
        "train_count": len(train),
        "test_count": len(test),
        "train_row_hashes": finetune_reward._hash_subset(rows, train),
        "test_row_hashes": finetune_reward._hash_subset(rows, test),
    }
    if mutation == "count":
        metadata["test_count"] += 1
    elif mutation == "hash":
        metadata["test_row_hashes"][0] = "0" * 64
    else:
        metadata["test_row_hashes"][0] = metadata["train_row_hashes"][0]

    with pytest.raises(ValueError, match="metadata"):
        eval_reward.select_fixed_test_rows(rows, metadata)


def test_evaluation_split_arguments_must_match_persisted_metadata():
    metadata = {"split_seed": 11, "test_fraction": 0.25}

    eval_reward.validate_split_arguments(metadata, split_seed=11, test_fraction=0.25)
    with pytest.raises(ValueError, match="split_seed"):
        eval_reward.validate_split_arguments(metadata, split_seed=12, test_fraction=0.25)
    with pytest.raises(ValueError, match="test_fraction"):
        eval_reward.validate_split_arguments(metadata, split_seed=11, test_fraction=0.5)
