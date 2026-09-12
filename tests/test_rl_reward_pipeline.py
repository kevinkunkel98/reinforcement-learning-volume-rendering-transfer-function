import numpy as np
import torch

from rl import pretrain_reward
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
