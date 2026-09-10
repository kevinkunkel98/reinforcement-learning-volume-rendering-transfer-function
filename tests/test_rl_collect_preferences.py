"""Fast test for rl/collect_preferences.py -- no real console input, no real VTK render."""
import json

import numpy as np

import rl.collect_preferences as collect_preferences


def test_collect_preferences_writes_expected_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_load_dataset(name):
        return np.zeros((4, 4, 4)), (1.0, 1.0, 1.0)

    def fake_render(volume, params, spacing, camera):
        return object()

    def fake_grab(win):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_features(rgb):
        return {"mean": 1.0, "std": 2.0, "coverage": 0.1, "entropy": 0.5}

    def fake_human(before_path, after_path):
        return 1

    monkeypatch.setattr(collect_preferences, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(collect_preferences.render, "render", fake_render)
    monkeypatch.setattr(collect_preferences.render, "grab", fake_grab)
    monkeypatch.setattr(collect_preferences.render, "features", fake_features)
    monkeypatch.setattr(collect_preferences, "human", fake_human)

    collect_preferences.collect_preferences(n_pairs=3, rater_id="tester", seed=0,
                                             out_path="out/rlhf_preferences.jsonl")

    with open("out/rlhf_preferences.jsonl") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 3
    for row in rows:
        assert row["rater_id"] == "tester"
        assert row["label"] == 1
        assert row["before_features"] == {"mean": 1.0, "std": 2.0, "coverage": 0.1, "entropy": 0.5}
        assert "target_tissue" in row
        assert "direction" in row
        assert "delta" in row
