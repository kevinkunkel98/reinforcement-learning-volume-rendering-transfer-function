"""/collect: preference items, the judgment log, repeats and the anchor pool."""
import json

import numpy as np
import pytest

import collect
import visibility
from rl.candidates import SOURCES, anchor_items, sample_item


def test_policy_items_pair_two_different_conditions(policy):
    model = visibility.for_volume("ts_s9001")
    item = sample_item("ts_s9001", model, np.random.default_rng(0), policy=policy)
    assert item["a"]["source"] != item["b"]["source"] and {item["a"]["source"], item["b"]["source"]} <= set(SOURCES)
    assert item["objective_choice"] in ("a", "b") and set(item["features"]["attainment"]) == {"a", "b"}


def test_anchor_pool_is_cached_and_stable(tmp_path):
    path = str(tmp_path / "anchors.json")
    first = anchor_items(count=3, volumes=["ts_s9001"], cache_path=path)
    second = anchor_items(count=3, volumes=["ts_s9001"], cache_path=path)
    assert [i["instruction"]["text"] for i in first] == [i["instruction"]["text"] for i in second]
    assert all(np.array_equal(a["a"]["params"], b["a"]["params"]) for a, b in zip(first, second))


def _stub_item(volume, model, rng, policy=None):
    params = np.zeros(48)
    agg = {"vis": {}, "bright": {}, "coverage": 0.0}
    return {"volume": volume, "start_params": params, "a": {"params": params, "source": "policy"},
            "b": {"params": params + 0.1, "source": "B4_hill_climb_200"}, "objective_choice": "a",
            "near_duplicate": False, "features": {"start": agg, "a": agg, "b": agg},
            "instruction": {"kind": "relative", "text": "more liver", "targets": {}, "goal": np.zeros(32)}}


@pytest.fixture
def collector(tmp_path):
    return collect.Collector(pref_path=str(tmp_path / "prefs.jsonl"), volumes=["ts_s9001"],
                             rng=np.random.default_rng(0), policy_provider=lambda: None,
                             model_for_volume=lambda name: None, image_fn=lambda volume, params: "img",
                             sample_item_fn=_stub_item, anchor_items_fn=lambda **kwargs: [])


def test_judging_appends_a_row_and_serves_the_next_item(collector):
    item = collector.next_item("rater-1", "researcher", "5 years")
    nxt = collector.judge(item["pair_id"], "a", 1234.0)
    rows = [json.loads(line) for line in open(collector.pref_path, encoding="utf-8")]
    assert len(rows) == 1 and rows[0]["choice"] == "a" and rows[0]["rater"]["role"] == "researcher"
    assert rows[0]["task_version"] == "v3" and nxt["pair_id"] != item["pair_id"]
    with pytest.raises(KeyError):
        collector.judge(item["pair_id"], "a", 1.0)
    with pytest.raises(ValueError):
        collector.judge(nxt["pair_id"], "maybe", 1.0)


def test_repeats_come_back_with_sides_swapped(collector, monkeypatch):
    monkeypatch.setattr(collect, "REPEAT_MIN_GAP", 1)
    monkeypatch.setattr(collect, "REPEAT_RATE", 1.0)
    first = collector.next_item("r")
    repeat = collector.judge(first["pair_id"], "b", 1.0)
    assert repeat["repeat_of"] == first["pair_id"] and repeat["a_image"] == first["b_image"]
