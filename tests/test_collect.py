"""Tests exercise `Collector` directly and, for the HTTP layer, the router
coroutines directly (`asyncio.run(...)`) rather than through `TestClient` --
see `collect.py`'s module docstring and `tests/test_server.py`'s for why:
VTK's Cocoa render window can only be created on the true process main
thread on macOS, and `TestClient` runs the app on a background thread.
"""
import asyncio
import json
import os

import numpy as np
import pytest
from fastapi import HTTPException

import collect
import goals
import transfer
from collect import Collector, JudgeRequest, NextRequest
from rl.candidates import SOURCES, sample_item


class _StubModel:
    """Like `rl/candidates.py`'s test stub: each goal-class peak's
    height/colour maps directly to that class's visibility/brightness --
    fast and deterministic, but still responsive to every controllable
    value, unlike the real renderer-backed VisibilityModel."""

    MEASURED = {"skeleton": "skeleton", "lungs": "lungs", "soft": "organs", "vessels": "vessels"}
    SOLO_MAX = {"skeleton": 0.6, "lungs": 0.3, "organs": 0.4, "muscle": 0.1, "vessels": 0.05}

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)

    def features(self, params) -> dict:
        vis, bright = {}, {}
        for goal_class, measured in self.MEASURED.items():
            idx = goals.PEAK_INDEX[goal_class]
            peak = transfer.peak_internal(params, idx)
            vis[measured] = max(float(peak["height"]), 0.0)
            bright[measured] = float(sum(peak["rgb"]) / 3.0)
        vis["muscle"] = 0.0
        bright["muscle"] = 0.0
        coverage = min(1.0, sum(vis.values()))
        return {"vis": vis, "bright": bright, "coverage": coverage}

    def solo_max(self, name: str) -> float:
        return self.SOLO_MAX[name]


CLASSES_PRESENT = {"fake_a": ["skeleton", "lungs", "organs", "muscle", "vessels"]}
CONTRAST = {"fake_a": True}


def _patch_totalseg(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: CLASSES_PRESENT[name])
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: CONTRAST[name])


def _stub_image_fn(volume, params):
    tag = ",".join(f"{v:.4f}" for v in np.asarray(params, dtype=np.float64).tolist())
    return f"data:image/png;base64,{volume}:{tag}"


def _make_collector(monkeypatch, tmp_path, rng=None, volumes=("fake_a",), policy=None, anchor_items_fn=None):
    _patch_totalseg(monkeypatch)
    model = _StubModel()
    return Collector(
        pref_path=str(tmp_path / "vis_preferences.jsonl"),
        volumes=list(volumes),
        rng=rng if rng is not None else np.random.default_rng(0),
        policy_provider=lambda: policy,
        model_for_volume=lambda name: model,
        image_fn=_stub_image_fn,
        dataset_version_fn=lambda name: f"version:{name}",
        anchor_items_fn=anchor_items_fn if anchor_items_fn is not None else (lambda **kwargs: []),
    )


# --- next: images and no source labels --------------------------------------

def test_next_item_has_data_url_images_and_no_source_labels(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)

    item = collector.next_item("kk")

    assert set(item.keys()) == {"pair_id", "text", "kind", "volume", "a_image", "b_image",
                                 "start_image", "repeat_of"}
    for image in (item["a_image"], item["b_image"], item["start_image"]):
        assert image.startswith("data:image/png;base64,")
    payload = json.dumps(item)
    for source in SOURCES:
        assert source not in payload
    assert "objective_choice" not in payload
    assert "near_duplicate" not in payload
    assert "params" not in payload


# --- judge: one row, well-formed, returns the next item ---------------------

def test_judge_appends_one_well_formed_row_and_returns_next_item(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)
    item = collector.next_item("kk")

    next_item = collector.judge(item["pair_id"], "a", 4200)

    with open(collector.pref_path) as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])

    assert row["pair_id"] == item["pair_id"]
    assert row["rater_id"] == "kk"
    assert row["volume"] == "fake_a"
    assert row["volume_version"] == "version:fake_a"
    assert set(row["instruction"].keys()) == {"kind", "text", "targets", "goal"}
    assert len(row["instruction"]["goal"]) == 4 * len(goals.GOAL_CLASSES)
    assert len(row["start_params"]) == transfer.TOTAL_PARAMS
    assert set(row["a"].keys()) == {"params", "source"}
    assert len(row["a"]["params"]) == transfer.TOTAL_PARAMS
    assert row["a"]["source"] in SOURCES
    assert row["b"]["source"] in SOURCES
    assert row["choice"] == "a"
    assert row["objective_choice"] in ("a", "b")
    assert isinstance(row["near_duplicate"], bool)
    assert set(row["features"].keys()) == {"start", "a", "b"}
    assert row["decision_ms"] == 4200
    assert row["repeat_of"] is None
    assert isinstance(row["timestamp"], str) and row["timestamp"]

    # the response is a fresh item, not the judged one
    assert next_item["pair_id"] != item["pair_id"]
    assert set(next_item.keys()) == {"pair_id", "text", "kind", "volume", "a_image", "b_image",
                                      "start_image", "repeat_of"}


def test_rows_land_in_the_configured_path(monkeypatch, tmp_path):
    path = tmp_path / "nested" / "prefs.jsonl"
    _patch_totalseg(monkeypatch)
    model = _StubModel()
    collector = Collector(
        pref_path=str(path), volumes=["fake_a"], rng=np.random.default_rng(1),
        policy_provider=lambda: None, model_for_volume=lambda name: model,
        image_fn=_stub_image_fn, dataset_version_fn=lambda name: "v",
        anchor_items_fn=lambda **kwargs: [],
    )
    item = collector.next_item("kk")
    collector.judge(item["pair_id"], "b", 100)

    assert os.path.exists(path)
    with open(path) as f:
        assert len([line for line in f if line.strip()]) == 1


# --- validation --------------------------------------------------------------

def test_unknown_pair_id_raises(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)
    with pytest.raises(KeyError):
        collector.judge("not-a-real-pair-id", "a", 100)


def test_choice_is_validated(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)
    item = collector.next_item("kk")
    with pytest.raises(ValueError):
        collector.judge(item["pair_id"], "bogus", 100)


def test_judging_the_same_pair_id_twice_raises(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)
    item = collector.next_item("kk")
    collector.judge(item["pair_id"], "a", 100)
    with pytest.raises(KeyError):
        collector.judge(item["pair_id"], "b", 100)


# --- repeat scheduler ---------------------------------------------------------

def test_repeat_scheduler_returns_previously_seen_item_with_swapped_sides(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path, rng=np.random.default_rng(3))

    for _ in range(20):
        item = collector.next_item("kk")
        collector.judge(item["pair_id"], "a", 100)

    monkeypatch.setattr(collector, "_should_repeat", lambda rater_id: True)
    repeat_response = collector.next_item("kk")

    assert repeat_response["repeat_of"] is not None
    original_pair_id = repeat_response["repeat_of"]
    original_item = collector._items[original_pair_id]
    new_item = collector._items[repeat_response["pair_id"]]

    assert np.array_equal(new_item["a"]["params"], original_item["b"]["params"])
    assert new_item["a"]["source"] == original_item["b"]["source"]
    assert np.array_equal(new_item["b"]["params"], original_item["a"]["params"])
    assert new_item["b"]["source"] == original_item["a"]["source"]
    assert new_item["objective_choice"] == {"a": "b", "b": "a"}[original_item["objective_choice"]]


def test_no_repeat_before_the_minimum_gap(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path, rng=np.random.default_rng(3))
    for _ in range(19):
        item = collector.next_item("kk")
        collector.judge(item["pair_id"], "a", 100)
    # Fewer than REPEAT_MIN_GAP judged items so far -- must never repeat,
    # regardless of what the RNG would otherwise pick.
    assert collector._should_repeat("kk") is False


# --- anchor pool: shared items for inter-rater agreement ---------------------

def _anchor_pool(model, count=3, seed_base=100):
    return [sample_item("fake_a", model, np.random.default_rng(seed_base + i), policy=None) for i in range(count)]


def test_anchor_pool_served_in_deterministic_order_then_exhausted(monkeypatch, tmp_path):
    _patch_totalseg(monkeypatch)
    model = _StubModel()
    pool = _anchor_pool(model, count=3)
    collector = _make_collector(monkeypatch, tmp_path, anchor_items_fn=lambda **kwargs: pool)
    monkeypatch.setattr(collect, "ANCHOR_RATE", 1.0)  # always draw an anchor while the pool isn't exhausted

    served = []
    for _ in range(3):
        item = collector.next_item("kk")
        served.append(collector._anchor_id[item["pair_id"]])
    assert served == [0, 1, 2]

    # pool exhausted for "kk" -- falls back to a fresh sample
    item = collector.next_item("kk")
    assert collector._anchor_id[item["pair_id"]] is None

    # a different rater starts from the same beginning of the pool
    other_item = collector.next_item("other")
    assert collector._anchor_id[other_item["pair_id"]] == 0


def test_anchor_items_are_shown_with_correct_content_possibly_swapped(monkeypatch, tmp_path):
    _patch_totalseg(monkeypatch)
    model = _StubModel()
    pool = _anchor_pool(model, count=1)
    collector = _make_collector(monkeypatch, tmp_path, anchor_items_fn=lambda **kwargs: pool)
    monkeypatch.setattr(collect, "ANCHOR_RATE", 1.0)

    item_response = collector.next_item("kk")
    served = collector._items[item_response["pair_id"]]
    raw = pool[0]

    matches_direct = np.allclose(served["a"]["params"], raw["a"]["params"])
    matches_swapped = np.allclose(served["a"]["params"], raw["b"]["params"])
    assert matches_direct or matches_swapped
    if matches_direct:
        assert np.allclose(served["b"]["params"], raw["b"]["params"])
    else:
        assert np.allclose(served["b"]["params"], raw["a"]["params"])


def test_anchor_rows_get_the_anchor_id(monkeypatch, tmp_path):
    _patch_totalseg(monkeypatch)
    model = _StubModel()
    pool = _anchor_pool(model, count=1)
    collector = _make_collector(monkeypatch, tmp_path, anchor_items_fn=lambda **kwargs: pool)
    monkeypatch.setattr(collect, "ANCHOR_RATE", 1.0)

    item = collector.next_item("kk")
    collector.judge(item["pair_id"], "a", 100)

    with open(collector.pref_path) as f:
        row = json.loads(f.readline())
    assert row["anchor_id"] == 0


def test_non_anchor_rows_have_a_null_anchor_id(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)  # default: empty anchor pool
    item = collector.next_item("kk")
    collector.judge(item["pair_id"], "a", 100)

    with open(collector.pref_path) as f:
        row = json.loads(f.readline())
    assert row["anchor_id"] is None


def test_anchor_pool_is_generated_once_and_cached_in_memory(monkeypatch, tmp_path):
    _patch_totalseg(monkeypatch)
    model = _StubModel()
    pool = _anchor_pool(model, count=1)
    calls = []

    def anchor_items_fn(**kwargs):
        calls.append(1)
        return pool

    collector = _make_collector(monkeypatch, tmp_path, anchor_items_fn=anchor_items_fn)
    monkeypatch.setattr(collect, "ANCHOR_RATE", 1.0)

    collector.next_item("kk")   # anchor 0, the only one
    collector.next_item("kk")   # exhausted -- but still needs to consult the (cached) pool

    assert len(calls) == 1


def test_existing_repeat_rate_and_mechanics_are_unaffected_by_an_empty_anchor_pool(monkeypatch, tmp_path):
    # With the default (empty) anchor pool used by _make_collector, the new
    # anchor branch must short-circuit without consuming any extra draws from
    # `collector.rng`, so every pre-existing seeded test keeps behaving
    # exactly as before.
    collector = _make_collector(monkeypatch, tmp_path, rng=np.random.default_rng(3))
    for _ in range(20):
        item = collector.next_item("kk")
        collector.judge(item["pair_id"], "a", 100)
    monkeypatch.setattr(collector, "_should_repeat", lambda rater_id: True)
    repeat_response = collector.next_item("kk")
    assert repeat_response["repeat_of"] is not None


# --- rater metadata: role and experience -------------------------------------

def test_row_carries_rater_role_and_experience_alongside_rater_id(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)
    item = collector.next_item("kk", rater_role="radiologist", rater_experience="8 years CT")

    collector.judge(item["pair_id"], "a", 100)

    with open(collector.pref_path) as f:
        row = json.loads(f.readline())
    assert row["rater_id"] == "kk"  # unchanged, for backwards compatibility
    assert row["rater"] == {"id": "kk", "role": "radiologist", "experience": "8 years CT"}


def test_rater_role_defaults_to_none_when_never_given(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)
    item = collector.next_item("kk")

    collector.judge(item["pair_id"], "a", 100)

    with open(collector.pref_path) as f:
        row = json.loads(f.readline())
    assert row["rater"] == {"id": "kk", "role": None, "experience": None}


def test_rater_role_persists_across_later_calls_that_omit_it(monkeypatch, tmp_path):
    # The frontend only needs to send role/experience once per rater (they're
    # asked once, per the task); a later next_item call for the same rater_id
    # that omits them (e.g. the one judge() makes internally) must keep using
    # what was given earlier, not reset to unknown.
    collector = _make_collector(monkeypatch, tmp_path)
    first = collector.next_item("kk", rater_role="clinician", rater_experience="")
    next_item = collector.judge(first["pair_id"], "a", 100)  # calls next_item("kk") internally
    collector.judge(next_item["pair_id"], "b", 100)

    with open(collector.pref_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows[1]["rater"] == {"id": "kk", "role": "clinician", "experience": ""}


def test_invalid_role_raises(monkeypatch, tmp_path):
    collector = _make_collector(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        collector.next_item("kk", rater_role="surgeon")


def test_next_route_rejects_invalid_role_with_http_400(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "collector", _make_collector(monkeypatch, tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(collect.next_route(NextRequest(rater_id="kk", rater_role="surgeon")))
    assert exc_info.value.status_code == 400


def test_next_route_accepts_a_valid_role_and_experience(monkeypatch, tmp_path):
    test_collector = _make_collector(monkeypatch, tmp_path)
    monkeypatch.setattr(collect, "collector", test_collector)
    result = asyncio.run(collect.next_route(
        NextRequest(rater_id="kk", rater_role="researcher", rater_experience="RL for TFs")))
    assert "pair_id" in result
    assert test_collector._rater_meta["kk"] == {"role": "researcher", "experience": "RL for TFs"}


# --- routes, driven directly (see module docstring) --------------------------

def test_next_route_returns_item(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "collector", _make_collector(monkeypatch, tmp_path))
    result = asyncio.run(collect.next_route(NextRequest(rater_id="kk")))
    assert "pair_id" in result
    assert "policy" not in json.dumps(result)


def test_judge_route_unknown_pair_id_raises_http_400(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "collector", _make_collector(monkeypatch, tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(collect.judge_route(JudgeRequest(pair_id="nope", choice="a", decision_ms=1)))
    assert exc_info.value.status_code == 400


def test_judge_route_invalid_choice_raises_http_400(monkeypatch, tmp_path):
    test_collector = _make_collector(monkeypatch, tmp_path)
    monkeypatch.setattr(collect, "collector", test_collector)
    item = asyncio.run(collect.next_route(NextRequest(rater_id="kk")))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(collect.judge_route(JudgeRequest(pair_id=item["pair_id"], choice="bogus", decision_ms=1)))
    assert exc_info.value.status_code == 400


def test_collect_page_route_serves_the_static_file(monkeypatch, tmp_path):
    result = asyncio.run(collect.collect_page())
    assert result.path == "static/collect.html"
