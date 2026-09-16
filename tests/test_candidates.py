import numpy as np
import pytest

import goals
import transfer
from rl import candidates
from rl.baselines import CONTROLLABLE
from rl.candidates import SOURCES, sample_item


class _StubModel:
    """Like the baselines'/oneshot_env's stub: each goal-class peak's
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


class _StubPolicy:
    """A callable policy: deterministic given the rng it is passed, so
    `sample_item` stays reproducible for a seed. Each call draws a fresh
    12-value action from `rng`, so two independent calls (policy vs policy)
    differ."""

    def __call__(self, observation, rng, deterministic):
        return rng.uniform(-1.0, 1.0, size=len(CONTROLLABLE))


class _ConstantPolicy:
    """Always proposes the same transfer function, regardless of input --
    used to force the near-duplicate path."""

    def predict(self, observation, deterministic=False):
        return np.zeros(len(CONTROLLABLE)), None


def _make_model(monkeypatch):
    _patch_totalseg(monkeypatch)
    return _StubModel()


# --- both candidates differ from the start ---------------------------------

def test_both_candidates_differ_from_the_start(monkeypatch):
    model = _make_model(monkeypatch)
    item = sample_item("fake_a", model, np.random.default_rng(7), policy=_StubPolicy())
    assert not np.allclose(item["a"]["params"], item["start_params"])
    assert not np.allclose(item["b"]["params"], item["start_params"])


# --- sources ------------------------------------------------------------

def test_sources_are_drawn_from_sources_and_repeat_only_as_policy(monkeypatch):
    model = _make_model(monkeypatch)
    for seed in range(30):
        item = sample_item("fake_a", model, np.random.default_rng(seed), policy=_StubPolicy())
        a_source, b_source = item["a"]["source"], item["b"]["source"]
        assert a_source in SOURCES
        assert b_source in SOURCES
        if a_source == b_source:
            assert a_source == "policy"


# --- near-duplicate filter ------------------------------------------------

def test_near_duplicate_filter_rejects_an_identical_pair(monkeypatch):
    model = _make_model(monkeypatch)
    # seed 4 makes `sample_item` pick ("policy", "policy") for this stub model;
    # paired with a policy that always proposes the same params, every attempt
    # produces an identical pair.
    item = sample_item("fake_a", model, np.random.default_rng(4), policy=_ConstantPolicy())
    assert item["a"]["source"] == "policy"
    assert item["b"]["source"] == "policy"
    assert item["near_duplicate"] is True
    assert np.allclose(item["a"]["params"], item["b"]["params"])


# --- objective_choice ------------------------------------------------------

def test_objective_choice_matches_the_lower_distance(monkeypatch):
    model = _make_model(monkeypatch)
    for seed in range(10):
        item = sample_item("fake_a", model, np.random.default_rng(seed), policy=_StubPolicy())
        start_agg = goals.aggregate(model.features(item["start_params"]))
        a_agg = goals.aggregate(model.features(item["a"]["params"]))
        b_agg = goals.aggregate(model.features(item["b"]["params"]))
        a_distance = goals.distance(item["instruction"]["goal"], start_agg, a_agg)
        b_distance = goals.distance(item["instruction"]["goal"], start_agg, b_agg)
        expected = "a" if a_distance <= b_distance else "b"
        assert item["objective_choice"] == expected


# --- determinism ------------------------------------------------------

def test_sampling_is_deterministic_for_a_seed(monkeypatch):
    model = _make_model(monkeypatch)
    item1 = sample_item("fake_a", model, np.random.default_rng(42), policy=_StubPolicy())
    item2 = sample_item("fake_a", model, np.random.default_rng(42), policy=_StubPolicy())
    assert item1["a"]["source"] == item2["a"]["source"]
    assert item1["b"]["source"] == item2["b"]["source"]
    assert np.allclose(item1["a"]["params"], item2["a"]["params"])
    assert np.allclose(item1["b"]["params"], item2["b"]["params"])
    assert item1["instruction"]["text"] == item2["instruction"]["text"]


# --- policy=None fallback ------------------------------------------------

def test_no_policy_falls_back_to_non_policy_sources(monkeypatch):
    model = _make_model(monkeypatch)
    for seed in range(20):
        item = sample_item("fake_a", model, np.random.default_rng(seed), policy=None)
        assert item["a"]["source"] != "policy"
        assert item["b"]["source"] != "policy"
        assert item["a"]["source"] != item["b"]["source"]
