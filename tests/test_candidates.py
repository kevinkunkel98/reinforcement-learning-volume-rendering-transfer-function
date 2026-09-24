import json
import os

import numpy as np

import goals
import transfer
from rl import candidates
from rl.baselines import CONTROLLABLE
from rl.candidates import SOURCES, anchor_items, sample_item
from rl.oneshot_env import OneShotEnv


class _StubModel:
    """Like the baselines'/oneshot_env's stub: each goal-class peak's
    height/colour maps directly to that class's visibility/brightness --
    fast and deterministic, but still responsive to every controllable
    value, unlike the real renderer-backed VisibilityModel."""

    MEASURED = {goal_class: goal_class for goal_class in goals.GOAL_CLASSES}
    SOLO_MAX = {goal_class: 0.4 for goal_class in goals.GOAL_CLASSES}
    SOLO_MAX.update({"skeleton": 0.6, "lungs": 0.3, "soft": 0.4, "vessels": 0.05})

    def __init__(self):
        self.histogram = np.full(16, 1.0 / 16.0, dtype=np.float32)

    def features(self, params) -> dict:
        vis, bright = {}, {}
        for goal_class, measured in self.MEASURED.items():
            idx = goals.PEAK_INDEX[goal_class]
            peak = transfer.peak_internal(params, idx)
            vis[measured] = max(float(peak["height"]), 0.0)
            bright[measured] = float(sum(peak["rgb"]) / 3.0)
        vis["other"] = 0.0
        bright["other"] = 0.0
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


# --- observation shared with OneShotEnv -------------------------------------

def test_candidates_observation_matches_oneshot_env_observation(monkeypatch):
    # rl/candidates.py used to reimplement OneShotEnv's 57-value observation
    # layout by hand; both now go through rl.oneshot_env.build_observation,
    # so this pins them to stay identical for the same inputs.
    model = _make_model(monkeypatch)
    start_params = goals.starting_params()
    start_agg = goals.aggregate(model.features(start_params))
    instruction = goals.sample_instruction("fake_a", model, start_agg, np.random.default_rng(0))

    env = OneShotEnv(["fake_a"], model_for_volume=lambda name: model)
    env._volume = "fake_a"
    env._model = model
    env._start_params = start_params
    env._instruction = instruction
    env._start_agg = start_agg
    env_observation = env._build_observation()

    candidates_observation = candidates._observation_for(model, start_params, instruction, start_agg)

    assert np.array_equal(env_observation, candidates_observation)


# --- policy=None fallback ------------------------------------------------

def test_no_policy_falls_back_to_non_policy_sources(monkeypatch):
    model = _make_model(monkeypatch)
    for seed in range(20):
        item = sample_item("fake_a", model, np.random.default_rng(seed), policy=None)
        assert item["a"]["source"] != "policy"
        assert item["b"]["source"] != "policy"
        assert item["a"]["source"] != item["b"]["source"]


# --- anchor_items: a fixed, shared pool for inter-rater agreement -----------

def _counting_model_for_volume(model):
    """A `model_for_volume` that counts calls, so a test can tell whether
    `anchor_items` actually regenerated the pool or served it from cache."""
    calls = {"n": 0}

    def fn(name):
        calls["n"] += 1
        return model

    return fn, calls


def test_anchor_items_is_deterministic_for_the_same_seed_and_count(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    model_for_volume, _ = _counting_model_for_volume(model)

    items1 = anchor_items(count=4, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume,
                           cache_path=str(tmp_path / "a.json"))
    items2 = anchor_items(count=4, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume,
                           cache_path=str(tmp_path / "b.json"))  # different cache -> regenerated, not reused

    assert len(items1) == len(items2) == 4
    for a, b in zip(items1, items2):
        assert a["instruction"]["text"] == b["instruction"]["text"]
        assert np.allclose(a["start_params"], b["start_params"])
        assert a["a"]["source"] == b["a"]["source"]
        assert a["b"]["source"] == b["b"]["source"]
        assert np.allclose(a["a"]["params"], b["a"]["params"])
        assert np.allclose(a["b"]["params"], b["b"]["params"])


def test_anchor_items_differs_for_a_different_seed(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    model_for_volume, _ = _counting_model_for_volume(model)

    items1 = anchor_items(count=4, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume,
                           cache_path=str(tmp_path / "a.json"))
    items2 = anchor_items(count=4, seed=1, volumes=["fake_a"], model_for_volume=model_for_volume,
                           cache_path=str(tmp_path / "b.json"))

    texts1 = [item["instruction"]["text"] for item in items1]
    texts2 = [item["instruction"]["text"] for item in items2]
    params1 = [item["a"]["params"] for item in items1]
    params2 = [item["a"]["params"] for item in items2]
    assert texts1 != texts2 or any(not np.allclose(p1, p2) for p1, p2 in zip(params1, params2))


def test_anchor_items_shape_matches_sample_item(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    model_for_volume, _ = _counting_model_for_volume(model)

    [item] = anchor_items(count=1, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume,
                           cache_path=str(tmp_path / "a.json"))

    assert set(item.keys()) == {"volume", "start_params", "instruction", "a", "b",
                                "objective_choice", "near_duplicate", "features", "metadata"}
    assert item["volume"] == "fake_a"
    assert len(item["start_params"]) == transfer.TOTAL_PARAMS
    assert item["a"]["source"] in SOURCES
    assert item["b"]["source"] in SOURCES


def test_anchor_items_writes_and_reuses_the_cache_file(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    model_for_volume, calls = _counting_model_for_volume(model)
    cache_path = str(tmp_path / "anchor_items.json")

    first = anchor_items(count=3, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume,
                          cache_path=cache_path)
    assert calls["n"] > 0
    assert os.path.exists(cache_path)

    calls["n"] = 0

    def _broken_model_for_volume(name):
        raise AssertionError("should not regenerate: the cache should have been reused")

    second = anchor_items(count=3, seed=0, volumes=["fake_a"], model_for_volume=_broken_model_for_volume,
                           cache_path=cache_path)

    assert calls["n"] == 0
    assert len(second) == 3
    for a, b in zip(first, second):
        assert a["instruction"]["text"] == b["instruction"]["text"]
        assert np.allclose(a["a"]["params"], b["a"]["params"])
        assert np.allclose(a["b"]["params"], b["b"]["params"])
        assert a["a"]["source"] == b["a"]["source"]


def test_anchor_items_regenerates_when_count_differs_from_cache(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    model_for_volume, calls = _counting_model_for_volume(model)
    cache_path = str(tmp_path / "anchor_items.json")

    anchor_items(count=3, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume, cache_path=cache_path)
    calls["n"] = 0

    regenerated = anchor_items(count=5, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume,
                                cache_path=cache_path)

    assert calls["n"] > 0
    assert len(regenerated) == 5
    with open(cache_path) as f:
        data = json.load(f)
    assert data["count"] == 5


def test_anchor_items_regenerates_when_seed_differs_from_cache(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    model_for_volume, calls = _counting_model_for_volume(model)
    cache_path = str(tmp_path / "anchor_items.json")

    anchor_items(count=3, seed=0, volumes=["fake_a"], model_for_volume=model_for_volume, cache_path=cache_path)
    calls["n"] = 0

    anchor_items(count=3, seed=1, volumes=["fake_a"], model_for_volume=model_for_volume, cache_path=cache_path)

    assert calls["n"] > 0
    with open(cache_path) as f:
        data = json.load(f)
    assert data["seed"] == 1


def test_anchor_items_regenerates_cache_missing_observation_metadata(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    cache_path = str(tmp_path / "anchor_items.json")
    anchor_items(count=1, seed=0, volumes=["fake_a"], model_for_volume=lambda name: model,
                 cache_path=cache_path)
    with open(cache_path) as stream:
        data = json.load(stream)
    del data["items"][0]["metadata"]
    with open(cache_path, "w") as stream:
        json.dump(data, stream)

    regenerated = anchor_items(count=1, seed=0, volumes=["fake_a"],
                                model_for_volume=lambda name: model, cache_path=cache_path)

    assert regenerated[0]["metadata"] == candidates.observation_metadata()


def test_anchor_items_regenerates_cache_with_mismatching_observation_metadata(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    cache_path = str(tmp_path / "anchor_items.json")
    anchor_items(count=1, seed=0, volumes=["fake_a"], model_for_volume=lambda name: model,
                 cache_path=cache_path)
    with open(cache_path) as stream:
        data = json.load(stream)
    data["items"][0]["metadata"]["observation_size"] = 57
    with open(cache_path, "w") as stream:
        json.dump(data, stream)

    regenerated = anchor_items(count=1, seed=0, volumes=["fake_a"],
                                model_for_volume=lambda name: model, cache_path=cache_path)

    assert regenerated[0]["metadata"] == candidates.observation_metadata()


def _write_cache(path, items, **overrides):
    data = {"schema_version": candidates.ANCHOR_CACHE_SCHEMA_VERSION,
            "seed": 0, "count": len(items), "volumes": ["fake_a"], "items": items}
    data.update(overrides)
    with open(path, "w") as stream:
        json.dump(data, stream)


def test_anchor_items_rejects_truncated_cache(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    cache_path = str(tmp_path / "anchor_items.json")
    items = anchor_items(count=2, seed=0, volumes=["fake_a"], model_for_volume=lambda name: model,
                         cache_path=str(tmp_path / "valid.json"))
    item = items[0]
    _write_cache(cache_path, [candidates._item_to_json(item)], count=2)

    regenerated = anchor_items(count=2, seed=0, volumes=["fake_a"],
                                model_for_volume=lambda name: model, cache_path=cache_path)

    assert len(regenerated) == 2


def test_anchor_items_rejects_wrong_schema_and_volume(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    cache_path = str(tmp_path / "anchor_items.json")
    [item] = anchor_items(count=1, seed=0, volumes=["fake_a"], model_for_volume=lambda name: model,
                          cache_path=str(tmp_path / "valid.json"))
    encoded = candidates._item_to_json(item)
    _write_cache(cache_path, [encoded], schema_version="old", volumes=["other"])

    regenerated = anchor_items(count=1, seed=0, volumes=["fake_a"],
                                model_for_volume=lambda name: model, cache_path=cache_path)

    assert regenerated[0]["volume"] == "fake_a"


def test_anchor_items_rejects_malformed_item(monkeypatch, tmp_path):
    model = _make_model(monkeypatch)
    cache_path = str(tmp_path / "anchor_items.json")
    _write_cache(cache_path, [{"volume": "fake_a"}])

    regenerated = anchor_items(count=1, seed=0, volumes=["fake_a"],
                                model_for_volume=lambda name: model, cache_path=cache_path)

    assert set(regenerated[0]) >= {"volume", "start_params", "instruction", "a", "b"}
