"""Sample one preference item: a volume, a start transfer function, an
instruction, and two candidate transfer functions from different sources.

`sample_item` is the entry point `collect.py` calls per rater request. A
candidate's source is one of `SOURCES`: two independent stochastic samples of
a one-shot policy (`rl.oneshot_env.OneShotEnv`'s observation, action ->
params, see `rl.oneshot_env.build_observation`/`_apply_action`), a
non-learned baseline (`rl.baselines.BASELINES`), or a random perturbation of
the start state. The policy is injected -- a plain callable `policy(
observation, rng, deterministic) -> action` or an SB3-style model exposing
`.predict(observation, deterministic=...) -> (action, state)` -- so this
module never hard-codes a checkpoint path, and `policy=None` still produces
usable items from the non-policy sources alone.
"""
import json
import math
import os

import numpy as np

import goals
import transfer
from rl.baselines import BASELINES, CONTROLLABLE, apply_controllable
from rl.oneshot_env import build_observation, observation_metadata, resolve_policy_metadata

SOURCES = ("policy", "policy", "B1_current_executor", "B3_hill_climb_10", "B5_occlusion_rule", "perturbation")
NON_POLICY_SOURCES = tuple(source for source in dict.fromkeys(SOURCES) if source != "policy")

ANCHOR_CACHE_PATH = "out/cache/anchor_items.json"
ANCHOR_CACHE_SCHEMA_VERSION = 2
INSTRUCTION_KINDS = {kind for kind, _ in goals.INSTRUCTION_MIX}

MIN_VISIBILITY_DIFFERENCE = 0.1     # log10 units, per class
MIN_BRIGHTNESS_DIFFERENCE = 0.05
MAX_ATTEMPTS = 10
PERTURBATION_NOISE = 0.3            # uniform +-, normalized parameter units, per controllable group


def _observation_for(model, start_params: np.ndarray, instruction: dict, start_agg: dict) -> np.ndarray:
    """The one-shot observation for `start_params`/`instruction` on `model`,
    via `rl.oneshot_env.build_observation` -- the single shared
    implementation of the layout `OneShotEnv` trains on, so a policy queried
    standalone here (without a live env) sees exactly the same input."""
    solo_max_log = [math.log10(sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[c]) + goals.EPSILON)
                     for c in goals.GOAL_CLASSES]
    controllable = [float(np.mean([start_params[i] for i in group])) for group in CONTROLLABLE]
    return build_observation(instruction["goal"], model.histogram, start_agg, solo_max_log, controllable)


def _apply_action(start_params: np.ndarray, action: np.ndarray, action_mode: str = "absolute") -> np.ndarray:
    if action_mode == "residual":
        action = np.asarray(action) + np.asarray(
            [float(np.mean([start_params[i] for i in group])) for group in CONTROLLABLE])
    elif action_mode != "absolute":
        raise ValueError("action_mode must be absolute or residual")
    return apply_controllable(start_params, action)


def _predict(policy, observation: np.ndarray, rng: np.random.Generator, deterministic: bool) -> np.ndarray:
    if hasattr(policy, "predict"):
        action, _ = policy.predict(observation, deterministic=deterministic)
    else:
        action = policy(observation, rng, deterministic)
    return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)


def _perturb(start_params: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    params = start_params.copy()
    for group in CONTROLLABLE:
        noise = rng.uniform(-PERTURBATION_NOISE, PERTURBATION_NOISE)
        for index in group:
            params[index] = params[index] + noise
    return np.clip(params, -1.0, 1.0)


def _choose_sources(rng: np.random.Generator, policy) -> tuple:
    """Half the time, pair two stochastic policy samples; the other half,
    pair one policy sample against a uniformly chosen non-policy source.
    With no policy available, fall back to two distinct non-policy sources."""
    if policy is None:
        chosen = rng.choice(NON_POLICY_SOURCES, size=2, replace=False)
        return str(chosen[0]), str(chosen[1])
    if rng.random() < 0.5:
        return "policy", "policy"
    return "policy", str(rng.choice(NON_POLICY_SOURCES))


def _sample_candidate(source: str, model, start_params: np.ndarray, instruction: dict,
                       start_agg: dict, rng: np.random.Generator, policy,
                       policy_metadata=None) -> np.ndarray:
    if source == "policy":
        observation = _observation_for(model, start_params, instruction, start_agg)
        action = _predict(policy, observation, rng, deterministic=False)
        return _apply_action(start_params, action,
                             (policy_metadata or {}).get("action_mode", "absolute"))
    if source == "perturbation":
        return _perturb(start_params, rng)
    return np.asarray(BASELINES[source](model, start_params, instruction), dtype=np.float64)


def _is_near_duplicate(agg_a: dict, agg_b: dict) -> bool:
    vis_close = all(
        abs(math.log10(agg_a["vis"][c] + goals.EPSILON) - math.log10(agg_b["vis"][c] + goals.EPSILON))
        < MIN_VISIBILITY_DIFFERENCE for c in goals.GOAL_CLASSES)
    bright_close = all(
        abs(agg_a["bright"][c] - agg_b["bright"][c]) < MIN_BRIGHTNESS_DIFFERENCE for c in goals.GOAL_CLASSES)
    return vis_close and bright_close


def sample_item(volume: str, model, rng: np.random.Generator, policy=None,
                policy_metadata=None) -> dict:
    """One preference item: a start transfer function, an instruction, and
    two candidates from different sources (see module docstring). Regenerates
    the pair (up to `MAX_ATTEMPTS` times) while the candidates are visually
    near-identical, then accepts the last attempt and marks
    `"near_duplicate"`. `objective_choice` -- the candidate with the lower
    `goals.distance` to the instruction -- is recorded for analysis but never
    shown to the rater."""
    policy_metadata = resolve_policy_metadata(policy_metadata)
    start_params = goals.starting_params()
    start_agg = goals.aggregate(model.features(start_params))
    instruction = goals.sample_instruction(volume, model, start_agg, rng)

    a_source, b_source = _choose_sources(rng, policy)

    near_duplicate = True
    a_params = b_params = a_agg = b_agg = None
    for _ in range(MAX_ATTEMPTS):
        a_params = _sample_candidate(a_source, model, start_params, instruction, start_agg, rng, policy,
                                     policy_metadata)
        b_params = _sample_candidate(b_source, model, start_params, instruction, start_agg, rng, policy,
                                     policy_metadata)
        a_agg = goals.aggregate(model.features(a_params))
        b_agg = goals.aggregate(model.features(b_params))
        if not _is_near_duplicate(a_agg, b_agg):
            near_duplicate = False
            break

    a_distance = goals.distance(instruction["goal"], start_agg, a_agg)
    b_distance = goals.distance(instruction["goal"], start_agg, b_agg)
    objective_choice = "a" if a_distance <= b_distance else "b"

    return {
        "volume": volume,
        "start_params": start_params,
        "instruction": instruction,
        "a": {"params": a_params, "source": a_source},
        "b": {"params": b_params, "source": b_source},
        "objective_choice": objective_choice,
        "near_duplicate": near_duplicate,
        "features": {"start": start_agg, "a": a_agg, "b": b_agg},
        "metadata": observation_metadata(**policy_metadata),
    }


# --- anchor pool: a fixed set of items every rater judges --------------------
#
# Inter-rater agreement is only measurable when different raters judge the
# *same* items. `anchor_items` produces a small, deterministic pool of them --
# same volumes, start parameters, instructions and candidate pairs for every
# rater on every machine -- and caches it to disk so it never silently drifts
# (e.g. because a newer policy checkpoint changed what "policy" proposes: the
# default `policy=None` keeps the pool on baselines/perturbation only, stable
# across the life of the project).

def _item_to_json(item: dict) -> dict:
    """`sample_item`'s output (numpy arrays), narrowed and converted for
    `json.dump` -- the inverse of `_item_from_json`."""
    return {
        "volume": item["volume"],
        "start_params": np.asarray(item["start_params"], dtype=np.float64).tolist(),
        "instruction": {
            "kind": item["instruction"]["kind"],
            "text": item["instruction"]["text"],
            "targets": item["instruction"]["targets"],
            "goal": np.asarray(item["instruction"]["goal"], dtype=np.float64).tolist(),
        },
        "a": {"params": np.asarray(item["a"]["params"], dtype=np.float64).tolist(), "source": item["a"]["source"]},
        "b": {"params": np.asarray(item["b"]["params"], dtype=np.float64).tolist(), "source": item["b"]["source"]},
        "objective_choice": item["objective_choice"],
        "near_duplicate": bool(item["near_duplicate"]),
        "features": item["features"],
        "metadata": item["metadata"],
    }


def _item_from_json(data: dict) -> dict:
    """The inverse of `_item_to_json` -- restores a `sample_item`-shaped
    dict (numpy arrays back where `collect.py`'s `_to_displayed` expects
    them) from one cached JSON entry."""
    return {
        "volume": data["volume"],
        "start_params": np.asarray(data["start_params"], dtype=np.float64),
        "instruction": {
            "kind": data["instruction"]["kind"],
            "text": data["instruction"]["text"],
            "targets": data["instruction"]["targets"],
            "goal": np.asarray(data["instruction"]["goal"], dtype=np.float64),
        },
        "a": {"params": np.asarray(data["a"]["params"], dtype=np.float64), "source": data["a"]["source"]},
        "b": {"params": np.asarray(data["b"]["params"], dtype=np.float64), "source": data["b"]["source"]},
        "objective_choice": data["objective_choice"],
        "near_duplicate": data["near_duplicate"],
        "features": data["features"],
        "metadata": data["metadata"],
    }


def _finite_vector(value, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def _valid_features(features: dict) -> bool:
    if not isinstance(features, dict) or set(features) != {"start", "a", "b"}:
        return False
    for aggregate in features.values():
        if not isinstance(aggregate, dict) or set(aggregate) != {"vis", "bright", "coverage"}:
            return False
        if not isinstance(aggregate["vis"], dict) or not isinstance(aggregate["bright"], dict):
            return False
        measured_classes = set(goals.GOAL_CLASSES) | {"other"}
        if set(aggregate["vis"]) != measured_classes:
            return False
        if set(aggregate["bright"]) != set(goals.GOAL_CLASSES):
            return False
        if not all(isinstance(value, (int, float)) and np.isfinite(value)
                   for values in (aggregate["vis"], aggregate["bright"])
                   for value in values.values()):
            return False
        if not isinstance(aggregate["coverage"], (int, float)) or not np.isfinite(aggregate["coverage"]):
            return False
    return True


def _cache_item_is_valid(item: dict, volumes: tuple, metadata=None) -> bool:
    if not isinstance(item, dict) or set(item) != {
        "volume", "start_params", "instruction", "a", "b", "objective_choice",
        "near_duplicate", "features", "metadata"}:
        return False
    if item["volume"] not in volumes or item["metadata"] != observation_metadata(**resolve_policy_metadata(metadata)):
        return False
    if not _finite_vector(item["start_params"], transfer.TOTAL_PARAMS):
        return False
    instruction = item["instruction"]
    if (not isinstance(instruction, dict) or set(instruction) != {"kind", "text", "targets", "goal"}
            or not isinstance(instruction["kind"], str)
            or instruction["kind"] not in INSTRUCTION_KINDS
            or not isinstance(instruction["text"], (str, type(None)))
            or not isinstance(instruction["targets"], dict)):
        return False
    if not _finite_vector(instruction["goal"], 4 * len(goals.GOAL_CLASSES)):
        return False
    for goal_class, target in instruction["targets"].items():
        if goal_class not in goals.GOAL_CLASSES or not isinstance(target, dict):
            return False
        if set(target) - {"vis", "bright"} or not set(target):
            return False
        if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in target.values()):
            return False
    if not _valid_features(item["features"]):
        return False
    for candidate in (item["a"], item["b"]):
        if not isinstance(candidate, dict) or set(candidate) != {"params", "source"}:
            return False
        if not _finite_vector(candidate["params"], transfer.TOTAL_PARAMS):
            return False
        if candidate["source"] not in SOURCES:
            return False
    return item["objective_choice"] in ("a", "b") and isinstance(item["near_duplicate"], bool)


def _load_anchor_cache(cache_path: str, count: int, seed: int, volumes: tuple, metadata=None):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != ANCHOR_CACHE_SCHEMA_VERSION:
        return None
    if data.get("seed") != seed or data.get("count") != count:
        return None
    if data.get("volumes") != list(volumes) or not isinstance(data.get("items"), list):
        return None
    if len(data["items"]) != count:
        return None
    if any(not _cache_item_is_valid(entry, volumes, metadata) for entry in data["items"]):
        return None
    return [_item_from_json(entry) for entry in data["items"]]


def _save_anchor_cache(cache_path: str, count: int, seed: int, volumes: tuple, items: list) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    data = {"schema_version": ANCHOR_CACHE_SCHEMA_VERSION, "seed": seed, "count": count,
            "volumes": list(volumes), "items": [_item_to_json(item) for item in items]}
    with open(cache_path, "w") as f:
        json.dump(data, f)


def anchor_items(count: int = 40, seed: int = 0, policy=None, volumes=None, model_for_volume=None,
                  cache_path: str = ANCHOR_CACHE_PATH, policy_metadata=None) -> list:
    """`count` items (see `sample_item`) generated deterministically from
    `seed` alone -- independent of any rater's history -- so every rater on
    every machine judging the same `cache_path` sees byte-identical items.
    Cached to `cache_path`; regenerated only when the file is missing or its
    recorded `seed`/`count` differ from the ones requested here, so a
    checked-in cache is never silently invalidated by, say, a newer policy
    checkpoint (`policy` defaults to `None`, the same "baselines and
    perturbation only" fallback `sample_item` itself uses -- see its
    docstring -- keeping the pool stable even as models change).

    `volumes`/`model_for_volume` default to the real dataset split and
    `visibility.for_volume` (imported lazily so importing this module never
    depends on the TotalSegmentator data being present); both are injectable
    so tests never touch either.
    """
    if volumes is None:
        from datasets import volumes_for_split
        volumes = volumes_for_split("train") + volumes_for_split("val")
    sorted_volumes = tuple(sorted(volumes))
    metadata = resolve_policy_metadata(policy_metadata)
    cached = _load_anchor_cache(cache_path, count, seed, sorted_volumes, metadata)
    if cached is not None:
        return cached

    if model_for_volume is None:
        import visibility
        model_for_volume = visibility.for_volume

    rng = np.random.default_rng(seed)
    model_cache = {}
    items = []
    for _ in range(count):
        volume = str(rng.choice(sorted_volumes))
        model = model_cache.get(volume)
        if model is None:
            model = model_for_volume(volume)
            model_cache[volume] = model
        items.append(sample_item(volume, model, rng, policy=policy, policy_metadata=metadata))

    _save_anchor_cache(cache_path, count, seed, sorted_volumes, items)
    return items
