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
from rl.baselines import BASELINES, CONTROLLABLE, apply_controllable
from rl.oneshot_env import build_observation

SOURCES = ("policy", "policy", "B1_current_executor", "B3_hill_climb_10", "B5_occlusion_rule", "perturbation")
NON_POLICY_SOURCES = tuple(source for source in dict.fromkeys(SOURCES) if source != "policy")

ANCHOR_CACHE_PATH = "out/cache/anchor_items.json"

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


def _apply_action(start_params: np.ndarray, action: np.ndarray) -> np.ndarray:
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
                       start_agg: dict, rng: np.random.Generator, policy) -> np.ndarray:
    if source == "policy":
        observation = _observation_for(model, start_params, instruction, start_agg)
        action = _predict(policy, observation, rng, deterministic=False)
        return _apply_action(start_params, action)
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


def sample_item(volume: str, model, rng: np.random.Generator, policy=None) -> dict:
    """One preference item: a start transfer function, an instruction, and
    two candidates from different sources (see module docstring). Regenerates
    the pair (up to `MAX_ATTEMPTS` times) while the candidates are visually
    near-identical, then accepts the last attempt and marks
    `"near_duplicate"`. `objective_choice` -- the candidate with the lower
    `goals.distance` to the instruction -- is recorded for analysis but never
    shown to the rater."""
    start_params = goals.starting_params()
    start_agg = goals.aggregate(model.features(start_params))
    instruction = goals.sample_instruction(volume, model, start_agg, rng)

    a_source, b_source = _choose_sources(rng, policy)

    near_duplicate = True
    a_params = b_params = a_agg = b_agg = None
    for _ in range(MAX_ATTEMPTS):
        a_params = _sample_candidate(a_source, model, start_params, instruction, start_agg, rng, policy)
        b_params = _sample_candidate(b_source, model, start_params, instruction, start_agg, rng, policy)
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
    }


def _load_anchor_cache(cache_path: str, count: int, seed: int):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        data = json.load(f)
    if data.get("seed") != seed or data.get("count") != count:
        return None
    return [_item_from_json(entry) for entry in data["items"]]


def _save_anchor_cache(cache_path: str, count: int, seed: int, items: list) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    data = {"seed": seed, "count": count, "items": [_item_to_json(item) for item in items]}
    with open(cache_path, "w") as f:
        json.dump(data, f)


def anchor_items(count: int = 40, seed: int = 0, policy=None, volumes=None, model_for_volume=None,
                  cache_path: str = ANCHOR_CACHE_PATH) -> list:
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
    cached = _load_anchor_cache(cache_path, count, seed)
    if cached is not None:
        return cached

    if volumes is None:
        from datasets import volumes_for_split
        volumes = volumes_for_split("train") + volumes_for_split("val")
    if model_for_volume is None:
        import visibility
        model_for_volume = visibility.for_volume

    sorted_volumes = sorted(volumes)
    rng = np.random.default_rng(seed)
    model_cache = {}
    items = []
    for _ in range(count):
        volume = str(rng.choice(sorted_volumes))
        model = model_cache.get(volume)
        if model is None:
            model = model_for_volume(volume)
            model_cache[volume] = model
        items.append(sample_item(volume, model, rng, policy=policy))

    _save_anchor_cache(cache_path, count, seed, items)
    return items
