"""Preference items for /collect: a start, an instruction, and two candidate
transfer functions from different methods, judged blind.

`sample_item` draws the start and the instruction as the evaluation does
(`rl.episodes.EpisodeSampler`) and two distinct sources from `SOURCES`, the
conditions the thesis compares: the policy's proposal, the proposal planned
through its dynamics ensemble, that plan refined with 30 or 200 evaluations of
search (the viewer's answer), 200 evaluations of search from the start, and the
rule executor. Without a policy, the search, the rule executor and a random
perturbation. Source names are those stored in the preference rows.

`anchor_items` is a fixed pool every rater judges, so their agreement can be
measured; it is cached to out/cache/anchor_items.json.
"""
import json
import math
import os

import numpy as np

import goals
import transfer
from rl.episodes import EpisodeSampler, contract_metadata, controllable_values, solo_max_log
from rl.search import CONTROLLABLE, apply_controllable, hill_climb, rule_executor

SOURCES = ("policy", "policy_plan", "policy_plan_30", "policy_plan_200", "B4_hill_climb_200", "B1_current_executor")
NON_POLICY_SOURCES = ("B4_hill_climb_200", "B1_current_executor", "perturbation")
REFINE_INITIAL_STEP = 0.1           # as rl.evaluate: a search warm-started at a proposal

ANCHOR_CACHE_PATH = "out/cache/anchor_items.json"
ANCHOR_CACHE_SCHEMA_VERSION = 2
ANCHOR_COUNT = 40

MIN_VISIBILITY_DIFFERENCE = 0.1     # log10 units, per class
MIN_BRIGHTNESS_DIFFERENCE = 0.05
MAX_ATTEMPTS = 10
PERTURBATION_NOISE = 0.3            # uniform +-, normalized parameter units, per controllable group


def _perturb(start_params: np.ndarray, rng) -> np.ndarray:
    params = start_params.copy()
    for group in CONTROLLABLE:
        noise = rng.uniform(-PERTURBATION_NOISE, PERTURBATION_NOISE)
        for index in group:
            params[index] = params[index] + noise
    return np.clip(params, -1.0, 1.0)


def _is_near_duplicate(agg_a: dict, agg_b: dict) -> bool:
    vis_close = all(
        abs(math.log10(agg_a["vis"][c] + goals.EPSILON) - math.log10(agg_b["vis"][c] + goals.EPSILON))
        < MIN_VISIBILITY_DIFFERENCE for c in goals.GOAL_CLASSES)
    bright_close = all(abs(agg_a["bright"][c] - agg_b["bright"][c]) < MIN_BRIGHTNESS_DIFFERENCE
                       for c in goals.GOAL_CLASSES)
    return vis_close and bright_close


def sample_item(volume: str, model, rng, policy=None) -> dict:
    """One preference item. Each source is computed at most once; a pair that
    renders near-identically is redrawn as another pair of sources, up to
    MAX_ATTEMPTS times, then kept and marked `near_duplicate`.
    `objective_choice` -- the candidate closer to the goal -- is recorded for
    analysis, never shown."""
    episode = EpisodeSampler([volume], model_for_volume=lambda _name: model).reset(int(rng.integers(2 ** 31)))
    start_params, instruction, start_agg = episode.start_params, episode.instruction, episode.start_agg
    weights = instruction.get("weights")
    sources = NON_POLICY_SOURCES if policy is None else SOURCES
    params_by_source, agg_by_source = {}, {}

    def attainment_of(agg):
        return goals.attainment(instruction["goal"], start_agg, agg, weights)

    def proposal():
        action, _ = policy.predict(episode.observation())
        return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    def params_for(source):
        if source in params_by_source:
            return params_by_source[source]
        controllable = np.asarray(controllable_values(start_params))
        if source == "policy":
            params = apply_controllable(start_params, controllable + proposal())
        elif source == "policy_plan":
            from rl.model_based import plan
            planned = plan(policy.dynamics, instruction["goal"], weights, model.histogram, solo_max_log(model),
                           list(controllable), start_agg, proposal())
            params = apply_controllable(start_params, controllable + planned)
        elif source.startswith("policy_plan_"):
            planned = params_for("policy_plan")
            refined = hill_climb(model, planned, instruction, evaluations=int(source.rsplit("_", 1)[1]),
                                 initial_step=REFINE_INITIAL_STEP, reference_params=start_params)
            # The viewer's contract: keep the better of the two.
            better = attainment_of(episode.aggregate(refined)) > attainment_of(episode.aggregate(planned))
            params = refined if better else planned
        elif source == "perturbation":
            params = _perturb(start_params, rng)
        elif source == "B4_hill_climb_200":
            params = hill_climb(model, start_params, instruction, evaluations=200)
        else:
            params = rule_executor(model, start_params, instruction)
        params_by_source[source] = np.asarray(params, dtype=np.float64)
        agg_by_source[source] = episode.aggregate(params_by_source[source])
        return params_by_source[source]

    near_duplicate = True
    for _ in range(MAX_ATTEMPTS):
        a_source, b_source = (str(s) for s in rng.choice(sources, size=2, replace=False))
        params_for(a_source), params_for(b_source)
        if not _is_near_duplicate(agg_by_source[a_source], agg_by_source[b_source]):
            near_duplicate = False
            break

    a_agg, b_agg = agg_by_source[a_source], agg_by_source[b_source]
    a_distance = goals.distance(instruction["goal"], start_agg, a_agg, weights)
    b_distance = goals.distance(instruction["goal"], start_agg, b_agg, weights)
    return {
        "volume": volume, "start_params": start_params, "instruction": instruction,
        "a": {"params": params_by_source[a_source], "source": a_source},
        "b": {"params": params_by_source[b_source], "source": b_source},
        "objective_choice": "a" if a_distance <= b_distance else "b",
        "near_duplicate": near_duplicate,
        "features": {"start": start_agg, "a": a_agg, "b": b_agg,
                     "attainment": {"a": attainment_of(a_agg), "b": attainment_of(b_agg)}},
        "metadata": contract_metadata(),
    }


# --- anchor pool -------------------------------------------------------------------

def _array_fields(item: dict, convert) -> dict:
    instruction = item["instruction"]
    return {
        **item, "start_params": convert(item["start_params"]),
        "instruction": {**{k: instruction[k] for k in ("kind", "text", "targets")},
                        "goal": convert(instruction["goal"]),
                        **({"weights": instruction["weights"]} if instruction.get("weights") else {})},
        "a": {**item["a"], "params": convert(item["a"]["params"])},
        "b": {**item["b"], "params": convert(item["b"]["params"])},
    }


def _valid(item: dict, volumes) -> bool:
    try:
        return (item["volume"] in volumes and item["metadata"] == contract_metadata()
                and len(item["start_params"]) == transfer.TOTAL_PARAMS
                and len(item["instruction"]["goal"]) == 4 * len(goals.GOAL_CLASSES)
                and item["a"]["source"] in SOURCES + NON_POLICY_SOURCES
                and item["b"]["source"] in SOURCES + NON_POLICY_SOURCES
                and item["objective_choice"] in ("a", "b"))
    except (KeyError, TypeError):
        return False


def _load_anchor_cache(path: str, count: int, seed: int, volumes, policy_id):
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return None
    if (data.get("schema_version") != ANCHOR_CACHE_SCHEMA_VERSION or data.get("seed") != seed
            or data.get("count") != count or data.get("policy_id") != policy_id
            or data.get("volumes") != list(volumes) or len(data.get("items", [])) != count
            or not all(_valid(item, volumes) for item in data["items"])):
        return None
    return [_array_fields(item, lambda v: np.asarray(v, dtype=np.float64)) for item in data["items"]]


def anchor_items(count: int = ANCHOR_COUNT, seed: int = 0, policy=None, policy_id=None, volumes=None,
                 model_for_volume=None, cache_path: str = ANCHOR_CACHE_PATH) -> list:
    """`count` items generated from `seed` alone, the same for every rater
    on every machine. Cached to `cache_path`; regenerated when the seed, the
    count, the volumes or the policy (`policy_id`, its checkpoint path) differ."""
    if volumes is None:
        import scans
        volumes = scans.split_names("train") + scans.split_names("val")
    volumes = tuple(sorted(volumes))
    cached = _load_anchor_cache(cache_path, count, seed, volumes, policy_id)
    if cached is not None:
        return cached
    if model_for_volume is None:
        import visibility
        model_for_volume = visibility.for_volume
    rng = np.random.default_rng(seed)
    models, items = {}, []
    for _ in range(count):
        volume = str(rng.choice(volumes))
        if volume not in models:
            models[volume] = model_for_volume(volume)
        items.append(sample_item(volume, models[volume], rng, policy=policy))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as stream:
        json.dump({"schema_version": ANCHOR_CACHE_SCHEMA_VERSION, "seed": seed, "count": count,
                   "volumes": list(volumes), "policy_id": policy_id,
                   "items": [_array_fields(item, lambda v: np.asarray(v, dtype=np.float64).tolist())
                             for item in items]}, stream)
    return items
