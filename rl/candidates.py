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

A checkpoint trained on task v2 or v3 (its metadata names the task) takes
`_task_item` instead: the start and the instruction come from `OneShotEnv.reset`
exactly as `rl.vis_eval` draws its episodes, and the candidates are the
conditions the thesis compares (`TASK_SOURCES`) -- the policy's deterministic
proposal, the proposal planned through its dynamics ensemble, that plan
refined with 30 or 200 renders of search (the viewer's answer), 200 renders of
search from the start, and the rule executor.
"""
import json
import math
import os

import numpy as np

import goals
import transfer
from rl.baselines import BASELINES, CONTROLLABLE, apply_controllable, hill_climb
from rl.oneshot_env import build_observation, observation_metadata, resolve_policy_metadata
from anatomy_layers import default_layers, normalize_layers

SOURCES = ("policy", "policy", "B1_current_executor", "B3_hill_climb_10", "B5_occlusion_rule", "perturbation")
NON_POLICY_SOURCES = tuple(source for source in dict.fromkeys(SOURCES) if source != "policy")
# Task v2/v3 items (`_task_item`). Two distinct sources per item, uniformly.
TASK_SOURCES = ("policy", "policy_plan", "policy_plan_30", "policy_plan_200",
                "B4_hill_climb_200", "B1_current_executor")
TASK_NON_POLICY_SOURCES = ("B4_hill_climb_200", "B1_current_executor", "perturbation")
REFINE_INITIAL_STEP = 0.1           # rl.vis_eval.REFINE_INITIAL_STEP: search warm-started at a proposal

ANCHOR_CACHE_PATH = "out/cache/anchor_items.json"
ANCHOR_CACHE_SCHEMA_VERSION = 2
INSTRUCTION_KINDS = {kind for kind, _ in goals.INSTRUCTION_MIX}

MIN_VISIBILITY_DIFFERENCE = 0.1     # log10 units, per class
MIN_BRIGHTNESS_DIFFERENCE = 0.05
MAX_ATTEMPTS = 10
PERTURBATION_NOISE = 0.3            # uniform +-, normalized parameter units, per controllable group


def _observation_for(model, start_params: np.ndarray, instruction: dict, start_agg: dict,
                     policy_metadata=None) -> np.ndarray:
    """The one-shot observation for `start_params`/`instruction` on `model`,
    via `rl.oneshot_env.build_observation` -- the single shared
    implementation of the layout `OneShotEnv` trains on, so a policy queried
    standalone here (without a live env) sees exactly the same input."""
    solo_max_log = [math.log10(sum(model.solo_max(m) for m in goals.MEASURED_FOR_GOAL[c]) + goals.EPSILON)
                     for c in goals.GOAL_CLASSES]
    controllable = [float(np.mean([start_params[i] for i in group])) for group in CONTROLLABLE]
    layer_opacity = None
    if policy_metadata and policy_metadata.get("policy_version") == "oneshot-v8":
        layer_opacity = [1.0] * len(goals.GOAL_CLASSES)
    return build_observation(instruction["goal"], model.histogram, start_agg, solo_max_log,
                             controllable, layer_opacity)


def _apply_action(start_params: np.ndarray, action: np.ndarray, action_mode: str = "absolute") -> np.ndarray:
    if action_mode == "residual":
        action = np.asarray(action) + np.asarray(
            [float(np.mean([start_params[i] for i in group])) for group in CONTROLLABLE])
    elif action_mode != "absolute":
        raise ValueError("action_mode must be absolute or residual")
    return apply_controllable(start_params, action)


def _apply_policy_action(model, volume, start_params, action, policy_metadata, active_layers=None):
    metadata = resolve_policy_metadata(policy_metadata)
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    layers = None
    if metadata["policy_version"] == "oneshot-v8":
        layers = normalize_layers(active_layers or default_layers())
        reachable = set(goals.reachable_goal_classes(volume, model, layers))
        for name, residual in zip(goals.GOAL_CLASSES, action[len(CONTROLLABLE):]):
            if name in reachable:
                layers[name]["opacity"] = float(np.clip(1.0 + residual, 0.0, 1.0))
        action = action[:len(CONTROLLABLE)]
    return _apply_action(start_params, action, metadata["action_mode"]), layers


def _features(model, params, layers=None):
    return goals.features(model, params, layers)


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
                        policy_metadata=None, volume=None) -> tuple:
    if source == "policy":
        observation = _observation_for(model, start_params, instruction, start_agg, policy_metadata)
        action = _predict(policy, observation, rng, deterministic=False)
        params, _layers = _apply_policy_action(model, volume, start_params,
                                               action, policy_metadata or {})
        return params, _layers
    if source == "perturbation":
        return _perturb(start_params, rng), None
    return np.asarray(BASELINES[source](model, start_params, instruction), dtype=np.float64), None


def _is_near_duplicate(agg_a: dict, agg_b: dict) -> bool:
    vis_close = all(
        abs(math.log10(agg_a["vis"][c] + goals.EPSILON) - math.log10(agg_b["vis"][c] + goals.EPSILON))
        < MIN_VISIBILITY_DIFFERENCE for c in goals.GOAL_CLASSES)
    bright_close = all(
        abs(agg_a["bright"][c] - agg_b["bright"][c]) < MIN_BRIGHTNESS_DIFFERENCE for c in goals.GOAL_CLASSES)
    return vis_close and bright_close


def _task_item(volume: str, model, rng: np.random.Generator, policy, policy_metadata: dict) -> dict:
    """`sample_item` for a task v2/v3 checkpoint (see the module docstring).
    Each source is computed at most once per item; a near-duplicate pair is
    redrawn as another pair of sources, up to `MAX_ATTEMPTS` times."""
    from rl.oneshot_env import OneShotEnv
    env = OneShotEnv([volume], model_for_volume=lambda _name: model, **policy_metadata)
    observation, _info = env.reset(seed=int(rng.integers(2 ** 31)))
    start_params, instruction, start_agg = env._start_params.copy(), env._instruction, env._start_agg
    weights = instruction.get("weights")
    can_plan = getattr(policy, "dynamics", None) is not None and policy_metadata["action_mode"] == "residual"
    if policy is None:
        sources = TASK_NON_POLICY_SOURCES
    else:
        sources = tuple(s for s in TASK_SOURCES if can_plan or not s.startswith("policy_plan"))

    params_by_source, agg_by_source = {}, {}

    def agg_of(params):
        return goals.aggregate(goals.features(model, params))

    def attainment_of(agg):
        return goals.attainment(instruction["goal"], start_agg, agg, weights=weights)

    def params_for(source):
        if source in params_by_source:
            return params_by_source[source]
        if source == "policy":
            action = _predict(policy, observation, rng, deterministic=True)
            params = _apply_action(start_params, action, policy_metadata["action_mode"])
        elif source == "policy_plan":
            from rl.model_based import plan
            action = _predict(policy, observation, rng, deterministic=True)
            planned = plan(policy.dynamics, instruction["goal"], weights, model.histogram,
                           env._solo_max_log(volume, model), env._controllable_values(start_params),
                           start_agg, action)
            params = _apply_action(start_params, planned, "residual")
        elif source.startswith("policy_plan_"):
            proposal = params_for("policy_plan")
            refined = hill_climb(model, proposal, instruction, evaluations=int(source.rsplit("_", 1)[1]),
                                 initial_step=REFINE_INITIAL_STEP, reference_params=start_params)
            # The viewer's contract (server.refine_policy_proposal): keep the better of the two.
            better = attainment_of(agg_of(refined)) > attainment_of(agg_of(proposal))
            params = refined if better else proposal
        elif source == "perturbation":
            params = _perturb(start_params, rng)
        else:
            params = np.asarray(BASELINES[source](model, start_params, instruction), dtype=np.float64)
        params_by_source[source] = params
        agg_by_source[source] = agg_of(params)
        return params

    near_duplicate = True
    for _ in range(MAX_ATTEMPTS):
        a_source, b_source = (str(s) for s in rng.choice(sources, size=2, replace=False))
        params_for(a_source), params_for(b_source)
        if not _is_near_duplicate(agg_by_source[a_source], agg_by_source[b_source]):
            near_duplicate = False
            break

    a_agg, b_agg = agg_by_source[a_source], agg_by_source[b_source]
    a_distance = goals.distance(instruction["goal"], start_agg, a_agg, weights=weights)
    b_distance = goals.distance(instruction["goal"], start_agg, b_agg, weights=weights)
    return {
        "volume": volume,
        "start_params": start_params,
        "instruction": instruction,
        "a": {"params": params_by_source[a_source], "source": a_source},
        "b": {"params": params_by_source[b_source], "source": b_source},
        "objective_choice": "a" if a_distance <= b_distance else "b",
        "near_duplicate": near_duplicate,
        "features": {"start": start_agg, "a": a_agg, "b": b_agg,
                     "attainment": {"a": attainment_of(a_agg), "b": attainment_of(b_agg)}},
        "metadata": observation_metadata(**policy_metadata),
    }


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
    if policy_metadata.get("task_version", "v1") != "v1":
        return _task_item(volume, model, rng, policy, policy_metadata)
    start_params = goals.starting_params()
    start_layers = default_layers() if policy_metadata["policy_version"] == "oneshot-v8" else None
    start_agg = goals.aggregate(_features(model, start_params, start_layers), start_layers)
    instruction = goals.sample_instruction(volume, model, start_agg, rng, start_layers)

    a_source, b_source = _choose_sources(rng, policy)

    near_duplicate = True
    a_params = b_params = a_agg = b_agg = None
    for _ in range(MAX_ATTEMPTS):
        a_params, a_layers = _sample_candidate(a_source, model, start_params, instruction, start_agg, rng, policy,
                                                policy_metadata, volume)
        b_params, b_layers = _sample_candidate(b_source, model, start_params, instruction, start_agg, rng, policy,
                                                policy_metadata, volume)
        a_raw = _features(model, a_params, a_layers)
        b_raw = _features(model, b_params, b_layers)
        a_agg = goals.aggregate(a_raw, a_layers)
        b_agg = goals.aggregate(b_raw, b_layers)
        if not _is_near_duplicate(a_agg, b_agg):
            near_duplicate = False
            break

    # Bare keyword: no test in this file monkeypatches goals.distance with a
    # fixed-arity stub, so there's no risk an unconditional weights= breaks.
    weights = instruction.get("weights")
    a_distance = goals.distance(instruction["goal"], start_agg, a_agg, weights=weights)
    b_distance = goals.distance(instruction["goal"], start_agg, b_agg, weights=weights)
    objective_choice = "a" if a_distance <= b_distance else "b"

    result = {
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
    if policy_metadata["policy_version"] == "oneshot-v8":
        result["a"]["layers"] = a_layers
        result["b"]["layers"] = b_layers
        result["features"]["layers"] = {"a": a_layers, "b": b_layers}
    return result


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
            **({"weights": {c: float(w) for c, w in item["instruction"]["weights"].items()}}
               if item["instruction"].get("weights") else {}),
        },
        "a": {key: (np.asarray(value, dtype=np.float64).tolist() if key == "params" else value)
              for key, value in item["a"].items()},
        "b": {key: (np.asarray(value, dtype=np.float64).tolist() if key == "params" else value)
              for key, value in item["b"].items()},
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
            **({"weights": data["instruction"]["weights"]} if data["instruction"].get("weights") else {}),
        },
        "a": {key: (np.asarray(value, dtype=np.float64) if key == "params" else value)
              for key, value in data["a"].items()},
        "b": {key: (np.asarray(value, dtype=np.float64) if key == "params" else value)
              for key, value in data["b"].items()},
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
    if not isinstance(features, dict) or not {"start", "a", "b"} <= set(features):
        return False
    for aggregate in (features["start"], features["a"], features["b"]):
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
    if (not isinstance(instruction, dict) or set(instruction) - {"weights"} != {"kind", "text", "targets", "goal"}
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
        if not isinstance(candidate, dict) or not {"params", "source"} <= set(candidate):
            return False
        if not _finite_vector(candidate["params"], transfer.TOTAL_PARAMS):
            return False
        if candidate["source"] not in SOURCES + TASK_SOURCES:
            return False
    return item["objective_choice"] in ("a", "b") and isinstance(item["near_duplicate"], bool)


def _load_anchor_cache(cache_path: str, count: int, seed: int, volumes: tuple, metadata=None, policy_id=None):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != ANCHOR_CACHE_SCHEMA_VERSION:
        return None
    if data.get("seed") != seed or data.get("count") != count or data.get("policy_id") != policy_id:
        return None
    if data.get("volumes") != list(volumes) or not isinstance(data.get("items"), list):
        return None
    if len(data["items"]) != count:
        return None
    if any(not _cache_item_is_valid(entry, volumes, metadata) for entry in data["items"]):
        return None
    return [_item_from_json(entry) for entry in data["items"]]


def _save_anchor_cache(cache_path: str, count: int, seed: int, volumes: tuple, items: list,
                       policy_id=None) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    data = {"schema_version": ANCHOR_CACHE_SCHEMA_VERSION, "seed": seed, "count": count,
            "volumes": list(volumes), "items": [_item_to_json(item) for item in items]}
    if policy_id is not None:
        data["policy_id"] = policy_id
    with open(cache_path, "w") as f:
        json.dump(data, f)


def anchor_items(count: int = 40, seed: int = 0, policy=None, volumes=None, model_for_volume=None,
                  cache_path: str = ANCHOR_CACHE_PATH, policy_metadata=None, policy_id=None) -> list:
    """`count` items (see `sample_item`) generated deterministically from
    `seed` alone -- independent of any rater's history -- so every rater on
    every machine judging the same `cache_path` sees byte-identical items.
    Cached to `cache_path`; regenerated only when the file is missing or its
    recorded `seed`/`count` differ from the ones requested here, so a
    checked-in cache is never silently invalidated by, say, a newer policy
    checkpoint (`policy` defaults to `None`, the same "baselines and
    perturbation only" fallback `sample_item` itself uses -- see its
    docstring -- keeping the pool stable even as models change). A caller
    that does pass a policy names it with `policy_id` (its checkpoint path),
    and a cache made with another `policy_id` is regenerated.

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
    cached = _load_anchor_cache(cache_path, count, seed, sorted_volumes, metadata, policy_id)
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

    _save_anchor_cache(cache_path, count, seed, sorted_volumes, items, policy_id)
    return items
