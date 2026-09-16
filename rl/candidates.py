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
import math

import numpy as np

import goals
from rl.baselines import BASELINES, CONTROLLABLE
from rl.oneshot_env import build_observation

SOURCES = ("policy", "policy", "B1_current_executor", "B3_hill_climb_10", "B5_occlusion_rule", "perturbation")
NON_POLICY_SOURCES = tuple(source for source in dict.fromkeys(SOURCES) if source != "policy")

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
    params = start_params.copy()
    for group, value in zip(CONTROLLABLE, action):
        for index in group:
            params[index] = float(value)
    return params


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
