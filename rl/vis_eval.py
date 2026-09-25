"""Evaluate a trained policy against the non-learned baselines on a fixed,
seeded set of episodes.

    .venv/bin/python -m rl.vis_eval --policy out/rl_v2/seed0/best.zip --split test --episodes 200

`fixed_episodes` materializes the same episodes (volume, noised start
parameters, sampled instruction) every time it is called with the same
arguments. The policy is scored by driving a `VisibilityTFEnv` loaded
directly into that exact state (bypassing `reset()`'s randomness, so replay
is exact independent of which volume list the episode was generated from)
and stepping with the deterministic action for `MAX_STEPS`. Every baseline
gets the identical `(model, start_params, instruction)` and is scored with
`goals.attainment` -- so policy and baselines are measured identically.
`compare` reports robust attainment stats per method (median, mean clipped to
[-1, 1], raw mean, share of episodes that improved on doing nothing -- see
`goals.summarise_attainment`) overall and per instruction kind, and a paired
Wilcoxon signed-rank test of the policy against each baseline (on the raw,
unclipped paired values), computed directly (`scipy` is not installed in this
environment).

Every result file carries a `provenance` block (see `provenance.py`) naming
the commit and the scoring code that produced it, and both the run and

    .venv/bin/python -m rl.vis_eval --show out/rl_v2/eval_test.json

warn loudly when those numbers no longer match the code in this checkout.
"""
import argparse
import collections
import json
import math
import os

import numpy as np

import datasets
import goals
import provenance
import visibility
from rl.baselines import BASELINES, hill_climb
from rl.oneshot_env import OneShotEnv, observation_metadata
from rl.vis_env import MAX_STEPS, VisibilityTFEnv, _ATTAINMENT_FLOOR

EVAL_KINDS = tuple(kind for kind, _ in goals.INSTRUCTION_MIX)
DEFAULT_EPISODES = 300
DEFAULT_OUT_TEMPLATE = "out/rl_v2/eval_{split}.json"
DEFAULT_FORMULATION = "one_shot"

_ENV_FOR_FORMULATION = {"multi_step": VisibilityTFEnv, "one_shot": OneShotEnv}


# --- fixed episodes ----------------------------------------------------------

def fixed_episodes(split: str, count: int, seed: int = 0,
                    volume_ids=None, model_for_volume=None,
                    formulation: str = "multi_step") -> list:
    """A deterministic list of `count` `{"volume", "start_params",
    "instruction"}` episodes drawn from `split` (or `volume_ids`, for tests):
    one `reset(seed=seed + i)` per episode, recording the exact state it
    produced. `formulation` picks which env generates the episodes --
    `"multi_step"` (`VisibilityTFEnv`, the default) or `"one_shot"`
    (`OneShotEnv`, which uses its own reset-noise convention) -- but the
    episode record's shape is the same either way, so baselines (and
    `compare`) score both formulations identically. Calling this again with
    the same arguments reproduces the same list -- that is what makes
    evaluation runs comparable across policies and seeds."""
    ids = volume_ids if volume_ids is not None else datasets.volumes_for_split(split)
    kwargs = {} if model_for_volume is None else {"model_for_volume": model_for_volume}
    env_cls = _ENV_FOR_FORMULATION[formulation]
    env = env_cls(list(ids), **kwargs)

    episodes = []
    for i in range(count):
        obs, info = env.reset(seed=seed + i)
        episodes.append({
            "volume": info["volume"],
            "start_params": env._params.copy(),
            "instruction": env._instruction,
        })
    return episodes


# --- running the policy and the baselines -------------------------------------

def _load_sac(path: str):
    from stable_baselines3 import SAC
    return SAC.load(path)


def _features(model, params, active_layers=None):
    return (model.features(params, active_layers) if active_layers is not None
            else model.features(params))


def _frozen_episode_env(volume: str, start_params: np.ndarray, instruction: dict,
                         model_for_volume=visibility.for_volume, active_layers=None):
    """A `VisibilityTFEnv` loaded directly into one episode's exact state,
    instead of relying on `reset(seed=...)` reproducing it -- exact and
    independent of the length/order of whatever volume list the episode was
    originally generated from."""
    env = VisibilityTFEnv([volume], model_for_volume=model_for_volume)
    model = model_for_volume(volume)
    raw_features = _features(model, start_params, active_layers)
    start_agg = goals.aggregate(raw_features, active_layers)
    start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

    env._volume = volume
    env._model = model
    env._params = np.asarray(start_params, dtype=np.float64).copy()
    env._instruction = instruction
    env._start_agg = start_agg
    env._start_distance = start_distance
    env._prev_distance = start_distance
    env._step_count = 0
    env._active_layers = active_layers

    obs = env._build_observation(env._params, start_agg)
    info = env._info(start_distance, start_agg, goals.is_useless(raw_features, active_layers))
    return env, obs, info


def _frozen_one_shot_env(volume: str, start_params: np.ndarray, instruction: dict,
                          model_for_volume=visibility.for_volume, active_layers=None):
    """A `OneShotEnv` loaded directly into one episode's exact state, the
    `OneShotEnv` counterpart of `_frozen_episode_env` above."""
    env = OneShotEnv([volume], model_for_volume=model_for_volume)
    model = model_for_volume(volume)
    raw_features = _features(model, start_params, active_layers)
    start_agg = goals.aggregate(raw_features, active_layers)
    start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

    env._volume = volume
    env._model = model
    env._start_params = np.asarray(start_params, dtype=np.float64).copy()
    env._params = env._start_params
    env._instruction = instruction
    env._start_agg = start_agg
    env._start_distance = start_distance
    env._active_layers = active_layers

    obs = env._build_observation()
    info = env._info(env._attainment(start_agg), goals.is_useless(raw_features, active_layers))
    return env, obs, info


def run_policy(model_path: str, episodes: list, model_for_volume=None, load_model=None,
               formulation: str = "multi_step", active_layers=None) -> list:
    """Run the policy at `model_path` (an SB3 checkpoint) on every episode
    with the deterministic action; return one `{"attainment", "kind"}` dict
    per episode, from the final step's info. For `formulation="multi_step"`
    (the default) that drives a frozen `VisibilityTFEnv` for up to
    `MAX_STEPS` steps; for `"one_shot"`, a frozen `OneShotEnv` for exactly
    one step, since the action there already is the final transfer function.
    `load_model` lets tests inject a stub instead of loading a real
    checkpoint."""
    loader = load_model or _load_sac
    model = loader(model_path)
    kwargs = {} if model_for_volume is None else {"model_for_volume": model_for_volume}

    results = []
    for episode in episodes:
        if formulation == "one_shot":
            env, obs, info = _frozen_one_shot_env(
                episode["volume"], episode["start_params"], episode["instruction"],
                active_layers=active_layers, **kwargs)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            env, obs, info = _frozen_episode_env(
                episode["volume"], episode["start_params"], episode["instruction"],
                active_layers=active_layers, **kwargs)
            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        results.append({"attainment": info["attainment"], "kind": info["kind"]})
    return results


def run_policy_with_refinement(model_path: str, episodes: list, evaluations: int = 3,
                                model_for_volume=None, load_model=None, active_layers=None) -> list:
    """Run the one-shot policy's proposal (as `run_policy` does for
    `formulation="one_shot"`), then, when `evaluations > 0`, refine it with
    `rl.baselines.hill_climb`'s coordinate search -- the same search the
    B3/B4 baselines use -- starting from that proposal and budgeted at
    `evaluations` visibility evaluations (matching `hill_climb`'s own
    `evaluations` contract: one call to score its start state, one per
    proposal after that). Returns one `{"attainment", "kind"}` dict per
    episode, from whichever of {proposal, refined} scores higher, so this is
    directly comparable to `run_policy`'s one-shot results.

    `evaluations=0` skips the search entirely and returns exactly what
    `run_policy` would.

    `hill_climb` keeps the best state *it* has seen, but relative to its own
    reference point: the proposal, passed to it as `start_params`, not the
    episode's original start. An instruction's goal deltas are fixed,
    absolute log10/brightness changes computed once relative to the
    original start; reapplied by `hill_climb` relative to a start that has
    already made some of that progress, the search can chase past the goal
    and, once scored back against the original start (as `run_policy`
    scores the plain proposal), come out worse than the proposal it began
    from. So this function does not trust `hill_climb`'s internal notion of
    "improved" -- it independently scores the refined result against the
    same original-start baseline `run_policy` uses via `goals.attainment`,
    and keeps whichever of {proposal, refined} is better itself.
    """
    loader = load_model or _load_sac
    model = loader(model_path)
    kwargs = {} if model_for_volume is None else {"model_for_volume": model_for_volume}

    results = []
    for episode in episodes:
        env, obs, info = _frozen_one_shot_env(
            episode["volume"], episode["start_params"], episode["instruction"],
            active_layers=active_layers, **kwargs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if evaluations > 0:
            search_kwargs = {} if active_layers is None else {"active_layers": active_layers}
            refined_params = hill_climb(env._model, env._params, episode["instruction"],
                                         evaluations=evaluations, **search_kwargs)
            refined_agg = goals.aggregate(_features(env._model, refined_params, active_layers), active_layers)
            refined_attainment = env._attainment(refined_agg)
            if refined_attainment > info["attainment"]:
                info = {**info, "attainment": refined_attainment}

        results.append({"attainment": info["attainment"], "kind": info["kind"]})
    return results


def _attainment_or_zero(goal, start_agg: dict, final_agg: dict) -> float:
    # Same floor as VisibilityTFEnv._attainment: when the start state
    # already meets the goal, distance(start) is ~0 and attainment's ratio
    # is meaningless -- report 0 instead of dividing by ~0, so the policy
    # and the baselines are floored identically.
    start_distance = goals.distance(goal, start_agg, start_agg)
    if start_distance <= _ATTAINMENT_FLOOR:
        return 0.0
    return goals.attainment(goal, start_agg, final_agg)


def run_baseline(name: str, episodes: list, model_for_volume=None, active_layers=None) -> list:
    """Run baseline `name` (from `rl.baselines.BASELINES`) on every episode;
    return one `{"attainment", "kind"}` dict per episode. `attainment` is
    `None` when the baseline function raises on that episode, so `compare`
    can drop just that episode instead of the whole run failing."""
    baseline_fn = BASELINES[name]
    get_model = model_for_volume or visibility.for_volume

    results = []
    for episode in episodes:
        instruction = episode["instruction"]
        try:
            model = get_model(episode["volume"])
            start_agg = goals.aggregate(_features(model, episode["start_params"], active_layers), active_layers)
            final_params = baseline_fn(model, episode["start_params"], instruction)
            final_agg = goals.aggregate(_features(model, final_params, active_layers), active_layers)
            attainment = _attainment_or_zero(instruction["goal"], start_agg, final_agg)
        except Exception:
            attainment = None
        results.append({"attainment": attainment, "kind": instruction["kind"]})
    return results


# --- paired Wilcoxon signed-rank test (normal approximation) -----------------

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon(x, y) -> dict:
    """Paired Wilcoxon signed-rank test of `x` against `y`, via the normal
    approximation with continuity correction (`scipy` is not installed
    here). Pairs with `x_i == y_i` (zero difference) are dropped before
    ranking; among the remaining differences, equal `|difference|` values
    share the average of the ranks they would occupy sorted ascending.

    Returns `{"statistic", "p_value", "n"}`, where `n` is the number of
    non-zero differences and `statistic = min(W+, W-)`, the smaller of the
    rank sums on either side of zero. If every pair is tied (`n == 0`, e.g.
    `x` and `y` are identical), there is no evidence of a difference:
    `p_value = 1.0`.

    Worked example, verified in `tests/test_vis_eval.py`
    (`test_wilcoxon_matches_a_hand_computed_example`):

        x - y = [3, -1, 2, -4, 5]     (n = 5, no zero differences, no ties)
        |diff| ranks, ascending and all distinct:
            |1| -> 1   |2| -> 2   |3| -> 3   |4| -> 4   |5| -> 5
        W+ (ranks of positive diffs: 3, 2, 5) = 3 + 2 + 5 = 10
        W- (ranks of negative diffs: -1, -4)  = 1 + 4     = 5
        statistic = min(W+, W-) = 5
        mean  = n(n+1)/4              = 5*6/4        = 7.5
        sigma = sqrt(n(n+1)(2n+1)/24) = sqrt(5*6*11/24) = sqrt(13.75) ~= 3.70810
        continuity-corrected z = (statistic - mean + 0.5) / sigma
                               = (5 - 7.5 + 0.5) / 3.70810 ~= -0.53936
        p_value = 2 * (1 - Phi(|z|)) ~= 0.58964
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must be paired (same shape), got {x.shape} and {y.shape}")

    diffs = x - y
    nonzero = diffs[diffs != 0.0]
    n = nonzero.shape[0]
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}

    abs_diffs = np.abs(nonzero)
    order = np.argsort(abs_diffs, kind="stable")
    sorted_abs = abs_diffs[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0   # average of 1-based ranks [i+1, j+1]
        i = j + 1

    w_pos = float(np.sum(ranks[nonzero > 0.0]))
    w_neg = float(np.sum(ranks[nonzero < 0.0]))
    statistic = min(w_pos, w_neg)

    mean = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    diff_from_mean = statistic - mean
    if diff_from_mean > 0.0:
        adjusted = diff_from_mean - 0.5
    elif diff_from_mean < 0.0:
        adjusted = diff_from_mean + 0.5
    else:
        adjusted = 0.0
    z = adjusted / sigma
    p_value = min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z))))

    return {"statistic": statistic, "p_value": p_value, "n": n}


# --- comparison table ----------------------------------------------------------

def _summarize(rows: list) -> dict:
    # Robust stats (median, mean_clipped, mean_raw, share_positive, n) from
    # goals.summarise_attainment -- attainment is unbounded below, so a plain
    # mean is misleading (see that function's docstring). The per-kind
    # breakdown used to be a plain mean, on the grounds that it only had to
    # spot weak kinds; it did the opposite. One episode at -100 on an easy
    # goal made a kind whose median was +0.51 read as -23.5, and an hour went
    # into chasing a kind that was fine. So each kind now reports median
    # first, with the mean kept beside it rather than dropped.
    attainments = [row["attainment"] for row in rows]
    stats = goals.summarise_attainment(attainments)
    by_kind = {}
    for kind in EVAL_KINDS:
        values = [row["attainment"] for row in rows if row["kind"] == kind and row["attainment"] is not None]
        if not values:
            by_kind[kind] = {"median": None, "mean": None, "share_positive": None, "n": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        by_kind[kind] = {"median": float(np.median(array)),
                         "mean": float(np.mean(array)),
                         "share_positive": float(np.mean(array > 0.0)),
                         "n": int(array.size)}
    return {**stats, "by_kind": by_kind}


def kind_stats(value) -> dict:
    """One `by_kind` entry as `{"median", "mean", "share_positive", "n"}`,
    whatever shape it was written in.

    Result files in `out/rl_v2/` written before per-kind medians existed store
    a bare float (the mean) per kind, or `None` for a kind with no episodes.
    A median cannot be recovered from a mean, so those fields stay `None`
    instead of being quietly filled in with the mean -- the point of this
    whole exercise is not to let an old number pass for a current one."""
    if isinstance(value, dict):
        return {key: value.get(key) for key in ("median", "mean", "share_positive", "n")}
    if value is None:
        return {"median": None, "mean": None, "share_positive": None, "n": None}
    return {"median": None, "mean": float(value), "share_positive": None, "n": None}


def episodes_detail(results: dict, episodes: list) -> dict:
    """Every method's per-episode rows, each tagged with the volume the
    episode ran on: `{method: [{"attainment", "kind", "volume"}, ...]}`.

    `summary` and `comparisons` are aggregates, and an aggregate cannot be
    re-aggregated a different way. The v2-vs-v3 comparison in the write-up
    needs the median of the *seed-averaged per-episode* attainment, which is
    not recoverable from a stored median -- so it had to be recomputed by
    re-running the policy, and for a while was quoted from a session that no
    longer existed. Writing the rows themselves means any later question
    ("per patient?", "lungs only?", "paired against which arm?") is answered
    from the file rather than from another evaluation run.

    Raises if a method's rows do not line up with `episodes`: the pairing is
    positional, so a length mismatch means the rows describe some other run
    and every per-episode statistic drawn from them would be silently
    misaligned."""
    detail = {}
    for name, rows in results.items():
        if len(rows) != len(episodes):
            raise ValueError(
                f"{name}: {len(rows)} rows for {len(episodes)} episodes -- "
                "per-episode rows are paired by position and must match")
        detail[name] = [{"attainment": row["attainment"], "kind": row["kind"],
                         "volume": episode["volume"]}
                        for row, episode in zip(rows, episodes)]
    return detail


def per_class_summary(results: dict, episodes: list, model_for_volume=None,
                      active_layers=None) -> dict:
    """Report class support/reachability for every episode and scored classes.

    Every canonical class gets one status count per episode. Attainment is
    attributed only to classes explicitly mentioned by the episode instruction;
    unmentioned reachable classes remain visible in coverage counts but do not
    dilute class performance metrics.
    """
    get_model = model_for_volume or visibility.for_volume
    models = {}
    report = {}
    for method, rows in results.items():
        if len(rows) != len(episodes):
            raise ValueError(
                f"{method}: {len(rows)} rows for {len(episodes)} episodes -- "
                "per-class rows must align with episodes")
        grouped = collections.defaultdict(list)
        counts = collections.defaultdict(collections.Counter)
        for row, episode in zip(rows, episodes):
            volume = episode["volume"]
            if volume not in models:
                try:
                    models[volume] = get_model(volume)
                except (KeyError, OSError, ValueError):
                    models[volume] = None
            model = models[volume]
            if model is None or "instruction" not in episode:
                supported = set()
                reachable = set()
                mentioned = set()
            else:
                supported = set(goals.goal_classes_for_volume(volume))
                reachable = set(goals.reachable_goal_classes(volume, model))
                mentioned = set(episode["instruction"]["targets"])
            for goal_class in goals.GOAL_CLASSES:
                status = ("reachable" if goal_class in reachable else
                          "unreachable" if goal_class in supported else "unsupported")
                counts[goal_class][status] += 1
                if goal_class in mentioned and status == "reachable" and row["attainment"] is not None:
                    grouped[goal_class].append(float(row["attainment"]))
        report[method] = {}
        for goal_class in goals.GOAL_CLASSES:
            values = grouped[goal_class]
            report[method][goal_class] = {
                **goals.summarise_attainment(values),
                "supported": counts[goal_class]["reachable"] + counts[goal_class]["unreachable"],
                "reachable": counts[goal_class]["reachable"],
                "unsupported": counts[goal_class]["unsupported"],
                "unreachable": counts[goal_class]["unreachable"],
            }
    return report


def compare(results: dict, policy_name: str = "policy", episodes=None,
            model_for_volume=None, active_layers=None) -> dict:
    """Robust attainment stats per method (overall and per instruction kind),
    plus a paired Wilcoxon signed-rank test of `results[policy_name]` against
    every other method in `results`. Episodes where either side's
    `attainment` is `None` (a failed baseline, or a start state that already
    meets the goal) are dropped from that pair's test and from that method's
    statistics."""
    summary = {name: _summarize(rows) for name, rows in results.items()}

    policy_rows = results[policy_name]
    comparisons = {}
    for name, rows in results.items():
        if name == policy_name:
            continue
        pairs = [(p["attainment"], b["attainment"]) for p, b in zip(policy_rows, rows)
                 if p["attainment"] is not None and b["attainment"] is not None]
        if pairs:
            xs, ys = zip(*pairs)
            comparisons[name] = wilcoxon(xs, ys)
        else:
            comparisons[name] = {"statistic": None, "p_value": None, "n": 0}

    comparison = {"summary": summary, "comparisons": comparisons,
                  "metadata": observation_metadata()}
    if episodes is not None:
        comparison["per_class"] = per_class_summary(
            results, episodes, model_for_volume, active_layers)
    return comparison


# --- CLI -----------------------------------------------------------------------

def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_count(value) -> str:
    return "n/a" if value is None else f"{value:d}"


def check_provenance(result: dict, active_layers=None) -> dict:
    """Compare a result's recorded `provenance` block against the code
    running now (see `provenance.compare`). A result with no block at all is
    reported stale -- an untraceable file is the exact thing that cost a
    week."""
    return provenance.compare(result.get("provenance"), expected_layers=active_layers)


def _print_table(comparison: dict, active_layers=None) -> None:
    report = check_provenance(comparison, active_layers=active_layers)
    banner = provenance.format_staleness(report)
    if banner:
        print(banner)
        print()

    summary = comparison["summary"]
    print(f"{'method':22s} {'median':>8s} {'mean_clip':>10s} {'mean_raw':>10s} {'share+':>8s} {'n':>6s}")
    for name in sorted(summary):
        entry = summary[name]
        print(f"{name:22s} {_fmt(entry['median']):>8s} {_fmt(entry['mean_clipped']):>10s} "
              f"{_fmt(entry['mean_raw']):>10s} {_fmt(entry['share_positive']):>8s} {entry['n']:6d}")

    # Per kind, median first: the mean is kept beside it (and is all an older
    # result file has), but it is the number that misreads a heavy negative
    # tail as a broken instruction kind.
    print()
    print(f"{'method':22s} {'kind':12s} {'median':>8s} {'mean':>10s} {'share+':>8s} {'n':>6s}")
    for name in sorted(summary):
        for kind, raw in summary[name].get("by_kind", {}).items():
            stats = kind_stats(raw)
            print(f"{name:22s} {kind:12s} {_fmt(stats['median']):>8s} {_fmt(stats['mean']):>10s} "
                  f"{_fmt(stats['share_positive']):>8s} {_fmt_count(stats['n']):>6s}")

    print()
    print(f"{'baseline':22s} {'p-value':>10s} {'n pairs':>8s}")
    for name in sorted(comparison["comparisons"]):
        entry = comparison["comparisons"][name]
        p_value = "n/a" if entry["p_value"] is None else f"{entry['p_value']:.4f}"
        print(f"{name:22s} {p_value:>10s} {entry['n']:8d}")

    if banner:
        print()
        print("!!! the numbers above are STALE -- see the warning at the top of this table")


def load_result(path: str) -> dict:
    """A result file as written by `main`, for re-printing it later. The
    printing path checks its provenance, so re-reading an old file is how the
    staleness warning reaches a human who did not run the job."""
    with open(path) as stream:
        return json.load(stream)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", help="path to an SB3 checkpoint (e.g. best.zip)")
    parser.add_argument("--show", type=str, default=None,
                         help="re-print an existing result file instead of evaluating, "
                              "checking its provenance against the current code")
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "out_of_source"))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--formulation", default=DEFAULT_FORMULATION, choices=("one_shot", "multi_step"),
                         help="which policy formulation to evaluate (default: %(default)s)")
    parser.add_argument("--refine", type=int, default=0,
                        help="visibility-evaluation budget to refine the one-shot policy's proposal "
                             "with a hill_climb search (0 = off, default: %(default)s); adds a "
                             "policy_plus_refine row to the comparison")
    layers = parser.add_mutually_exclusive_group()
    layers.add_argument("--layers", help="active anatomy layers as inline JSON")
    layers.add_argument("--layers-file", help="JSON file containing active anatomy layers")
    args = parser.parse_args(argv)
    if args.show is None and not args.policy:
        parser.error("--policy is required unless --show is given")
    return args


def main(argv=None, active_layers=None):
    args = parse_args(argv)
    if active_layers is None:
        active_layers = provenance.load_active_layers(args.layers, args.layers_file)
    if args.show:
        _print_table(load_result(args.show), active_layers=active_layers)
        return

    episodes = fixed_episodes(args.split, args.episodes, seed=args.seed, formulation=args.formulation)

    results = {"policy": run_policy(args.policy, episodes, formulation=args.formulation,
                                      active_layers=active_layers)}
    if args.refine > 0:
        results["policy_plus_refine"] = run_policy_with_refinement(
            args.policy, episodes, evaluations=args.refine, active_layers=active_layers)
    for name in BASELINES:
        results[name] = run_baseline(name, episodes, active_layers=active_layers)

    # The import-time record, not a fresh one: this job scored with the code
    # it loaded when it started, which may be hours old by now. Printing the
    # table with that record in place is also what catches a scoring fix that
    # landed mid-run -- the fingerprint no longer matches the file on disk,
    # and the run says so before anyone quotes it.
    result = {"policy": args.policy, "split": args.split, "episodes": args.episodes,
              "seed": args.seed, "formulation": args.formulation, "refine": args.refine,
              "provenance": provenance.result_provenance(active_layers),
              "metadata": observation_metadata(),
              "episodes_detail": episodes_detail(results, episodes),
               **compare(results, episodes=episodes, active_layers=active_layers)}
    _print_table(result, active_layers=active_layers)

    out = args.out or DEFAULT_OUT_TEMPLATE.format(split=args.split)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"\n[vis_eval] wrote {out}")


if __name__ == "__main__":
    main()
