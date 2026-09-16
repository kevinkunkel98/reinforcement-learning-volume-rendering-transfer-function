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
`goals.summarise_attainment`) overall, a plain mean per instruction kind, and
a paired Wilcoxon signed-rank test of the policy against each baseline (on
the raw, unclipped paired values), computed directly (`scipy` is not
installed in this environment).
"""
import argparse
import json
import math
import os

import numpy as np

import datasets
import goals
import visibility
from rl.baselines import BASELINES
from rl.oneshot_env import OneShotEnv
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


def _frozen_episode_env(volume: str, start_params: np.ndarray, instruction: dict,
                         model_for_volume=visibility.for_volume):
    """A `VisibilityTFEnv` loaded directly into one episode's exact state,
    instead of relying on `reset(seed=...)` reproducing it -- exact and
    independent of the length/order of whatever volume list the episode was
    originally generated from."""
    env = VisibilityTFEnv([volume], model_for_volume=model_for_volume)
    model = model_for_volume(volume)
    raw_features = model.features(start_params)
    start_agg = goals.aggregate(raw_features)
    start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

    env._volume = volume
    env._model = model
    env._params = np.asarray(start_params, dtype=np.float64).copy()
    env._instruction = instruction
    env._start_agg = start_agg
    env._start_distance = start_distance
    env._prev_distance = start_distance
    env._step_count = 0

    obs = env._build_observation(env._params, start_agg)
    info = env._info(start_distance, start_agg, goals.is_useless(raw_features))
    return env, obs, info


def _frozen_one_shot_env(volume: str, start_params: np.ndarray, instruction: dict,
                          model_for_volume=visibility.for_volume):
    """A `OneShotEnv` loaded directly into one episode's exact state, the
    `OneShotEnv` counterpart of `_frozen_episode_env` above."""
    env = OneShotEnv([volume], model_for_volume=model_for_volume)
    model = model_for_volume(volume)
    raw_features = model.features(start_params)
    start_agg = goals.aggregate(raw_features)
    start_distance = goals.distance(instruction["goal"], start_agg, start_agg)

    env._volume = volume
    env._model = model
    env._start_params = np.asarray(start_params, dtype=np.float64).copy()
    env._params = env._start_params
    env._instruction = instruction
    env._start_agg = start_agg
    env._start_distance = start_distance

    obs = env._build_observation()
    info = env._info(env._attainment(start_agg), goals.is_useless(raw_features))
    return env, obs, info


def run_policy(model_path: str, episodes: list, model_for_volume=None, load_model=None,
               formulation: str = "multi_step") -> list:
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
                episode["volume"], episode["start_params"], episode["instruction"], **kwargs)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            env, obs, info = _frozen_episode_env(
                episode["volume"], episode["start_params"], episode["instruction"], **kwargs)
            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
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


def run_baseline(name: str, episodes: list, model_for_volume=None) -> list:
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
            start_agg = goals.aggregate(model.features(episode["start_params"]))
            final_params = baseline_fn(model, episode["start_params"], instruction)
            final_agg = goals.aggregate(model.features(final_params))
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
    # breakdown stays a plain mean: those buckets are small, and they exist
    # to spot which instruction kinds the method struggles with, not to
    # stand alone as a headline number.
    attainments = [row["attainment"] for row in rows]
    stats = goals.summarise_attainment(attainments)
    by_kind = {}
    for kind in EVAL_KINDS:
        values = [row["attainment"] for row in rows if row["kind"] == kind and row["attainment"] is not None]
        by_kind[kind] = float(np.mean(values)) if values else None
    return {**stats, "by_kind": by_kind}


def compare(results: dict, policy_name: str = "policy") -> dict:
    """Mean attainment per method (overall and per instruction kind), plus a
    paired Wilcoxon signed-rank test of `results[policy_name]` against every
    other method in `results`. Episodes where either side's `attainment` is
    `None` (a failed baseline, or a start state that already meets the goal)
    are dropped from that pair's test and from that method's means."""
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

    return {"summary": summary, "comparisons": comparisons}


# --- CLI -----------------------------------------------------------------------

def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_table(comparison: dict) -> None:
    summary = comparison["summary"]
    print(f"{'method':22s} {'median':>8s} {'mean_clip':>10s} {'mean_raw':>10s} {'share+':>8s} {'n':>6s}")
    for name in sorted(summary):
        entry = summary[name]
        print(f"{name:22s} {_fmt(entry['median']):>8s} {_fmt(entry['mean_clipped']):>10s} "
              f"{_fmt(entry['mean_raw']):>10s} {_fmt(entry['share_positive']):>8s} {entry['n']:6d}")

    print()
    print(f"{'baseline':22s} {'p-value':>10s} {'n pairs':>8s}")
    for name in sorted(comparison["comparisons"]):
        entry = comparison["comparisons"][name]
        p_value = "n/a" if entry["p_value"] is None else f"{entry['p_value']:.4f}"
        print(f"{name:22s} {p_value:>10s} {entry['n']:8d}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="path to an SB3 checkpoint (e.g. best.zip)")
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "out_of_source"))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--formulation", default=DEFAULT_FORMULATION, choices=("one_shot", "multi_step"),
                         help="which policy formulation to evaluate (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    episodes = fixed_episodes(args.split, args.episodes, seed=args.seed, formulation=args.formulation)

    results = {"policy": run_policy(args.policy, episodes, formulation=args.formulation)}
    for name in BASELINES:
        results[name] = run_baseline(name, episodes)

    comparison = compare(results)
    _print_table(comparison)

    out = args.out or DEFAULT_OUT_TEMPLATE.format(split=args.split)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as stream:
        json.dump({"policy": args.policy, "split": args.split, "episodes": args.episodes,
                    "seed": args.seed, "formulation": args.formulation, **comparison}, stream, indent=2)
    print(f"\n[vis_eval] wrote {out}")


if __name__ == "__main__":
    main()
