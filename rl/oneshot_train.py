"""SAC training CLI for `OneShotEnv`, across the TotalSegmentator training
volumes, with periodic evaluation on the validation volumes.

    .venv/bin/python -m rl.oneshot_train --timesteps 150000 --seed 0 --out out/rl_v2/oneshot_seed0

Every `--eval-interval` steps: run a fixed, seeded set of validation episodes
with the deterministic policy, append attainment stats (mean, median, mean
clipped to [-1, 1], share of episodes that improved on doing nothing --
overall and per instruction kind) to `eval_progress.csv` in the run
directory, save a checkpoint, and keep the best-scoring checkpoint as
`best.zip`, selected by median attainment (robust to the single-episode
outliers a mean is vulnerable to -- see `goals.summarise_attainment`). SB3's
own CSV logger writes `progress.csv` to the same directory. This mirrors
`rl/vis_train.py` (same CSV fields, best-by-median selection, and run-
directory guard) so the ten-step and one-shot runs are directly comparable;
`learning_starts=500` (rather than SB3's default 100) gives SAC a few more
random one-step episodes before it starts learning from a mostly-empty
replay buffer.
"""
import argparse
import csv
import json
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

import datasets
import goals
from rl.oneshot_env import OneShotEnv, observation_metadata

VALIDATION_EPISODES = 40
VALIDATION_SEED_BASE = 10_000   # fixed offset so validation seeds never collide with training
EVAL_KINDS = tuple(kind for kind, _ in goals.INSTRUCTION_MIX)
CSV_FIELDS = (["timesteps", "mean_attainment", "median_attainment", "mean_clipped_attainment", "share_positive"]
              + [f"attainment_{kind}" for kind in EVAL_KINDS])
DEFAULT_OUT_TEMPLATE = "out/rl_v2/oneshot_seed{seed}"
CHECKPOINT_NAME = "checkpoint_{timesteps}.zip"
BEST_NAME = "best.zip"
EVAL_PROGRESS_NAME = "eval_progress.csv"
METADATA_NAME = "metadata.json"
LEARNING_STARTS = 500
SHORT_EXPERIMENT = {"timesteps": 1_000, "eval_interval": 500, "eval_episode_count": 8}


def build_env(volume_ids=None, model_for_volume=None, **env_kwargs) -> OneShotEnv:
    """The training environment: `OneShotEnv` over the training split, or
    `volume_ids` when given (tests inject a stub volume list)."""
    ids = volume_ids if volume_ids is not None else datasets.volumes_for_split("train")
    kwargs = {} if model_for_volume is None else {"model_for_volume": model_for_volume}
    return OneShotEnv(ids, **kwargs, **env_kwargs)


def build_eval_env(volume_ids=None, model_for_volume=None, **env_kwargs) -> OneShotEnv:
    """The validation environment: `OneShotEnv` over the validation split, or
    `volume_ids` when given."""
    ids = volume_ids if volume_ids is not None else datasets.volumes_for_split("val")
    kwargs = {} if model_for_volume is None else {"model_for_volume": model_for_volume}
    return OneShotEnv(ids, **kwargs, **env_kwargs)


def validation_episodes(env, count: int = VALIDATION_EPISODES, seed_base: int = VALIDATION_SEED_BASE) -> list:
    """A fixed, seeded list of `(env, seed)` pairs: the same `count` episodes
    every time evaluation runs, so validation attainment across eval
    intervals is comparable."""
    return [(env, seed_base + i) for i in range(count)]


def evaluate_policy_on(episodes, model) -> list:
    """Run each `(env, seed)` episode to completion (one step, for
    `OneShotEnv`) with the policy's deterministic action; return one
    `{"attainment", "kind"}` dict per episode, from the final step's info."""
    results = []
    for env, seed in episodes:
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        results.append({"attainment": info["attainment"], "kind": info["kind"]})
    return results


def summarize_eval(results: list) -> dict:
    """Robust attainment stats overall (`goals.summarise_attainment`: mean,
    median, mean clipped to [-1, 1], share of episodes that improved on
    doing nothing) and a plain mean per instruction kind; `None` for a kind
    with no episodes in `results` (rather than crashing on an empty mean)."""
    attainments = [row["attainment"] for row in results]
    stats = goals.summarise_attainment(attainments)
    by_kind = {}
    for kind in EVAL_KINDS:
        values = [row["attainment"] for row in results if row["kind"] == kind]
        by_kind[kind] = float(np.mean(values)) if values else None
    return {"mean_attainment": stats["mean_raw"], "median_attainment": stats["median"],
            "mean_clipped_attainment": stats["mean_clipped"], "share_positive": stats["share_positive"],
            "by_kind": by_kind}


def ensure_run_dir(out: str) -> None:
    """Create the run directory, refusing to silently overwrite one that
    already exists (and so already has checkpoints/logs in it)."""
    if os.path.exists(out):
        raise FileExistsError(
            f"run directory already exists: {out!r} -- choose a new --out or remove it first")
    os.makedirs(out)


def _eval_row(timesteps: int, summary: dict) -> dict:
    row = {"timesteps": timesteps, "mean_attainment": summary["mean_attainment"],
           "median_attainment": summary["median_attainment"],
           "mean_clipped_attainment": summary["mean_clipped_attainment"],
           "share_positive": summary["share_positive"]}
    for kind in EVAL_KINDS:
        row[f"attainment_{kind}"] = summary["by_kind"][kind]
    return row


def run_training(out: str, timesteps: int, seed: int, eval_interval: int,
                  train_env=None, eval_env=None, eval_episode_count: int = VALIDATION_EPISODES,
                  env_kwargs=None) -> dict:
    """Train SAC on `train_env` (default: the training split), evaluating on
    `eval_env` (default: the validation split) every `eval_interval` steps.
    Returns the run directory's paths and the logged rows, for tests and the
    CLI alike."""
    ensure_run_dir(out)
    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
    train_env = train_env if train_env is not None else build_env(**env_kwargs)
    eval_env = eval_env if eval_env is not None else build_eval_env(**env_kwargs)

    model = SAC("MlpPolicy", train_env, seed=seed, verbose=1, learning_starts=LEARNING_STARTS)
    model.set_logger(configure(out, ["stdout", "csv"]))

    episodes = validation_episodes(eval_env, count=eval_episode_count)
    eval_path = os.path.join(out, EVAL_PROGRESS_NAME)
    best_path = os.path.join(out, BEST_NAME)
    metadata = {**observation_metadata(train_env.policy_version, train_env.action_mode,
                                        train_env.reward_mode), "seed": seed, "timesteps": timesteps}
    with open(os.path.join(out, METADATA_NAME), "w") as stream:
        json.dump(metadata, stream, indent=2)

    rows = []
    best_attainment = None
    done = 0
    with open(eval_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        while done < timesteps:
            chunk = min(eval_interval, timesteps - done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=(done == 0))
            done += chunk

            results = evaluate_policy_on(episodes, model)
            summary = summarize_eval(results)
            row = _eval_row(done, summary)
            writer.writerow(row)
            stream.flush()
            rows.append(row)

            model.save(os.path.join(out, CHECKPOINT_NAME.format(timesteps=done)))
            # Select by median, not mean: attainment is unbounded below, and
            # a mean lets a single bad episode swing which checkpoint looks
            # best (see goals.summarise_attainment).
            median_attainment = summary["median_attainment"]
            if median_attainment is not None and (best_attainment is None or median_attainment > best_attainment):
                best_attainment = median_attainment
                model.save(best_path)

    return {"out": out, "eval_progress_path": eval_path, "best_path": best_path,
            "metadata": metadata, "rows": rows}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--policy-version", choices=("oneshot-v6", "oneshot-v7"), default="oneshot-v6")
    parser.add_argument("--action-mode", choices=("absolute", "residual"), default=None)
    parser.add_argument("--hindsight-ratio", type=float, default=0.0)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--reward-mode", choices=("attainment", "target"), default="attainment")
    parser.add_argument("--short", action="store_true", help="use the short smoke experiment preset")
    args = parser.parse_args(argv)
    if args.policy_version == "oneshot-v7" and args.action_mode is None:
        parser.error("--action-mode is required for oneshot-v7")
    if args.policy_version == "oneshot-v7" and args.action_mode != "residual":
        parser.error("oneshot-v7 requires --action-mode residual")
    if args.policy_version == "oneshot-v7" and args.reward_mode != "target":
        parser.error("oneshot-v7 requires --reward-mode target")
    if args.policy_version == "oneshot-v6" and args.action_mode == "residual":
        parser.error("oneshot-v6 requires --action-mode absolute")
    if args.policy_version == "oneshot-v6" and args.reward_mode == "target":
        parser.error("oneshot-v6 requires --reward-mode attainment")
    args.action_mode = args.action_mode or "absolute"
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out or DEFAULT_OUT_TEMPLATE.format(seed=args.seed)
    timesteps = SHORT_EXPERIMENT["timesteps"] if args.short else args.timesteps
    eval_interval = SHORT_EXPERIMENT["eval_interval"] if args.short else args.eval_interval
    result = run_training(out=out, timesteps=timesteps, seed=args.seed, eval_interval=eval_interval,
                          eval_episode_count=(SHORT_EXPERIMENT["eval_episode_count"]
                                               if args.short else VALIDATION_EPISODES),
                          env_kwargs={"policy_version": args.policy_version,
                                      "action_mode": args.action_mode,
                                      "hindsight_ratio": args.hindsight_ratio,
                                      "balance_classes": args.balance_classes,
                                      "reward_mode": args.reward_mode})
    print(f"\n[oneshot_train] wrote {result['eval_progress_path']} and {result['best_path']}")


if __name__ == "__main__":
    main()
