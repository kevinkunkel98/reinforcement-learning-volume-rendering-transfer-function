# Online Learning for the Transfer-Function RL Policy — Design

## Purpose

`rl/train.py` trains the TF-opacity SAC agent entirely offline, once, in a
single 200k-step batch across 4 parallel `TFEnv` instances, then `rl/eval.py`
checks it once at the end against a held-out episode set. `rl/serve.py` /
`server.py`'s `policy` evaluator then serve that frozen checkpoint live —
inference only, no further learning.

This adds a **standalone online-learning script**: a fresh SAC agent trains
on a single `TFEnv`, with training periodically interrupted to re-evaluate
against the same held-out episodes `rl/eval.py` uses, so the task metric's
improvement over training is directly observable — logged to CSV and
plotted, not just inferred from SB3's own reward curve. This is phase 1 of a
two-phase plan; phase 2 (wiring the same learning step into `server.py`'s
live `policy` path) is explicitly deferred until this phase shows the
mechanics work.

Motivation: the thesis's underlying question is whether RL can train a
transfer function at all — the specific algorithm/environment is expected to
change. Online learning is one more concrete data point toward that
question: does continual adaptation buy anything the frozen offline policy
doesn't, on this same task.

## Grounding in existing code

- `rl/train.py`: `model.set_logger(configure_logger(run_dir, ["stdout",
  "csv", "tensorboard"]))` called **before** `model.learn()` — verified
  against the installed `stable-baselines3` (2.9.0) in
  `docs/superpowers/specs/2026-09-05-training-curve-plots-design.md`. Reused
  unchanged: same call, same three format strings.
- `rl/eval.py`:
  - `_build_episodes(n_episodes, seed)` builds a fixed held-out set from
    `TFEnv(seed=seed).reset()` (default 20 episodes, `EVAL_SEED = 12345`).
  - `_run_policy(model, episode)` replays `MAX_STEPS` (20) deterministic
    actions from a given model against one episode, returning the list of
    `mass_fraction` values after each step.
  - `_run_hill_climb(episode)` replays the hill-climb baseline the same way
    — deterministic given the episode's start params/tissue/direction, so it
    only needs to run once per episode, not once per checkpoint.
  - `_steps_to_90pct(fractions, start, best)` — generic, reused as-is.
  - `evaluate()`'s per-episode `best = max(combined) if direction ==
    "increase" else min(combined)` where `combined = pf + hf` — reused so
    the online run's per-checkpoint numbers are computed the identical way
    the documented offline table (`mean final mass_fraction: 0.344` policy /
    `0.347` hill-climb) already was, making them directly comparable.
- `plots/style.py` (`apply_style`, `NAVY`/`BLUE`/`SAGE`/`AMBER`) and
  `plots/read_progress.py` (`load_progress`, generic CSV→`numpy`-by-column,
  no pandas) already exist and are format-agnostic — both reused unchanged.
  `plots/training_curves.py` already turns any `run_seed*`-shaped
  `progress.csv` into figures; since the online script attaches the same
  SB3 logger, that script works against the online run's directory
  unmodified (just pass a different `--run-dir`).
- Nothing currently plots the held-out-eval metric over training progress —
  that curve doesn't exist for either the offline or online runs. New.

## New module: `rl/online_train.py`

```python
MODEL_PATH = "out/rl_models/sac_tf_agent_online.zip"
LOG_DIR = "out/rl_logs"           # same root as rl/train.py
EVAL_N_EPISODES = 20
EVAL_SEED = 12345                  # matches rl.eval's defaults exactly
```

`online_train(total_timesteps=50_000, eval_interval=5_000, seed=0,
model_path=MODEL_PATH) -> SAC`:

1. `run_dir = out/rl_logs/online_run_seed{seed}`; `env = TFEnv(seed=seed)`
   (single env — continuous online interaction, not `make_vec_env`).
2. `model = SAC("MlpPolicy", env, seed=seed)`; attach the logger exactly as
   `rl/train.py` does, pointed at `run_dir` — this alone produces
   `run_dir/progress.csv` + tensorboard events as a side effect, reusable by
   `plots/training_curves.py`.
3. Build the held-out episodes once: `episodes = rl.eval._build_episodes(
   EVAL_N_EPISODES, EVAL_SEED)`. Precompute each episode's hill-climb
   trajectory once via `rl.eval._run_hill_climb`, so per-checkpoint eval
   only has to replay the *policy* side.
4. Open `run_dir/eval_progress.csv`, write header
   `timesteps,mean_final_mass_fraction,mean_steps_to_90pct`.
5. Loop for `ceil(total_timesteps / eval_interval)` chunks:
   - `model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)`
     — SB3 handles the actual online interaction/replay-buffer/gradient
     mechanics; no hand-rolled `replay_buffer.add`/`model.train()` calls.
   - For each held-out episode: `pf = rl.eval._run_policy(model, ep)`,
     combine with that episode's precomputed `hf` for `best` exactly as
     `evaluate()` does, collect `pf[-1]` and `_steps_to_90pct(pf, pf[0],
     best)`.
   - Append one row (cumulative timesteps so far, mean final
     `mass_fraction` across the 20 episodes, mean steps-to-90%) to
     `eval_progress.csv`, and print the same row to stdout.
6. `model.save(model_path)`; return `model`.

`main()`: `argparse` mirroring `rl/train.py`'s flags —
`--timesteps` (default `50_000`), `--eval-interval` (default `5_000`,
giving 10 checkpoints), `--seed` (default `0`), `--out` (default
`MODEL_PATH`).

## New plotting: `plots/online_eval_curve.py`

- `_find_latest_run(log_dir)`: same pattern as
  `plots/training_curves._find_latest_run`, but globs `online_run_seed*`
  instead of `run_seed*`.
- `plot_online_eval_curve(run_dir, output_dir=OUTPUT_DIR) -> str`:
  - `cols = load_progress(os.path.join(run_dir, "eval_progress.csv"))` —
    reused unchanged; the loader is already generic over column names.
  - One figure: left axis = mean final `mass_fraction` vs. timesteps
    (`NAVY`), right twin axis = mean steps-to-90% vs. timesteps (`AMBER`),
    styled via `apply_style()`.
  - Two dashed horizontal reference lines on the left axis at `0.344`
    (offline SAC policy) and `0.347` (hill-climb baseline) — the documented
    offline numbers from `rl/eval.py`'s results table
    (`docs/architecture.typ` §RL sub-project 1), hardcoded as named
    constants with a comment citing that source, so the online curve's
    progress toward/away from both baselines is visible at a glance without
    a runtime dependency on the offline results file existing.
  - Saves `plots/output/online_eval_curve.pdf` + `.png`, same as
    `training_curves.py`'s save pattern.
- `main()`: `--run-dir` (default: latest `online_run_seed*` under
  `out/rl_logs`), `--out` (default `plots/output`).

## Data flow

```
python -m rl.online_train
  → out/rl_models/sac_tf_agent_online.zip
  → out/rl_logs/online_run_seed{seed}/progress.csv       (SB3, via reused logger attach)
  → out/rl_logs/online_run_seed{seed}/eval_progress.csv  (new, held-out task metric)

python -m plots.training_curves --run-dir out/rl_logs/online_run_seed{seed}
  → plots/output/training_curves.{pdf,png}                (reused unmodified)

python -m plots.online_eval_curve --run-dir out/rl_logs/online_run_seed{seed}
  → plots/output/online_eval_curve.{pdf,png}               (new)
```

## Error handling

- `online_train`: no special-cased error handling beyond what SB3/`TFEnv`
  already raise — matches `rl/train.py`'s existing style.
- `online_eval_curve.py`: `FileNotFoundError` with an actionable message if
  no `online_run_seed*` directory exists yet (mirrors the existing
  `_find_latest_run` message style), and a distinct message if
  `eval_progress.csv` is missing under a given `--run-dir` (catches passing
  an offline `run_seed*` directory by mistake).

## Testing

- `tests/test_rl_online_train.py` — `@pytest.mark.slow`, following
  `tests/test_rl_train_logging.py`'s exact pattern: `monkeypatch.chdir(
  tmp_path)`, run `online_train(total_timesteps=200, eval_interval=100,
  seed=42, model_path="out/rl_models/test_online_agent.zip")`, assert the
  model file exists, assert `eval_progress.csv` exists with the correct
  header and exactly 2 data rows, assert `progress.csv` also exists (new
  call site for the logger-attach pattern, worth reconfirming here).
- `tests/test_plots_online_eval_curve.py` — following
  `tests/test_plots_training_curves.py`'s pattern: write a small synthetic
  `eval_progress.csv` (≥3 rows) into a temp `online_run_seed0` directory,
  call `plot_online_eval_curve()` directly, assert both output files exist
  and are non-empty.

## New dependencies

None — reuses `matplotlib`, `numpy`, `stable-baselines3`, `gymnasium`, all
already installed.

## Explicitly out of scope (this delivery)

- **Live chat-UI wiring** (`server.py`'s `policy` evaluator path) — phase 2,
  deferred until this phase demonstrates the online-learning mechanics work.
- **Warm-starting** from the existing offline checkpoint — fresh SAC agent
  each run, per explicit choice (this phase tests the online mechanism's
  ability to learn from zero).
- **Human-judgment reward** — automatic `mass_fraction` reward only,
  unchanged from `TFEnv`'s existing offline reward.
- **Overwriting** `out/rl_models/sac_tf_agent.zip` — the offline baseline
  stays untouched; the online run writes to a separate file.
- **Camera-viewpoint online learning** (`rl/camera_env.py`) — TF-opacity
  only.
