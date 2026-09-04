# RL Implementation Design

## Purpose

The MVP baseline (rule/LLM command parsing, hill-climbing search, `opacity_mass`
objective evaluator) is complete: 69 tests passing, real CT datasets working,
a polished chat UI. This spec covers the next phase: replacing hill-climbing's
exploration strategy with a learned continuous-control policy (SAC), trained
against the existing `opacity_mass` reward with no human data required to
start.

This intentionally supersedes the MVP's original "no torch" constraint — that
rule was scoped to the baseline phase only. The baseline code (`render.py`,
`evaluate.py`, `search.py`, `commands.py`) stays untouched and dependency-free;
new dependencies live only in the new `rl/` package.

## Grounding in existing code

- `transfer.py`: TF vector has `N_PEAKS = 4` Gaussian peaks (6 floats each:
  center, width, height, r, g, b), but there are 5 tissue targets (air, fat,
  soft, spongy, bone) — air has no dedicated peak by default.
- `commands.py::_find_or_create_peak(params, tissue)`: resolves which peak
  index currently represents a tissue, reseeding the weakest peak if none is
  within `NEAR_THRESHOLD_HU`. This is how the rule parser and hill-climbing
  already handle the 5-targets/4-peaks mismatch. The RL env reuses this
  exact function rather than inventing new peak-assignment logic.
- `evaluate.py::objective()`: for increase/decrease, reward signal today is a
  before/after delta in `opacity_mass` for the target band, computed directly
  on `params` — **no rendering involved**. This confirms training doesn't
  need VTK rendering either: the reward can be computed directly from the TF
  vector against the phantom's HU space, the same way hill-climbing already
  evaluates candidate steps.
- `search.py::propose_step(params, peak_idx, sign, step)`: mutates one peak's
  height by `sign * step` in internal `[0, 1]` unit space, clamped. The RL
  action reuses this directly (`sign=1.0, step=action_value * MAX_DELTA`) —
  no new step-application math needed.
- `transfer.py::opacity_mass(params, hu_lo, hu_hi)`: integrates opacity over
  an HU sub-range via `np.trapezoid`. Raw values scale with band width (e.g.
  bone's band is 1400 HU wide), so a normalized version
  (`opacity_mass / (hi - lo)`, an average-opacity-in-band fraction, roughly
  `[0, 1]`) is used for observations and reward instead of the raw integral.

## Architecture

New `rl/` package, isolated from the baseline:

- `rl/env.py` — `TFEnv(gymnasium.Env)`, the training/eval environment.
- `rl/train.py` — builds a vectorized `TFEnv`, trains stable-baselines3's
  SAC, saves the model and logs.
- `rl/eval.py` — runs the trained policy and the existing hill-climbing
  baseline over the same held-out episodes, reports comparison metrics.

`render.py`, `evaluate.py`, `search.py`, `commands.py`, `transfer.py` are
**not modified** except where noted (the normalized mass-fraction helper is
added to `transfer.py` since it's a natural sibling of `opacity_mass`, and is
useful outside the RL package too — e.g. future overlay/telemetry work).

## Environment specification (`rl/env.py`)

**Observation** — `Box(shape=(31,))`:
1. `[0:24]` — the TF vector, already in its native `[-1, 1]` range.
2. `[24:29]` — one-hot target tissue (`air, fat, soft, spongy, bone` in that
   fixed order).
3. `[29]` — direction: `+1.0` for increase, `-1.0` for decrease.
4. `[30]` — current `mass_fraction` in the target band, `[0, 1]`.

**Action** — `Box(shape=(1,), low=-1.0, high=1.0)`. Interpreted as a signed
delta applied to the *resolved* target peak's height:

```python
proposed = propose_step(params, peak_idx, sign=1.0, step=float(action[0]) * MAX_DELTA)
```

`MAX_DELTA = 0.2` (one action can move a peak's height by at most 20% of its
full range per step — matches hill-climbing's typical step scale).

**Peak resolution**: `peak_idx` is resolved once per episode via
`_find_or_create_peak(params, target_tissue)` at `reset()`, and held fixed
for the rest of the episode. This keeps the action's meaning stable — the
agent isn't punished for a peak reassignment happening mid-episode from
under it.

**Reward** — dense, per step:

```python
def mass_fraction(params, tissue):
    lo, hi = TISSUE_BANDS[tissue]
    return opacity_mass(params, lo, hi) / (hi - lo)

reward = direction_sign * (mass_fraction(after, target) - mass_fraction(before, target))
```

**Episode**:
- `reset()`: sample a random starting TF vector (perturb `default_params()`
  by adding uniform noise in `[-0.3, 0.3]` to each peak's height, clipped to
  valid range — centers/widths/colors stay at defaults, since only height is
  ever actuated), and a random goal: `target_tissue` uniform over the 5
  tissues, `direction` uniform over `{increase, decrease}`.
- Fixed length: `MAX_STEPS = 20` steps per episode, terminated via
  Gymnasium's `truncated=True` at the limit (`terminated` is always `False`
  — no early-success condition, since the dense reward already drives fast
  convergence and a hard success threshold would need its own tuning).
- `step()` returns `(obs, reward, terminated=False, truncated, info)` where
  `info` includes `mass_fraction_after` for logging/eval.

**No VTK/rendering dependency in `rl/env.py`** — it operates purely on the
`params` vector and the phantom's HU array via `opacity_mass`, matching how
`evaluate.objective()` already scores hill-climbing steps without rendering.

## Training (`rl/train.py`)

- Uses `stable_baselines3.SAC` with an `MlpPolicy` over the environment
  above, wrapped via `stable_baselines3.common.env_util.make_vec_env` for
  parallel rollout collection.
- Default budget: 200,000 timesteps (configurable via CLI arg), checkpointed
  to `out/rl_models/sac_tf_agent.zip`.
- Training logs (episode reward, length) go to `out/rl_logs/` via SB3's
  built-in `Monitor` wrapper — reuses the project's existing convention of
  writing all run artifacts under `out/`.

## Evaluation (`rl/eval.py`)

- Builds a fixed, seeded set of held-out (tissue, direction, starting-params)
  triples — same distribution as training's `reset()`, but with a fixed seed
  so the comparison is apples-to-apples and reproducible across runs.
- For each triple, runs two rollouts of `MAX_STEPS` steps:
  1. The trained SAC policy (deterministic action).
  2. The existing hill-climbing baseline
     (`search.propose_step`/`resize_step`, same acceptance rule as
     `evaluate.objective` uses today).
- Reports, per method: mean final `mass_fraction` reached, mean steps to
  reach 90% of the max fraction observed for that triple.
- Writes results to `out/rl_eval_results.json`.

## Testing (`tests/test_rl_env.py`)

- `reset()` returns an observation of the correct shape and within declared
  bounds; the one-hot block sums to 1; the direction flag is ±1.
- `step()` with a positive action on an `increase` goal increases the
  resolved peak's height (bounded by `MAX_DELTA`); a negative action on a
  `decrease` goal decreases it.
- Reward sign matches goal direction: an action that moves `mass_fraction`
  toward the goal yields positive reward, away from the goal yields negative
  reward.
- Height stays clipped to `[0, 1]` internal range even when repeated
  same-sign actions would otherwise push it out of bounds.
- Episode reaches `truncated=True` at exactly `MAX_STEPS` and never sets
  `terminated=True`.
- One short smoke-training test: `SAC(...).learn(total_timesteps=200)` on the
  env completes without error — catches SB3/Gymnasium API integration
  breakage without a slow full training run in the suite.

## Explicitly out of scope

- Wiring the trained policy into the live chat UI (`server.py`/`static/`) as
  a third search mode alongside rule/LLM parsing and hill-climbing. This spec
  covers training and offline evaluation only; UI integration is a natural
  follow-up once a trained policy demonstrably beats the baseline in
  `rl/eval.py`'s report.
- Training against `out/preferences.jsonl`/`out/feedback.jsonl` (offline
  human-preference data). Deferred per the agreed training-signal choice —
  the objective `opacity_mass` reward needs no human data and can start
  immediately; comparing the learned policy against human preferences is a
  later evaluation question, not a training source, for this phase.
- Full 24-dimension or full-peak (center/width/color) action space. Action
  scope is single-peak-height-only per command, matching the existing
  hill-climbing baseline for a direct, fair comparison.

## New dependencies

Added to `requirements.txt`: `torch`, `stable-baselines3`, `gymnasium`.
