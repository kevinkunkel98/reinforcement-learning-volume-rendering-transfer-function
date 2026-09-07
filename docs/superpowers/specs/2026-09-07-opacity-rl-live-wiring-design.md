# Opacity RL Live Wiring — Design

## Goal

Wire the already-trained opacity SAC policy (`rl/env.py` + `rl/train.py`,
`out/rl_models/sac_tf_agent.zip`) into the live chat UI as a third search
strategy, alongside the existing objective and human hill-climb evaluators.
Camera-viewpoint RL and the `mvp.py` CLI are explicitly out of scope (see
below) — this covers `server.py` and the chat UI only.

## Context

`Session.command()` already has one gate for entering a multi-step search:
`search=True`, `cmd["attribute"] == "opacity"`, `cmd["direction"] in
("increase", "decrease")`. Inside that gate, `evaluator` picks the strategy:
`"objective"` runs `_run_objective_search` (hill-climb via
`search.propose_step`/`resize_step`, auto-accepted by `evaluate.objective`),
`"human"` runs `_start_human_search` (same hill-climb, judged step-by-step
via `/api/judge`). `rl/eval.py` already demonstrates exactly how to drive a
trained policy: `SAC.load(path)`, then `model.predict(obs, deterministic=True)`
stepped through a `TFEnv` instance seeded with the live params/tissue/
direction/peak_idx. This design adds `evaluator="policy"` as a third
strategy reusing that same gate, calling the model instead of hill-climbing.

## Architecture / data flow

```
User command ("increase opacity for bone", search=on, evaluator=policy)
        |
        v
Session.command()  --(existing gate: search & opacity & inc/dec)--+
        |                                                          |
   evaluator == "policy"                                           |
        |                                                          |
        v                                                          |
rl.serve.run_policy(params, tissue, direction, peak_idx, steps) ----+
        |  loads out/rl_models/sac_tf_agent.zip once (lazy, cached)
        |  drives TFEnv.step() with model.predict() for `steps` iterations
        v
final params  ->  _render_step(..., search=True)  ->  appended to history,
                   same as the objective/human paths today
```

## New code: `rl/serve.py`

```python
"""Runs the trained opacity SAC policy against live params -- the serving
counterpart to rl/eval.py's offline comparison."""
import numpy as np
from stable_baselines3 import SAC

from rl.env import TFEnv

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
_model_cache = {}


def _load_model(model_path: str = MODEL_PATH):
    if model_path not in _model_cache:
        import os
        if not os.path.exists(model_path):
            raise ValueError(
                f"no trained model at {model_path!r} -- run `python -m rl.train` first"
            )
        _model_cache[model_path] = SAC.load(model_path)
    return _model_cache[model_path]


def run_policy(params: np.ndarray, target_tissue: str, direction: str,
               peak_idx: int, steps: int) -> np.ndarray:
    """Rolls the trained policy forward `steps` times from `params`, seeded
    with the same (tissue, direction, peak_idx) shape rl/eval.py uses.
    Raises ValueError if no trained model exists yet. Does not mutate the
    input `params` array."""
    model = _load_model()
    env = TFEnv()
    env.params = params.copy()
    env.target_tissue = target_tissue
    env.direction = direction
    env.peak_idx = peak_idx
    env.step_count = 0
    obs = env._obs()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
    return env.params
```

Model loading follows the same lazy-cache convention `asr.py` already uses
for the Whisper model (`_MODEL_CACHE` keyed dict, loaded on first use, reused
after) — no eager loading at server startup, no reload per request.

## `server.py` changes

- Import `run_policy` from `rl.serve`.
- In `command()`, extend the search branch:

  ```python
  if search and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
      if evaluator == "human":
          self.pending = self._start_human_search(text, cmd, current_params, steps, current_camera)
          return self.state()
      if evaluator == "policy":
          _, idx = _find_or_create_peak(current_params, cmd["target"])
          new_params = run_policy(current_params, cmd["target"], cmd["direction"], idx, steps)
      else:
          new_params = self._run_objective_search(cmd, current_params, steps)
      step = _render_step(new_params, text, cmd, None, True,
                           self.history[-1]["id"] + 1, self.session_id, current_camera)
  ```

- One `jsonl_append(LOG_PATH, ...)` after the rollout (start params -> final
  params, `verdict: None`), matching the direct-`apply_command` logging
  style in the non-search `else` branch — not per-step like
  `_run_objective_search`'s accept/reject logging, since a policy step has
  no accept/reject verdict to record (it always applies its action).
- Missing model file: `run_policy` raises `ValueError`, already caught by
  the existing `except ValueError -> HTTPException(400)` handler on
  `/api/command`. No new error handling needed.

## Frontend changes (`static/app.js`, `static/index.html`)

- `evaluatorToggle`'s click handler cycles three-way instead of two:
  `objective -> human -> policy -> objective`. No other UI change — same
  search checkbox, same `steps-input` (reused directly as the policy
  rollout's step count).
- A policy-search history step is tagged `"search"` in the chat log exactly
  like an objective hill-climb step (`_render_step(..., search=True)`),
  since from the UI's perspective both are "ran a multi-step search
  autonomously to a final result" — no new tag.

## What does not change

- `mvp.py` and its `--learn`/`--evaluator` CLI flags — untouched.
- The camera-viewpoint policy (`rl/camera_env.py`, `rl/camera_eval.py`) and
  all camera commands — untouched.
- `_run_objective_search`, `_start_human_search`, `search.py` — untouched;
  policy mode is an additional branch, not a replacement.
- `rl/env.py`, `rl/train.py`, `rl/eval.py` — untouched; `rl/serve.py` reuses
  `TFEnv` but adds no new training/eval logic.

## Testing

- `tests/test_rl_serve.py` (new):
  - `run_policy` raises `ValueError` when `out/rl_models/sac_tf_agent.zip`
    does not exist (point `MODEL_PATH` at a temp path via monkeypatch, or a
    guaranteed-missing path).
  - Given a trained model (skip/xfail in CI if `out/rl_models/sac_tf_agent.zip`
    isn't present — it's gitignored, not committed), `run_policy` returns a
    24-float array of the same shape as the input and does not mutate the
    input array.
- `tests/test_server.py`: extend the existing search/evaluator coverage with
  an `evaluator="policy"` case asserting the `ValueError` -> 400 path when no
  model is present (the default state of a fresh checkout/CI run).

## Explicitly out of scope

- `mvp.py` CLI wiring (per your answer — chat UI only).
- Camera-viewpoint policy wiring (separate, larger effort — needs a new
  "aim at tissue" command shape that doesn't exist in the live grammar
  today, unlike opacity search which already has a slot for it).
- Any change to how `out/rl_models/sac_tf_agent.zip` is trained or
  versioned — this only consumes an existing trained artifact.
- Retraining or re-evaluating the policy; `rl/eval.py`'s offline comparison
  numbers stand as-is.
