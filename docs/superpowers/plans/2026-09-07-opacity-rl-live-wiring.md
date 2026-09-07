# Opacity RL Live Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third `evaluator` option, `"policy"`, to the chat UI's opacity search so the already-trained SAC agent (`out/rl_models/sac_tf_agent.zip`) can run live, alongside the existing `objective` and `human` hill-climb evaluators.

**Architecture:** A new `rl/serve.py` module lazily loads the trained SAC model (same cache-on-first-use pattern `asr.py` already uses for Whisper) and rolls it forward through a `TFEnv` instance for a fixed number of steps, mirroring `rl/eval.py`'s `_run_policy`. `Session.command()` in `server.py` gains one new branch inside its existing search gate that calls this instead of the hill-climb loop. The frontend's evaluator toggle becomes a three-way cycle instead of two.

**Tech Stack:** Python, FastAPI, stable-baselines3 (SAC), Gymnasium, vanilla JS (no framework/build step).

Spec: `docs/superpowers/specs/2026-09-07-opacity-rl-live-wiring-design.md`

---

### Task 1: `rl/serve.py` — load and run the trained policy

**Files:**
- Create: `rl/serve.py`
- Test: `tests/test_rl_serve.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rl_serve.py`:

```python
"""Tests for rl/serve.py -- the live-serving counterpart to rl/eval.py's
offline comparison. Uses a fake model (not the real trained SAC agent) so
these run fast and don't depend on out/rl_models/sac_tf_agent.zip existing."""
import numpy as np
import pytest

import rl.serve as serve
from commands import _find_or_create_peak
from transfer import default_params


def test_run_policy_raises_when_model_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.zip")
    params = default_params()
    with pytest.raises(ValueError, match="no trained model"):
        serve.run_policy(params, "bone", "increase", peak_idx=0, steps=3, model_path=missing_path)


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([0.5], dtype=np.float32), None


def test_run_policy_moves_target_peak_and_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(serve, "_load_model", lambda model_path=serve.MODEL_PATH: _FakeModel())
    params, peak_idx = _find_or_create_peak(default_params(), "bone")
    original = params.copy()

    result = serve.run_policy(params, "bone", "increase", peak_idx, steps=3, model_path="unused")

    assert result.shape == params.shape
    assert not np.array_equal(result, original)  # the fake model's action moved the height
    assert np.array_equal(params, original)  # input array itself untouched


def test_run_policy_caches_loaded_model(monkeypatch, tmp_path):
    calls = []

    class _CountingFakeModel(_FakeModel):
        pass

    def fake_sac_load(path):
        calls.append(path)
        return _CountingFakeModel()

    monkeypatch.setattr(serve.SAC, "load", staticmethod(fake_sac_load))
    model_zip = tmp_path / "fake.zip"
    model_zip.write_text("not a real model, never opened by the fake loader")
    serve._model_cache.clear()

    params, peak_idx = _find_or_create_peak(default_params(), "bone")
    serve.run_policy(params, "bone", "increase", peak_idx, steps=1, model_path=str(model_zip))
    serve.run_policy(params, "bone", "increase", peak_idx, steps=1, model_path=str(model_zip))

    assert len(calls) == 1  # second call reused the cached model, did not call SAC.load again
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rl_serve.py -v`
Expected: `ModuleNotFoundError: No module named 'rl.serve'` (or collection error) — the module doesn't exist yet.

- [ ] **Step 3: Write `rl/serve.py`**

```python
"""Runs the trained opacity SAC policy against live params -- the serving
counterpart to rl/eval.py's offline policy-vs-hill-climb comparison."""
import os

import numpy as np
from stable_baselines3 import SAC

from rl.env import TFEnv

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
_model_cache = {}


def _load_model(model_path: str = MODEL_PATH):
    if model_path not in _model_cache:
        if not os.path.exists(model_path):
            raise ValueError(
                f"no trained model at {model_path!r} -- run `python -m rl.train` first"
            )
        _model_cache[model_path] = SAC.load(model_path)
    return _model_cache[model_path]


def run_policy(params: np.ndarray, target_tissue: str, direction: str,
               peak_idx: int, steps: int, model_path: str = MODEL_PATH) -> np.ndarray:
    """Rolls the trained policy forward `steps` times from `params`, seeded
    with the same (tissue, direction, peak_idx) shape rl/eval.py's
    _run_policy uses. Raises ValueError if no trained model exists at
    `model_path`. Does not mutate the input `params` array."""
    model = _load_model(model_path)
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rl_serve.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 3 from 156 — 159 passed, 2 deselected.

- [ ] **Step 6: Commit**

```bash
git add rl/serve.py tests/test_rl_serve.py
git commit -m "Add rl/serve.py to run the trained opacity SAC policy against live params"
```

---

### Task 2: Wire `evaluator="policy"` into `Session.command()`

**Files:**
- Modify: `server.py:29` (import), `server.py:263-299` (`Session.command`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`, near `test_objective_search_appends_one_final_step` (after line 81):

```python
def test_policy_search_raises_when_no_trained_model():
    # _isolate_cwd (autouse) has already chdir'd to a scratch tmp_path, so
    # out/rl_models/sac_tf_agent.zip does not exist here even if the real
    # repo has a trained model on disk -- this is the fresh-checkout/CI path.
    s = _fresh_session()
    with pytest.raises(ValueError, match="no trained model"):
        s.command("increase opacity for bone strongly", parser="rule",
                   search=True, evaluator="policy", steps=3)
    assert s.state()["total"] == 1  # nothing appended on failure


def test_policy_search_appends_one_final_step(monkeypatch):
    import server as server_module

    def fake_run_policy(params, tissue, direction, idx, steps):
        moved = params.copy()
        moved[idx * 3 + 2] = min(1.0, moved[idx * 3 + 2] + 0.1)
        return moved

    monkeypatch.setattr(server_module, "run_policy", fake_run_policy)

    s = _fresh_session()
    state = s.command("increase opacity for bone strongly", parser="rule",
                       search=True, evaluator="policy", steps=5)
    assert state["total"] == 2  # one new step, not one per iteration
    assert state["current"]["search"] is True
```

Check whether `tests/test_server.py` already imports numpy: `command grep -n "^import numpy" tests/test_server.py`. This test doesn't itself need it (no `np.` calls), so no import change is required unless a later step in this task adds one.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k policy_search`
Expected: both FAIL — `test_policy_search_raises_when_no_trained_model` fails because `evaluator="policy"` currently falls through to the `else` branch (direct apply, not search) rather than raising; `test_policy_search_appends_one_final_step` fails with `AttributeError: <module 'server'> does not have the attribute 'run_policy'` from `monkeypatch.setattr`.

- [ ] **Step 3: Add the import**

In `server.py`, line 29 currently reads:

```python
from commands import COMMAND_REFERENCE, STRENGTH_WORDS, _find_or_create_peak, apply_command, parse_command
```

Add a new import line directly after it:

```python
from commands import COMMAND_REFERENCE, STRENGTH_WORDS, _find_or_create_peak, apply_command, parse_command
from rl.serve import run_policy
```

- [ ] **Step 4: Add the `policy` branch to `Session.command()`**

In `server.py`, the search block currently (lines 278–284) reads:

```python
        if search and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
            if evaluator == "human":
                self.pending = self._start_human_search(text, cmd, current_params, steps, current_camera)
                return self.state()
            new_params = self._run_objective_search(cmd, current_params, steps)
            step = _render_step(new_params, text, cmd, None, True,
                                 self.history[-1]["id"] + 1, self.session_id, current_camera)
```

Replace it with:

```python
        if search and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
            if evaluator == "human":
                self.pending = self._start_human_search(text, cmd, current_params, steps, current_camera)
                return self.state()
            if evaluator == "policy":
                _, peak_idx = _find_or_create_peak(current_params, cmd["target"])
                new_params = run_policy(current_params, cmd["target"], cmd["direction"], peak_idx, steps)
            else:
                new_params = self._run_objective_search(cmd, current_params, steps)
            step = _render_step(new_params, text, cmd, None, True,
                                 self.history[-1]["id"] + 1, self.session_id, current_camera)
            if evaluator == "policy":
                jsonl_append(LOG_PATH, {
                    "timestamp": step["timestamp"], "command": cmd,
                    "params_before": current_params.tolist(), "params_after": new_params.tolist(),
                    "features_before": self.history[self.cursor]["features"], "features_after": step["features"],
                    "verdict": None,
                })
```

The new `jsonl_append` is deliberately gated on `evaluator == "policy"` — it must **not** fire for the `objective` branch, which already gets its own per-iteration logging inside `_run_objective_search` (unchanged, still the only logging for that path — no outer-level log existed for it before this change, and none should be added now). After this edit, run `grep -n "jsonl_append" server.py` and confirm there are exactly two calls inside `Session.command()`/`_run_objective_search` for the search paths: the pre-existing per-iteration one inside `_run_objective_search` (~line 251) and this new one, plus the untouched one in the `else` (direct-apply) branch below.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k policy_search`
Expected: 2 passed.

- [ ] **Step 6: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 2 from 159 (3 from Task 1 + 2 from this task) — 161 passed, 2 deselected.

- [ ] **Step 7: Manual smoke test against the real trained model**

The real `out/rl_models/sac_tf_agent.zip` exists in the project's own `out/` directory (untracked, gitignored) — this step exercises it for real, unlike the mocked unit tests above.

Start the server (`.venv/bin/python server.py`) from the project root, then in another terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/api/command -H "Content-Type: application/json" \
  -d '{"text": "increase opacity for bone strongly", "search": true, "evaluator": "policy", "steps": 5}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['current']; print('search:', c['search']); print('cmd_text:', c['cmd_text']); print('bone mass:', c['masses']['bone'])"
```

Expected: `search: True`, `cmd_text: increase opacity for bone strongly`, and a `bone mass` value (compare it's a plausible float, not an error). Stop the server afterward (`kill` the process or Ctrl-C).

- [ ] **Step 8: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "Wire evaluator=policy into Session.command() to run the live opacity RL policy"
```

---

### Task 3: Frontend — three-way evaluator toggle

**Files:**
- Modify: `static/app.js:233-237`

This task has no automated test (this project has no JS test framework — every existing frontend feature was verified manually in the browser; see Step 2 below).

- [ ] **Step 1: Change the toggle to cycle three ways**

In `static/app.js`, lines 233–237 currently read:

```javascript
const evaluatorToggle = el("evaluator-toggle");
evaluatorToggle.addEventListener("click", () => {
  config.evaluator = config.evaluator === "objective" ? "human" : "objective";
  evaluatorToggle.textContent = config.evaluator;
});
```

Replace with:

```javascript
const EVALUATOR_CYCLE = ["objective", "human", "policy"];
const evaluatorToggle = el("evaluator-toggle");
evaluatorToggle.addEventListener("click", () => {
  const next = (EVALUATOR_CYCLE.indexOf(config.evaluator) + 1) % EVALUATOR_CYCLE.length;
  config.evaluator = EVALUATOR_CYCLE[next];
  evaluatorToggle.textContent = config.evaluator;
});
```

- [ ] **Step 2: Manual verification in the browser**

Start the server: `.venv/bin/python server.py`, open `http://127.0.0.1:8000`.

1. Click the "search" chip to reveal the evaluator toggle (it should read "objective").
2. Click the evaluator chip repeatedly and confirm it cycles `objective -> human -> policy -> objective -> ...` with no other UI change.
3. With the toggle on "policy", type "increase opacity for bone strongly" and send it.
   - If `out/rl_models/sac_tf_agent.zip` exists (it does in this project's `out/` directory): confirm a new message appears in the chat log tagged "search", the rendered image updates, and no error banner appears.
   - Temporarily rename `out/rl_models/sac_tf_agent.zip` to `out/rl_models/sac_tf_agent.zip.bak`, repeat the same command, and confirm the chat log shows a parse-error-style message containing "no trained model" (reusing the existing `appendError` path — see `sendCommand`'s `if (r.status !== 200)` branch in `static/app.js`). Rename the file back afterward.
4. Click the evaluator chip back to "objective" and confirm a normal hill-climb command still works, to rule out any regression.

Stop the server afterward.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "Add policy as a third evaluator option in the chat UI's search toggle"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Reinforcement learning section**

In `README.md`, lines 110–113 currently read:

```
      python -m rl.train --timesteps 200000
      python -m rl.eval

- **`rl/camera_env.py` + `rl/camera_train.py` + `rl/camera_eval.py`** — camera
```

Replace with:

```
      python -m rl.train --timesteps 200000
      python -m rl.eval

  The trained policy is also wired live into the chat UI's search toggle
  (`server.py`, evaluator = policy — no CLI equivalent) — see `rl/serve.py`.

- **`rl/camera_env.py` + `rl/camera_train.py` + `rl/camera_eval.py`** — camera
```

- [ ] **Step 2: Update the "Neither is wired into the live loop" framing**

The Reinforcement learning section's intro currently reads:

```
**Neither is wired into the live chat/voice loop yet** — this is offline
training + evaluation only.
```

Replace with:

```
**The camera-viewpoint policy is not wired into the live loop yet** — the
opacity policy now is (chat UI only, see below); both remain trained and
evaluated offline first.
```

- [ ] **Step 3: Update the closing paragraph**

The paragraph starting "The hand-coded hill-climbing baseline (`search.py` + `evaluate.py`) remains the live loop's optimizer" currently reads:

```
The hand-coded hill-climbing baseline (`search.py` + `evaluate.py`) remains the
live loop's optimizer — both RL agents above are trained and evaluated
offline against this baseline, not yet wired into `mvp.py`/`server.py`.
```

Replace with:

```
The hand-coded hill-climbing baseline (`search.py` + `evaluate.py`) remains
the default live-loop optimizer, and the only one `mvp.py` uses. `server.py`
additionally offers the trained opacity policy as a third search option
(`evaluator=policy`); the camera-viewpoint policy is trained and evaluated
offline only, not yet wired into either loop.
```

- [ ] **Step 4: Add `rl/serve.py` to the Files list**

The Files list currently has this line:

```
- `rl/env.py`, `rl/train.py`, `rl/eval.py` — offline SAC RL for transfer-function opacity
```

Replace with:

```
- `rl/env.py`, `rl/train.py`, `rl/eval.py`, `rl/serve.py` — SAC RL for transfer-function opacity (train/eval offline, serve live)
```

- [ ] **Step 5: Verify the doc changes render sensibly**

Run: `command grep -n "wired into\|rl/serve" README.md`
Expected: the three updated passages and the Files-list line all appear, and none of them still say "Neither is wired" (that phrase should no longer exist verbatim — confirm with `command grep -n "Neither is wired" README.md`, expected: no output).

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Document the live opacity RL policy option in the chat UI"
```

---

## Explicitly out of scope (see spec)

- `mvp.py` CLI — untouched.
- Camera-viewpoint policy wiring — untouched, separate future effort.
- `docs/architecture.typ` / `architecture-mvp.drawio` — not updated by this plan; revisit once this is complete, as agreed separately.
