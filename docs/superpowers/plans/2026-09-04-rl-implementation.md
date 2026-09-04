# RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gymnasium environment and SAC training/evaluation pipeline that learns to move a single transfer-function peak's height toward a goal tissue/direction, trained purely against the existing `opacity_mass` metric (no rendering, no human data required).

**Architecture:** A new `rl/` package (`env.py`, `train.py`, `eval.py`) sits alongside the existing baseline code without modifying it, except for one new small helper (`mass_fraction`) added to `transfer.py`. `rl/env.py` reuses `commands._find_or_create_peak` for peak resolution and `search.propose_step`/`resize_step` for step application, so the RL path and the hill-climbing baseline apply changes to `params` identically — the only difference is what picks the step.

**Tech Stack:** Python, numpy, Gymnasium, PyTorch, stable-baselines3 (SAC), pytest.

Spec: `docs/superpowers/specs/2026-09-04-rl-implementation-design.md`

---

## File Structure

- Modify: `requirements.txt` — add `torch`, `stable-baselines3`, `gymnasium`.
- Modify: `transfer.py` — add `mass_fraction(params, tissue)`.
- Modify: `pytest.ini` — register the `slow` marker.
- Create: `rl/__init__.py` — empty package marker.
- Create: `rl/env.py` — `TFEnv(gymnasium.Env)`.
- Create: `rl/train.py` — SAC training CLI.
- Create: `rl/eval.py` — policy-vs-hill-climbing comparison CLI.
- Create: `tests/test_transfer.py` (append) — test for `mass_fraction`.
- Create: `tests/test_rl_env.py` — `TFEnv` unit tests + one slow SAC smoke test.

---

### Task 1: Install dependencies and add `mass_fraction` to `transfer.py`

**Files:**
- Modify: `requirements.txt`
- Modify: `transfer.py`
- Modify: `tests/test_transfer.py`

- [ ] **Step 1: Add the new dependencies**

Append to `requirements.txt`:

```
torch
stable-baselines3
gymnasium
```

- [ ] **Step 2: Install them**

Run: `.venv/bin/python -m pip install torch stable-baselines3 gymnasium`
Expected: installs `torch-2.14.0`, `stable_baselines3-2.9.0`, `gymnasium-1.3.0` (or newer patch versions) with no dependency conflicts.

- [ ] **Step 3: Write the failing test for `mass_fraction`**

Append to `tests/test_transfer.py`:

```python
from transfer import mass_fraction


def test_mass_fraction_is_opacity_mass_normalized_by_band_width():
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)
    lo, hi = TISSUE_BANDS["bone"]
    params[N_PEAKS * PARAMS_PER_PEAK - PARAMS_PER_PEAK + 2] = 2.0 * 0.5 - 1.0  # bone peak height = 0.5
    frac = mass_fraction(params, "bone")
    assert 0.0 <= frac <= 1.0
    assert abs(frac - opacity_mass(params, lo, hi) / (hi - lo)) < 1e-9


def test_mass_fraction_zero_for_all_zero_height():
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)
    assert mass_fraction(params, "spongy") < 1e-6
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer.py -k mass_fraction -v`
Expected: FAIL with `ImportError: cannot import name 'mass_fraction'`

- [ ] **Step 5: Implement `mass_fraction` in `transfer.py`**

Add this function immediately after `opacity_mass` (after line 93, before `vector_to_vtk`):

```python
def mass_fraction(params: np.ndarray, tissue: str) -> float:
    """opacity_mass normalized by band width -- an average-opacity-in-band
    fraction in [0, 1], used as the RL observation/reward signal since raw
    opacity_mass scales with band width (bone's band alone is 1400 HU wide)."""
    lo, hi = TISSUE_BANDS[tissue]
    return opacity_mass(params, lo, hi) / (hi - lo)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer.py -v`
Expected: all tests PASS, including the two new ones.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt transfer.py tests/test_transfer.py
git commit -m "Add mass_fraction helper and RL dependencies"
```

---

### Task 2: Build `TFEnv` and its unit tests

**Files:**
- Create: `rl/__init__.py`
- Create: `rl/env.py`
- Create: `tests/test_rl_env.py`

- [ ] **Step 1: Create the package marker**

Create `rl/__init__.py` (empty file).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_rl_env.py`:

```python
"""Tests for the TFEnv Gymnasium environment."""
import numpy as np
import pytest

from rl.env import DIRECTIONS, MAX_STEPS, PARAMS_PER_PEAK, TISSUES, TFEnv


def test_reset_observation_shape_and_bounds():
    env = TFEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (31,)
    onehot = obs[24:29]
    assert onehot.sum() == pytest.approx(1.0)
    assert set(np.unique(onehot)) <= {0.0, 1.0}
    assert obs[29] in (-1.0, 1.0)
    assert 0.0 <= obs[30] <= 1.0


def test_reset_picks_valid_tissue_and_direction():
    env = TFEnv(seed=1)
    env.reset()
    assert env.target_tissue in TISSUES
    assert env.direction in DIRECTIONS


def test_step_moves_height_in_action_direction():
    env = TFEnv(seed=2)
    env.reset()
    base = env.peak_idx * PARAMS_PER_PEAK
    height_before = env.params[base + 2]
    env.step(np.array([1.0], dtype=np.float32))
    height_after = env.params[base + 2]
    assert height_after > height_before or height_before >= 1.0 - 1e-6


def test_reward_sign_matches_goal_direction():
    env = TFEnv(seed=3)
    env.reset()
    env.direction = "increase"
    _, reward, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert reward >= 0.0

    env.reset()
    env.direction = "decrease"
    _, reward, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
    assert reward <= 0.0


def test_height_stays_clipped_to_unit_range():
    env = TFEnv(seed=4)
    env.reset()
    base = env.peak_idx * PARAMS_PER_PEAK
    for _ in range(50):
        env.step(np.array([1.0], dtype=np.float32))
    internal_height = (env.params[base + 2] + 1.0) / 2.0
    assert -1e-6 <= internal_height <= 1.0 + 1e-6


def test_episode_truncates_at_max_steps_and_never_terminates():
    env = TFEnv(seed=5)
    env.reset()
    truncated = False
    for i in range(MAX_STEPS):
        _, _, terminated, truncated, _ = env.step(np.array([0.1], dtype=np.float32))
        assert terminated is False
        if i < MAX_STEPS - 1:
            assert truncated is False
    assert truncated is True
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rl_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.env'`

- [ ] **Step 4: Implement `rl/env.py`**

```python
"""Gymnasium environment for training a continuous-control TF agent directly
against the opacity_mass metric -- no rendering involved, matching how
evaluate.objective() already scores hill-climbing steps on raw params."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from commands import _find_or_create_peak
from search import propose_step
from transfer import N_PEAKS, PARAMS_PER_PEAK, default_params, mass_fraction

TISSUES = ["air", "fat", "soft", "spongy", "bone"]
DIRECTIONS = ["increase", "decrease"]
MAX_DELTA = 0.2
MAX_STEPS = 20
HEIGHT_NOISE = 0.3
OBS_SIZE = N_PEAKS * PARAMS_PER_PEAK + len(TISSUES) + 2


class TFEnv(gym.Env):
    """One episode = one (target_tissue, direction) goal; the action moves
    the resolved target peak's height each step."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.params = None
        self.target_tissue = None
        self.direction = None
        self.peak_idx = None
        self.step_count = 0

    def _sample_start_params(self) -> np.ndarray:
        params = default_params().copy()
        for i in range(N_PEAKS):
            base = i * PARAMS_PER_PEAK
            noise = self._rng.uniform(-HEIGHT_NOISE, HEIGHT_NOISE)
            params[base + 2] = float(np.clip(params[base + 2] + noise, -1.0, 1.0))
        return params

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(len(TISSUES), dtype=np.float32)
        onehot[TISSUES.index(self.target_tissue)] = 1.0
        direction_flag = 1.0 if self.direction == "increase" else -1.0
        frac = mass_fraction(self.params, self.target_tissue)
        return np.concatenate([
            self.params.astype(np.float32),
            onehot,
            np.array([direction_flag, frac], dtype=np.float32),
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        params = self._sample_start_params()
        self.target_tissue = TISSUES[self._rng.integers(0, len(TISSUES))]
        self.direction = DIRECTIONS[self._rng.integers(0, len(DIRECTIONS))]
        self.params, self.peak_idx = _find_or_create_peak(params, self.target_tissue)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        direction_sign = 1.0 if self.direction == "increase" else -1.0
        delta = float(np.clip(action[0], -1.0, 1.0)) * MAX_DELTA
        before_frac = mass_fraction(self.params, self.target_tissue)
        self.params = propose_step(self.params, self.peak_idx, sign=1.0, step=delta)
        after_frac = mass_fraction(self.params, self.target_tissue)
        reward = direction_sign * (after_frac - before_frac)
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        terminated = False
        info = {"mass_fraction_after": after_frac}
        return self._obs(), reward, terminated, truncated, info
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rl_env.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rl/__init__.py rl/env.py tests/test_rl_env.py
git commit -m "Add TFEnv Gymnasium environment for the RL agent"
```

---

### Task 3: Add the SAC smoke-training test

**Files:**
- Modify: `tests/test_rl_env.py`
- Modify: `pytest.ini`

This is a separate task from Task 2 because it exercises the actual `stable_baselines3.SAC` integration (catching Gymnasium/SB3 API mismatches), not just the environment's own logic, and it is slow enough (a few seconds) to be worth marking and excludable from a fast test run.

- [ ] **Step 1: Register the `slow` marker**

Edit `pytest.ini` to:

```ini
[pytest]
pythonpath = .
markers =
    slow: exercises real training/inference (e.g. SB3), not run by default in fast test loops
```

- [ ] **Step 2: Append the smoke test**

Append to `tests/test_rl_env.py`:

```python
@pytest.mark.slow
def test_sac_smoke_training_runs_without_error():
    from stable_baselines3 import SAC

    env = TFEnv(seed=6)
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=200)
```

- [ ] **Step 3: Run it to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rl_env.py -v -m slow`
Expected: PASS (takes on the order of seconds; SB3 will print training setup info to stdout, that's normal).

- [ ] **Step 4: Run the full suite once to confirm nothing else broke**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests PASS (69 previous + the new ones from Tasks 1-3).

- [ ] **Step 5: Commit**

```bash
git add tests/test_rl_env.py pytest.ini
git commit -m "Add SAC smoke-training test"
```

---

### Task 4: Training CLI (`rl/train.py`)

**Files:**
- Create: `rl/train.py`

- [ ] **Step 1: Implement the training script**

```python
"""Train a SAC agent on TFEnv against the opacity_mass-derived reward."""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rl.env import TFEnv

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
LOG_DIR = "out/rl_logs"


def _make_env():
    return Monitor(TFEnv())


def train(total_timesteps: int, n_envs: int = 4, seed: int = 0) -> SAC:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    vec_env = make_vec_env(_make_env, n_envs=n_envs, seed=seed)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_PATH)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the TF SAC agent.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.timesteps, n_envs=args.n_envs, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run it with a tiny timestep budget**

Run: `.venv/bin/python -m rl.train --timesteps 500 --n-envs 2`
Expected: exits 0, prints SB3 training progress, and creates `out/rl_models/sac_tf_agent.zip` and a non-empty `out/rl_logs/` directory.

- [ ] **Step 3: Verify the saved model loads back**

Run:
```bash
.venv/bin/python -c "
from stable_baselines3 import SAC
model = SAC.load('out/rl_models/sac_tf_agent.zip')
print(model.policy)
"
```
Expected: prints the MLP policy architecture with no errors.

- [ ] **Step 4: Commit**

`out/` is gitignored, so the smoke-run artifacts are not staged.

```bash
git add rl/train.py
git commit -m "Add SAC training CLI"
```

---

### Task 5: Evaluation CLI (`rl/eval.py`)

**Files:**
- Create: `rl/eval.py`

- [ ] **Step 1: Implement the evaluation script**

```python
"""Compare the trained SAC policy against the hill-climbing baseline over
the same held-out (tissue, direction, start-params) episodes."""
import json

import numpy as np
from stable_baselines3 import SAC

from rl.env import MAX_DELTA, MAX_STEPS, TFEnv
from search import propose_step, resize_step
from transfer import mass_fraction

MODEL_PATH = "out/rl_models/sac_tf_agent.zip"
RESULTS_PATH = "out/rl_eval_results.json"
N_EPISODES = 20
EVAL_SEED = 12345


def _build_episodes(n_episodes: int, seed: int) -> list:
    env = TFEnv(seed=seed)
    episodes = []
    for _ in range(n_episodes):
        env.reset()
        episodes.append({
            "params": env.params.copy(),
            "target_tissue": env.target_tissue,
            "direction": env.direction,
            "peak_idx": env.peak_idx,
        })
    return episodes


def _run_policy(model, episode: dict) -> list:
    env = TFEnv()
    env.params = episode["params"].copy()
    env.target_tissue = episode["target_tissue"]
    env.direction = episode["direction"]
    env.peak_idx = episode["peak_idx"]
    env.step_count = 0
    fractions = [mass_fraction(env.params, env.target_tissue)]
    obs = env._obs()
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        fractions.append(info["mass_fraction_after"])
    return fractions


def _run_hill_climb(episode: dict) -> list:
    params = episode["params"].copy()
    peak_idx = episode["peak_idx"]
    tissue = episode["target_tissue"]
    sign = 1.0 if episode["direction"] == "increase" else -1.0
    step = MAX_DELTA
    fractions = [mass_fraction(params, tissue)]
    for _ in range(MAX_STEPS):
        proposed = propose_step(params, peak_idx, sign=sign, step=step)
        before = fractions[-1]
        after = mass_fraction(proposed, tissue)
        accepted = sign * (after - before) > 0.0
        if accepted:
            params = proposed
        step = resize_step(step, accepted)
        fractions.append(mass_fraction(params, tissue))
    return fractions


def _steps_to_90pct(fractions: list, start: float, best: float) -> int:
    target = start + 0.9 * (best - start)
    for i, f in enumerate(fractions):
        if (best >= start and f >= target) or (best < start and f <= target):
            return i
    return len(fractions) - 1


def evaluate(model_path: str = MODEL_PATH, n_episodes: int = N_EPISODES, seed: int = EVAL_SEED) -> dict:
    model = SAC.load(model_path)
    episodes = _build_episodes(n_episodes, seed)

    policy_finals, policy_steps = [], []
    hillclimb_finals, hillclimb_steps = [], []
    for ep in episodes:
        pf = _run_policy(model, ep)
        hf = _run_hill_climb(ep)
        combined = pf + hf
        best = max(combined) if ep["direction"] == "increase" else min(combined)
        policy_finals.append(pf[-1])
        hillclimb_finals.append(hf[-1])
        policy_steps.append(_steps_to_90pct(pf, pf[0], best))
        hillclimb_steps.append(_steps_to_90pct(hf, hf[0], best))

    results = {
        "n_episodes": n_episodes,
        "policy": {
            "mean_final_mass_fraction": float(np.mean(policy_finals)),
            "mean_steps_to_90pct": float(np.mean(policy_steps)),
        },
        "hill_climb": {
            "mean_final_mass_fraction": float(np.mean(hillclimb_finals)),
            "mean_steps_to_90pct": float(np.mean(hillclimb_steps)),
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
```

- [ ] **Step 2: Run it against the smoke-trained model from Task 4**

Run: `.venv/bin/python -m rl.eval`
Expected: exits 0, prints a JSON object with `policy` and `hill_climb` sub-objects, and writes `out/rl_eval_results.json`. With only 500 training timesteps the policy's numbers may be poor (untrained) — the goal here is confirming the *pipeline* runs end to end, not good numbers yet. A real comparison happens after the full 200k-timestep run in Task 6.

- [ ] **Step 3: Commit**

```bash
git add rl/eval.py
git commit -m "Add policy-vs-hill-climbing evaluation CLI"
```

---

### Task 6: Full training run and real comparison

**Files:** none (execution only)

- [ ] **Step 1: Run full training**

Run: `.venv/bin/python -m rl.train --timesteps 200000` (background this — it will take a while; on CPU-only PyTorch this small MLP/env pair should still complete in well under an hour given `opacity_mass` has no rendering cost).

- [ ] **Step 2: Run evaluation against the fully-trained model**

Run: `.venv/bin/python -m rl.eval`
Expected: `out/rl_eval_results.json` now reflects a properly trained policy. Report both methods' `mean_final_mass_fraction` and `mean_steps_to_90pct` to the user.

- [ ] **Step 3: Report results, no commit**

`out/` is gitignored — this task produces a trained model and a results file for the user to look at, not a code change. Summarize the comparison numbers in chat.

---

## Notes for the implementer

- Do not modify `render.py`, `evaluate.py`, `search.py`, or `commands.py` — the design deliberately reuses their existing functions unchanged.
- `commands._find_or_create_peak` is a "private" (underscore-prefixed) function; importing it across modules is intentional here per the design spec, not an oversight.
- If `test_sac_smoke_training_runs_without_error` is flaky on CI (SB3 occasionally prints deprecation warnings to stderr that some CI configs treat as failures), that's a warning-vs-error CI config issue, not a real failure — do not silence it by weakening the test's actual assertions.
