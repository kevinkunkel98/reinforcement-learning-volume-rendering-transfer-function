# Camera-Viewpoint RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an SAC agent to find a good camera viewpoint for a named tissue, using a centroid-alignment reward computed directly from a real CT dataset's HU array (no VTK rendering), and evaluate it against a 2D hill-climbing baseline.

**Architecture:** A new `rl/camera_env.py` (parallel to the existing `rl/env.py`), `rl/camera_train.py`, and `rl/camera_eval.py` (parallel to `rl/train.py`/`rl/eval.py`). No changes to the existing TF-opacity RL files. Trains against a real, already-cached CT dataset (`ct_skull` / `data/CT-brain.nrrd`) rather than the synthetic phantom, since the phantom's `soft`/`fat`/`air` tissues have no genuine off-center structure (verified by inspecting `phantom.py`).

**Tech Stack:** Existing stack only (numpy, gymnasium, stable-baselines3) — no new dependencies.

Spec: `docs/superpowers/specs/2026-09-06-camera-viewpoint-rl-design.md`

---

## File Structure

- Create: `rl/camera_env.py` — `CameraViewpointEnv`, `_view_direction`, `_tissue_centroid_direction`.
- Create: `rl/camera_train.py` — SAC training CLI.
- Create: `rl/camera_eval.py` — policy-vs-hill-climb comparison CLI.
- Create: `tests/test_camera_env.py`.
- Create: `tests/test_camera_eval.py`.

---

### Task 1: `rl/camera_env.py`

**Files:**
- Create: `rl/__init__.py` — already exists from the earlier RL plan; do not recreate, just confirm it's there.
- Create: `rl/camera_env.py`
- Create: `tests/test_camera_env.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_camera_env.py`:

```python
"""Tests for rl/camera_env.py -- pure geometry + environment logic, no rendering."""
import numpy as np
import pytest

from phantom import build_phantom
from rl.camera_env import (
    ELEVATION_RANGE, MAX_STEPS, TISSUES,
    CameraViewpointEnv, _tissue_centroid_direction, _view_direction,
)


def test_view_direction_is_unit_vector():
    for az in (0.0, 45.0, 130.0, 300.0):
        for el in (-60.0, 0.0, 60.0):
            v = _view_direction(az, el)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_view_direction_reference_orientation():
    v = _view_direction(0.0, 0.0)
    assert np.allclose(v, [0.0, 0.0, 1.0], atol=1e-9)


def test_synthetic_phantom_soft_fat_air_have_no_centroid_direction():
    # Verifies the finding that motivated training on real data instead:
    # these three tissues are built as spherically-symmetric blobs centered
    # on the volume's true center in phantom.py, so their centroid direction
    # is undefined.
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    for tissue in ("soft", "fat", "air"):
        assert _tissue_centroid_direction(volume, spacing, tissue) is None


def test_synthetic_phantom_bone_and_spongy_have_a_centroid_direction():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    for tissue in ("bone", "spongy"):
        direction = _tissue_centroid_direction(volume, spacing, tissue)
        assert direction is not None
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6


def test_reset_on_synthetic_phantom_always_picks_bone_or_spongy():
    # The phantom has no valid direction for soft/fat/air, so reset() must
    # never select them, across many random seeds.
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    for seed in range(30):
        env = CameraViewpointEnv(volume, spacing, seed=seed)
        env.reset(seed=seed)
        assert env.target_tissue in ("bone", "spongy")


def test_reset_observation_shape_and_bounds():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=0)
    obs, info = env.reset()
    assert obs.shape == (9,)
    onehot = obs[3:8]
    assert onehot.sum() == pytest.approx(1.0)
    assert set(np.unique(onehot)) <= {0.0, 1.0}
    assert -1.0 <= obs[8] <= 1.0


def test_step_moves_azimuth_and_elevation():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=1)
    env.reset()
    before_az, before_el = env.azimuth, env.elevation
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert env.azimuth != before_az
    assert env.elevation != before_el or before_el >= ELEVATION_RANGE[1] - 1e-6


def test_elevation_stays_clamped_under_repeated_extreme_action():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=2)
    env.reset()
    for _ in range(20):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.elevation <= ELEVATION_RANGE[1] + 1e-6


def test_azimuth_wraps_under_repeated_action():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=3)
    env.reset()
    for _ in range(20):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert 0.0 <= env.azimuth < 360.0


def test_episode_truncates_at_max_steps_and_never_terminates():
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=4)
    env.reset()
    truncated = False
    for i in range(MAX_STEPS):
        _, _, terminated, truncated, _ = env.step(np.array([0.1, 0.1], dtype=np.float32))
        assert terminated is False
        if i < MAX_STEPS - 1:
            assert truncated is False
    assert truncated is True


def test_reward_equals_the_actual_alignment_delta():
    # reward must always equal (alignment after the step) - (alignment before
    # the step), regardless of direction -- this is the dense reward's entire
    # definition, so pin it down directly rather than asserting a sign that
    # would depend on exactly where the target tissue happens to sit.
    volume = build_phantom(size=48)
    spacing = (1.0, 1.0, 1.0)
    env = CameraViewpointEnv(volume, spacing, seed=5)
    env.reset()
    env.target_tissue = "bone"
    env.target_direction = _tissue_centroid_direction(volume, spacing, "bone")
    env.azimuth = 0.0
    env.elevation = 0.0
    before_alignment = float(np.dot(_view_direction(0.0, 0.0), env.target_direction))
    _, reward, _, _, _ = env.step(np.array([1.0, 1.0], dtype=np.float32))
    after_alignment = float(np.dot(_view_direction(env.azimuth, env.elevation), env.target_direction))
    assert abs(reward - (after_alignment - before_alignment)) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_camera_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.camera_env'`

- [ ] **Step 3: Implement `rl/camera_env.py`**

```python
"""Gymnasium environment for training a continuous-control camera-viewpoint
agent against a centroid-alignment reward -- no VTK rendering, computed
directly on the loaded volume's HU array."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from transfer import TISSUE_BANDS

TISSUES = ["air", "fat", "soft", "spongy", "bone"]
MAX_DELTA_DEGREES = 30.0
MAX_STEPS = 20
ELEVATION_RANGE = (-85.0, 85.0)
OBS_SIZE = 2 + 1 + len(TISSUES) + 1  # sin/cos azimuth, elevation, tissue one-hot, alignment


def _view_direction(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])


def _tissue_centroid_direction(volume, spacing, tissue):
    """Unit vector from the volume's physical center to the tissue's voxel
    centroid, or None if no voxel falls in that tissue's HU band, or if the
    centroid coincides with the volume center (no meaningful direction)."""
    lo, hi = TISSUE_BANDS[tissue]
    mask = (volume >= lo) & (volume < hi)
    if not mask.any():
        return None
    idx = np.nonzero(mask)
    sx, sy, sz = spacing
    centroid = np.array([idx[0].mean() * sx, idx[1].mean() * sy, idx[2].mean() * sz])
    center = np.array(volume.shape) * np.array(spacing) / 2.0
    offset = centroid - center
    norm = np.linalg.norm(offset)
    if norm < 1e-6:
        return None
    return offset / norm


class CameraViewpointEnv(gym.Env):
    """One episode = one target tissue; the action moves the camera's
    azimuth/elevation to maximize alignment with that tissue's centroid
    direction from the volume center."""

    metadata = {"render_modes": []}

    def __init__(self, volume: np.ndarray, spacing: tuple, seed: int | None = None):
        super().__init__()
        self.volume = volume
        self.spacing = spacing
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.azimuth = 0.0
        self.elevation = 0.0
        self.target_tissue = None
        self.target_direction = None
        self.step_count = 0

    def _resolve_target(self):
        tissues = list(TISSUES)
        self._rng.shuffle(tissues)
        for tissue in tissues:
            direction = _tissue_centroid_direction(self.volume, self.spacing, tissue)
            if direction is not None:
                return tissue, direction
        return "bone", _tissue_centroid_direction(self.volume, self.spacing, "bone")

    def _alignment(self) -> float:
        cam_dir = _view_direction(self.azimuth, self.elevation)
        return float(np.dot(cam_dir, self.target_direction))

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(len(TISSUES), dtype=np.float32)
        onehot[TISSUES.index(self.target_tissue)] = 1.0
        az_rad = np.radians(self.azimuth)
        return np.concatenate([
            np.array([np.sin(az_rad), np.cos(az_rad), self.elevation / 85.0], dtype=np.float32),
            onehot,
            np.array([self._alignment()], dtype=np.float32),
        ])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.azimuth = float(self._rng.uniform(0.0, 360.0))
        self.elevation = float(self._rng.uniform(*ELEVATION_RANGE))
        self.target_tissue, self.target_direction = self._resolve_target()
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        before = self._alignment()
        d_az = float(np.clip(action[0], -1.0, 1.0)) * MAX_DELTA_DEGREES
        d_el = float(np.clip(action[1], -1.0, 1.0)) * MAX_DELTA_DEGREES
        self.azimuth = (self.azimuth + d_az) % 360.0
        self.elevation = float(np.clip(self.elevation + d_el, *ELEVATION_RANGE))
        after = self._alignment()
        reward = after - before
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        terminated = False
        info = {"alignment_after": after}
        return self._obs(), reward, terminated, truncated, info
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_camera_env.py -v`
Expected: all 10 tests pass.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 10 from the current baseline.

- [ ] **Step 6: Commit**

```bash
git add rl/camera_env.py tests/test_camera_env.py
git commit -m "Add CameraViewpointEnv: centroid-alignment reward for camera-viewpoint RL"
```

---

### Task 2: `rl/camera_train.py`

**Files:**
- Create: `rl/camera_train.py`

- [ ] **Step 1: Implement the training script**

```python
"""Train a SAC agent on CameraViewpointEnv against the centroid-alignment reward."""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as configure_logger

from datasets import load_dataset
from rl.camera_env import CameraViewpointEnv

MODEL_PATH = "out/rl_models/sac_camera_agent.zip"
LOG_DIR = "out/rl_camera_logs"


def train(total_timesteps: int, dataset: str = "ct_skull", n_envs: int = 4,
          seed: int = 0, model_path: str = MODEL_PATH) -> SAC:
    volume, spacing = load_dataset(dataset)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    run_dir = os.path.join(LOG_DIR, f"run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    vec_env = make_vec_env(lambda: CameraViewpointEnv(volume, spacing), n_envs=n_envs, seed=seed)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, seed=seed)
    model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the camera-viewpoint SAC agent.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--dataset", type=str, default="ct_skull")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    train(args.timesteps, dataset=args.dataset, n_envs=args.n_envs, seed=args.seed, model_path=args.out)


if __name__ == "__main__":
    main()
```

Note: `ct_skull` (`data/CT-brain.nrrd`) is already downloaded and
checksum-verified from earlier work in this repo — no network access is
needed to run this script.

- [ ] **Step 2: Smoke-run it with a tiny timestep budget**

Run: `.venv/bin/python -m rl.camera_train --timesteps 500 --n-envs 2`
Expected: exits 0, prints SB3 training progress, creates
`out/rl_models/sac_camera_agent.zip` and a non-empty
`out/rl_camera_logs/run_seed0/` directory (containing `progress.csv` and a
tensorboard event file, per the same logger setup already used and
verified in `rl/train.py`).

- [ ] **Step 3: Verify the saved model loads back**

```bash
.venv/bin/python -c "
from stable_baselines3 import SAC
model = SAC.load('out/rl_models/sac_camera_agent.zip')
print(model.policy)
"
```
Expected: prints the MLP policy architecture with no errors.

- [ ] **Step 4: Verify training-curve plotting works unmodified**

Run: `.venv/bin/python -m plots.training_curves --run-dir out/rl_camera_logs/run_seed0`
Expected: succeeds and writes `plots/output/training_curves.{pdf,png}`,
confirming the existing plotting tool needs zero changes for this new
agent (it only depends on SB3's own standard `progress.csv` column names).

- [ ] **Step 5: Run the full fast suite to confirm nothing broke**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (no new automated tests for this task; verified by the
manual run above, same convention as `rl/train.py`).

- [ ] **Step 6: Commit**

`out/` is gitignored — the smoke-run artifacts (model file, logs, plot
output) will not be staged. Verify with `git status` before committing.

```bash
git add rl/camera_train.py
git commit -m "Add camera-viewpoint SAC training CLI"
```

---

### Task 3: `rl/camera_eval.py`

**Files:**
- Create: `rl/camera_eval.py`
- Create: `tests/test_camera_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_eval.py`:

```python
"""Tests for the pure-logic pieces of rl/camera_eval.py."""
import numpy as np

from rl.camera_eval import _run_hill_climb


def test_hill_climb_alignment_never_decreases_step_to_step():
    rng = np.random.default_rng(0)
    target_direction = rng.normal(size=3)
    target_direction /= np.linalg.norm(target_direction)
    episode = {
        "azimuth": 10.0,
        "elevation": -20.0,
        "target_tissue": "bone",
        "target_direction": target_direction,
    }
    alignments = _run_hill_climb(episode)
    for before, after in zip(alignments, alignments[1:]):
        assert after >= before - 1e-9
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_camera_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.camera_eval'`

- [ ] **Step 3: Implement `rl/camera_eval.py`**

```python
"""Compare the trained camera-viewpoint SAC policy against a hill-climbing
baseline over the same held-out (tissue, start-angle) episodes."""
import json
import os

import numpy as np
from stable_baselines3 import SAC

from datasets import load_dataset
from rl.camera_env import ELEVATION_RANGE, MAX_DELTA_DEGREES, MAX_STEPS, CameraViewpointEnv, _view_direction
from rl.eval import _steps_to_90pct
from search import resize_step

MODEL_PATH = "out/rl_models/sac_camera_agent.zip"
RESULTS_PATH = "out/rl_camera_eval_results.json"
N_EPISODES = 20
EVAL_SEED = 12345


def _build_episodes(volume, spacing, n_episodes, seed):
    env = CameraViewpointEnv(volume, spacing, seed=seed)
    episodes = []
    for _ in range(n_episodes):
        env.reset()
        episodes.append({
            "azimuth": env.azimuth, "elevation": env.elevation,
            "target_tissue": env.target_tissue, "target_direction": env.target_direction,
        })
    return episodes


def _run_policy(model, volume, spacing, episode):
    env = CameraViewpointEnv(volume, spacing)
    env.azimuth = episode["azimuth"]
    env.elevation = episode["elevation"]
    env.target_tissue = episode["target_tissue"]
    env.target_direction = episode["target_direction"]
    env.step_count = 0
    alignments = [env._alignment()]
    obs = env._obs()
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        alignments.append(info["alignment_after"])
    return alignments


def _run_hill_climb(episode):
    azimuth, elevation = episode["azimuth"], episode["elevation"]
    target_direction = episode["target_direction"]

    def alignment(az, el):
        return float(np.dot(_view_direction(az, el), target_direction))

    step = MAX_DELTA_DEGREES
    alignments = [alignment(azimuth, elevation)]
    for _ in range(MAX_STEPS):
        current = alignments[-1]
        best, best_az, best_el = current, azimuth, elevation
        for d_az, d_el in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            cand_az = (azimuth + d_az) % 360.0
            cand_el = float(np.clip(elevation + d_el, *ELEVATION_RANGE))
            score = alignment(cand_az, cand_el)
            if score > best:
                best, best_az, best_el = score, cand_az, cand_el
        accepted = best > current
        if accepted:
            azimuth, elevation = best_az, best_el
        step = resize_step(step, accepted)
        alignments.append(alignment(azimuth, elevation))
    return alignments


def evaluate(model_path: str = MODEL_PATH, dataset: str = "ct_skull",
             n_episodes: int = N_EPISODES, seed: int = EVAL_SEED,
             results_path: str = RESULTS_PATH) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path!r} -- run `python -m rl.camera_train` first."
        )
    volume, spacing = load_dataset(dataset)
    model = SAC.load(model_path)
    episodes = _build_episodes(volume, spacing, n_episodes, seed)

    policy_finals, policy_steps = [], []
    hillclimb_finals, hillclimb_steps = [], []
    for ep in episodes:
        pf = _run_policy(model, volume, spacing, ep)
        hf = _run_hill_climb(ep)
        best = max(pf + hf)
        policy_finals.append(pf[-1])
        hillclimb_finals.append(hf[-1])
        policy_steps.append(_steps_to_90pct(pf, pf[0], best))
        hillclimb_steps.append(_steps_to_90pct(hf, hf[0], best))

    results = {
        "n_episodes": n_episodes,
        "dataset": dataset,
        "policy": {
            "mean_final_alignment": float(np.mean(policy_finals)),
            "mean_steps_to_90pct": float(np.mean(policy_steps)),
        },
        "hill_climb": {
            "mean_final_alignment": float(np.mean(hillclimb_finals)),
            "mean_steps_to_90pct": float(np.mean(hillclimb_steps)),
        },
    }
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_camera_eval.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run it against the smoke-trained model from Task 2**

Run: `.venv/bin/python -m rl.camera_eval`
Expected: exits 0, prints a JSON object with `policy`/`hill_climb`
sub-objects, writes `out/rl_camera_eval_results.json`. With only a
500-timestep smoke-trained model the policy's numbers may be mediocre —
the goal here is confirming the *pipeline* runs end to end, not good
numbers (the real comparison happens after Task 4's full training run).

- [ ] **Step 6: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 1.

- [ ] **Step 7: Commit**

```bash
git add rl/camera_eval.py tests/test_camera_eval.py
git commit -m "Add camera-viewpoint policy-vs-hill-climb evaluation CLI"
```

---

### Task 4: Full training run and real comparison

**Files:** none (execution only)

- [ ] **Step 1: Run full training**

Run: `.venv/bin/python -m rl.camera_train --timesteps 200000` (background
this if needed — on CPU-only PyTorch this small env/network pair should
complete well under an hour given there's no rendering cost, matching the
TF agent's ~7-minute training time for the same timestep budget).

- [ ] **Step 2: Run evaluation against the fully-trained model**

Run: `.venv/bin/python -m rl.camera_eval`
Expected: `out/rl_camera_eval_results.json` now reflects a properly trained
policy. Report both methods' `mean_final_alignment` and
`mean_steps_to_90pct` to the user.

- [ ] **Step 3: Report results, no commit**

`out/` is gitignored — this task produces a trained model and a results
file for the user to look at, not a code change. Summarize the comparison
numbers in chat, and note the reward's known limitation explicitly (no
occlusion modeling — this is a directional-alignment proxy, not a claim of
rendering-accurate visibility), consistent with how the TF agent's own
limitations were surfaced rather than glossed over.

---

## Notes for the implementer

- Do not modify `rl/env.py`, `rl/train.py`, `rl/eval.py`, `camera.py`,
  `commands.py`, `server.py`, or `mvp.py` — this plan adds a fully parallel
  set of files and touches nothing that already exists, except reading
  (never modifying) `search.py::resize_step` and `rl/eval.py::_steps_to_90pct`.
- `ct_skull` (`data/CT-brain.nrrd`) is already downloaded and
  checksum-verified in this repo from earlier work — no network access is
  required at any point in this plan.
- The centroid-alignment reward is an intentional, acknowledged proxy for
  "can you actually see this tissue" — it has no occlusion model. Do not
  attempt to add occlusion modeling as part of this plan; that was
  explicitly scoped out in the design spec as a separate, much larger
  effort (ray-marching visibility, considered and declined for this pass).
