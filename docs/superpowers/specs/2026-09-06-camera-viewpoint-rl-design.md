# Camera-Viewpoint RL — Design

## Purpose

Sub-project #2 of the VR roadmap (sub-project #1, camera-as-controllable-
state, is done and merged). This trains an SAC agent to find a good camera
viewpoint for viewing a named tissue, using a cheap, exact, non-neural
reward computed directly from the volume's HU array — no VTK rendering, no
CLIP/MLLM judge, consistent with this project's reward philosophy
throughout (`opacity_mass` for the TF agent, this project's centroid-
alignment metric for the camera agent).

## Grounding in existing code and data

- `camera.py`'s `DEFAULT_CAMERA`/`apply_camera_command` are for **discrete**
  voice commands (rotate/tilt/zoom + strength word). This RL environment
  does **not** go through that grammar — it manipulates continuous azimuth/
  elevation deltas directly, the same way `rl/env.py`'s `TFEnv` bypasses
  `commands.py`'s discrete grammar and manipulates the TF vector directly.
- `phantom.py::build_phantom()` (the synthetic phantom used for the
  existing TF-RL work) was inspected directly: `soft`/`fat`/`air` are all
  built as roughly spherically-symmetric blobs centered on the volume's
  true geometric center (`phantom.py:31-38`), while only `spongy`/`bone`
  are offset to `(center*1.15, center*0.9, center)` (`phantom.py:40-47`).
  This means a centroid-alignment reward is **degenerate** (no defined
  direction) for 3 of 5 tissues on the synthetic phantom — verified by
  inspection, not assumed. Per the agreed decision, this environment trains
  against a **real CT dataset** instead (default `ct_skull`), where every
  tissue has genuine anatomical asymmetry.
- `datasets.py::load_dataset(name) -> (volume, spacing)` already provides
  checksum-verified real datasets; `render.py` already establishes the
  convention that `volume.shape` is `(dx, dy, dz)` matching `spacing`
  axis-for-axis (confirmed via `render.py`'s `SetDimensions(dx, dy, dz)` +
  `SetSpacing(*spacing)` + Fortran-order flattening, which only works
  correctly on real non-cubic CT volumes because the axis order already
  matches `spacing`'s order) — this environment reuses that exact pairing,
  so it inherits axis-order correctness for free rather than re-deriving it.
- `transfer.py::TISSUE_BANDS` (HU ranges per tissue) is dataset-agnostic
  and reused directly for voxel masking.
- `search.py::resize_step(step, accepted)` is pure, generic step-size
  math with zero TF-specific logic — reused directly for this
  environment's 2D hill-climbing baseline.
- `rl/train.py`'s custom-logger pattern (CSV + tensorboard, per-seed run
  directory) is reused verbatim for the camera trainer — meaning
  `plots/training_curves.py` (built for the TF agent) works on camera-RL
  training runs with **zero changes**, since it only depends on the
  progress.csv column names SB3 itself produces (`rollout/ep_rew_mean`,
  `train/actor_loss`, etc.), not anything TF-specific.
- `rl/eval.py::_steps_to_90pct(fractions, start, best)` is fully generic
  (no TF-specific assumptions) — reused directly rather than reimplemented.

## New module: `rl/camera_env.py`

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
        # bone should exist with genuine off-center structure in any real CT dataset
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

Unlike the TF agent's reward (which has a direction concept — increase or
decrease), alignment has no direction: the goal is always "maximize it."
This makes the reward and the eventual eval comparison simpler than the TF
agent's (no increase/decrease branching needed).

## `rl/camera_train.py`

Mirrors `rl/train.py` exactly, including the per-seed CSV+tensorboard
logger:

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

`load_dataset(dataset)` is called once, outside the vec-env factory closure
— every parallel env shares the same read-only volume array (never
mutated), avoiding redundant loads/downloads.

## `rl/camera_eval.py`

Compares the trained policy against a 2D hill-climbing baseline (coordinate
search over azimuth/elevation, reusing `search.resize_step` for step-size
adaptation — a direct, standard generalization of the TF agent's 1D hill
climb to two degrees of freedom):

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
        best = max(pf + hf)  # alignment always wants to be maximized -- no direction concept
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

## Testing

- `tests/test_camera_env.py`: `_view_direction` produces unit vectors and
  the expected reference direction at azimuth=0/elevation=0;
  `_tissue_centroid_direction` returns `None` for a synthetic phantom's
  `soft`/`fat`/`air` (degenerate, centroid at volume center — this directly
  encodes the finding that motivated training on real data instead) and a
  genuine unit vector for `bone`/`spongy`; `CameraViewpointEnv.reset()`
  never selects a tissue with an undefined direction (loop over many seeds
  against the synthetic phantom, where this is actually exercisable,
  asserting `target_tissue` is always `bone` or `spongy`); `step()` moves
  azimuth/elevation in the correct direction and clamps/wraps correctly
  (reusing the same style of test as `camera.py`'s own wraparound tests).
- `tests/test_camera_eval.py`: `_run_hill_climb` never decreases alignment
  from one step to the next (`resize_step`'s accept/reject logic should
  guarantee monotonic improvement, the same invariant the TF hill-climb
  baseline has).
- No automated test trains a real SAC model against a real downloaded CT
  dataset (too slow for the test suite, and would require network access
  in CI-like environments) — `rl/camera_train.py`/`rl/camera_eval.py` are
  verified by a manual full run, the same way `rl/train.py`/`rl/eval.py`
  were.

## Explicitly out of scope

- Any change to the existing TF-opacity RL (`rl/env.py`, `rl/train.py`,
  `rl/eval.py`) — this is a fully separate, parallel environment.
- Zoom/distance as part of the action space (per the earlier decision) —
  angle-only for this first pass.
- Wiring the trained camera policy into the live chat UI or into
  `camera.py`'s discrete command grammar — this stays an offline
  training/evaluation exercise, exactly like the TF agent's current status.
- Occlusion modeling of any kind — the centroid-alignment reward is an
  explicit, acknowledged proxy, not a claim of rendering-accurate
  visibility. This should be named as a limitation in the thesis writeup,
  the same way the TF agent's reward/acceptance-criterion asymmetry was.
