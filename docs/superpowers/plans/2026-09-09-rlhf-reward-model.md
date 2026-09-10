# RLHF Reward Model for the Transfer-Function RL Policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Process note on Task 5:** the labeling step requires a real human typing
> answers to blocking `input()` prompts. A dispatched background subagent has
> no one to answer those prompts and will hang. Task 5 Step 1 MUST be run by
> the coordinator directly with the user present (e.g. the user runs it
> themselves via the `!` prefix in their own terminal), never delegated to a
> subagent. Steps 2+ of Task 5 (training/eval once real preference data
> exists) can be delegated normally.

**Goal:** Train a small reward model from ~50 human better/worse judgments of rendered transfer-function before/after pairs, then train an RL policy against that learned reward instead of the automatic `mass_fraction` metric — and actually run the full pipeline once for real.

**Architecture:** Four new modules layered bottom-up: `rl/reward_model.py` (pure PyTorch, no rendering — trainable/testable fast), `rl/reward_model_env.py` (a thin `TFEnv` subclass that renders once per step and asks the reward model for the reward, caching the previous render so it's one render/step not two), `rl/collect_preferences.py` (produces the labeled data the reward model trains on, reusing `evaluate.human()`'s existing console prompt), and `rl/reward_model_train.py` (mirrors `rl/online_train.py`'s chunked-`learn()`-plus-eval shape, logging both the learned reward and the true metric side by side, plus a final before/after render gallery).

**Tech Stack:** Python, PyTorch (already installed, pulled in by `stable-baselines3`), `stable-baselines3` (SAC), `gymnasium`, VTK (via the existing `render.py`), `pytest`.

**Full design:** `docs/superpowers/specs/2026-09-09-rlhf-reward-model-design.md`

---

## Task 1: `rl/reward_model.py`

**Files:**
- Create: `rl/reward_model.py`
- Test: `tests/test_rl_reward_model.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Fast tests for rl/reward_model.py -- no rendering, no real preference data."""
import json

import numpy as np
import pytest

from rl.reward_model import INPUT_DIM, featurize, load_reward_model, predict_reward, train_reward_model


def test_featurize_shape_and_values():
    before = {"mean": 10.0, "std": 5.0, "coverage": 0.2, "entropy": 1.0}
    after = {"mean": 15.0, "std": 5.0, "coverage": 0.3, "entropy": 1.2}
    x = featurize(before, after, "bone", "increase")
    assert x.shape == (INPUT_DIM,)
    assert x.dtype == np.float32
    assert x[0] == pytest.approx(5.0)
    assert x[1] == pytest.approx(0.0)
    assert x[2] == pytest.approx(0.1)
    assert x[3] == pytest.approx(0.2)
    # tissue one-hot order is rl.env.TISSUES = ["air", "fat", "soft", "spongy", "bone"]
    assert list(x[4:9]) == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert x[9] == 1.0


def test_featurize_decrease_direction_flag():
    before = {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0}
    after = {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0}
    x = featurize(before, after, "air", "decrease")
    assert x[9] == -1.0
    assert list(x[4:9]) == [1.0, 0.0, 0.0, 0.0, 0.0]


def _make_separable_preferences(path, n=40):
    """Trivially separable synthetic data: label=1 whenever mean went up,
    label=-1 whenever it went down. The reward model should learn this easily
    -- this is a correctness check on the training loop, not a claim about
    real preference data."""
    rng = np.random.default_rng(0)
    with open(path, "w") as f:
        for i in range(n):
            delta_mean = float(rng.uniform(1.0, 10.0)) * (1 if i % 2 == 0 else -1)
            before = {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            after = {"mean": 50.0 + delta_mean, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            record = {
                "timestamp": "2026-01-01T00:00:00", "rater_id": "default",
                "target_tissue": "bone", "direction": "increase", "delta": 0.1,
                "before_features": before, "after_features": after,
                "label": 1 if delta_mean > 0 else -1,
            }
            f.write(json.dumps(record) + "\n")


def test_train_reward_model_learns_separable_data(tmp_path):
    prefs_path = tmp_path / "preferences.jsonl"
    _make_separable_preferences(prefs_path)
    model_path = tmp_path / "reward_model.pt"

    model = train_reward_model(str(prefs_path), str(model_path), epochs=300, seed=0)

    assert model_path.exists()
    better = predict_reward(model, {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                             {"mean": 60.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                             "bone", "increase")
    worse = predict_reward(model, {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                            {"mean": 40.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                            "bone", "increase")
    assert better > worse


def test_load_reward_model_roundtrip(tmp_path):
    prefs_path = tmp_path / "preferences.jsonl"
    _make_separable_preferences(prefs_path)
    model_path = tmp_path / "reward_model.pt"
    train_reward_model(str(prefs_path), str(model_path), epochs=50, seed=0)

    loaded = load_reward_model(str(model_path))
    r = predict_reward(loaded, {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                        {"mean": 60.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0},
                        "bone", "increase")
    assert -1.0 <= r <= 1.0
```

Save this to `tests/test_rl_reward_model.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rl_reward_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.reward_model'`

- [ ] **Step 3: Write the implementation**

```python
"""Small reward model learned from human better/worse judgments of rendered
before/after transfer-function pairs -- used in place of the automatic
mass_fraction metric as the RL reward. See rl/reward_model_env.py for where
this gets used, and rl/collect_preferences.py for where the training data
comes from."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from rl.env import TISSUES

FEATURE_KEYS = ("mean", "std", "coverage", "entropy")
INPUT_DIM = len(FEATURE_KEYS) + len(TISSUES) + 1  # 10


def featurize(before_features: dict, after_features: dict, target_tissue: str, direction: str) -> np.ndarray:
    delta = [after_features[k] - before_features[k] for k in FEATURE_KEYS]
    onehot = [1.0 if t == target_tissue else 0.0 for t in TISSUES]
    direction_flag = 1.0 if direction == "increase" else -1.0
    return np.array(delta + onehot + [direction_flag], dtype=np.float32)


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reward_model(preferences_path: str, model_path: str, epochs: int = 200,
                        lr: float = 1e-2, seed: int = 0) -> RewardModel:
    with open(preferences_path) as f:
        records = [json.loads(line) for line in f]

    X = np.stack([
        featurize(r["before_features"], r["after_features"], r["target_tissue"], r["direction"])
        for r in records
    ])
    y = np.array([1.0 if r["label"] == 1 else 0.0 for r in records], dtype=np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 5)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    model = RewardModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = ((model(X_train) > 0).float() == y_train).float().mean().item()
        val_acc = ((model(X_val) > 0).float() == y_val).float().mean().item()
    print(f"[reward_model] train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
          f"(n_train={len(train_idx)}, n_val={len(val_idx)})")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model


def load_reward_model(model_path: str) -> RewardModel:
    model = RewardModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_reward(model: RewardModel, before_features: dict, after_features: dict,
                    target_tissue: str, direction: str) -> float:
    x = torch.from_numpy(featurize(before_features, after_features, target_tissue, direction)).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).item()
    return float(2.0 / (1.0 + np.exp(-logit)) - 1.0)


def main():
    parser = argparse.ArgumentParser(description="Train the RLHF reward model from logged preferences.")
    parser.add_argument("--preferences", type=str, default="out/rlhf_preferences.jsonl")
    parser.add_argument("--out", type=str, default="out/rl_models/reward_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_reward_model(args.preferences, args.out, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
```

Save this to `rl/reward_model.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rl_reward_model.py -v`
Expected: PASS (4 tests, fast — no rendering, pure PyTorch on tiny synthetic data)

- [ ] **Step 5: Commit**

```bash
git add rl/reward_model.py tests/test_rl_reward_model.py
git commit -m "Add rl/reward_model.py: small MLP reward model trained on human preferences"
```

---

## Task 2: `rl/reward_model_env.py`

**Files:**
- Create: `rl/reward_model_env.py`
- Test: `tests/test_rl_reward_model_env.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for rl/reward_model_env.py."""
import numpy as np
import pytest

import rl.reward_model_env as reward_model_env
from rl.reward_model_env import RewardModelTFEnv


class _StubRewardModel:
    pass  # predict_reward itself is monkeypatched in the fast test below, so
          # this never needs real behavior


def test_step_caches_previous_after_as_next_before(monkeypatch):
    render_calls = []

    def fake_render(volume, params, spacing, camera):
        render_calls.append(params.copy())
        return object()

    def fake_grab(win):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    feature_sequence = iter([
        {"mean": 0.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0},  # reset() seed
        {"mean": 1.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0},  # step 1 after
        {"mean": 2.0, "std": 0.0, "coverage": 0.0, "entropy": 0.0},  # step 2 after
    ])

    def fake_features(rgb):
        return next(feature_sequence)

    predict_calls = []

    def fake_predict_reward(model, before_features, after_features, target_tissue, direction):
        predict_calls.append((before_features, after_features))
        return 0.5

    monkeypatch.setattr(reward_model_env.render, "render", fake_render)
    monkeypatch.setattr(reward_model_env.render, "grab", fake_grab)
    monkeypatch.setattr(reward_model_env.render, "features", fake_features)
    monkeypatch.setattr(reward_model_env, "predict_reward", fake_predict_reward)

    env = RewardModelTFEnv(volume=np.zeros((4, 4, 4)), spacing=(1.0, 1.0, 1.0),
                            reward_model=_StubRewardModel(), seed=0)
    env.reset()
    env.step([0.3])
    env.step([0.3])

    assert len(render_calls) == 3  # 1 at reset (seeds cache) + 1 per step
    # step 2's "before" must equal step 1's "after" -- the cached value, not a fresh render
    assert predict_calls[1][0] == predict_calls[0][1]


@pytest.mark.slow
def test_real_render_and_reward_model_end_to_end(tmp_path):
    import json

    from datasets import load_dataset
    from rl.reward_model import train_reward_model

    prefs_path = tmp_path / "preferences.jsonl"
    with open(prefs_path, "w") as f:
        for i in range(10):
            before = {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            after = {"mean": 60.0 if i % 2 == 0 else 40.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            f.write(json.dumps({
                "timestamp": "2026-01-01T00:00:00", "rater_id": "default",
                "target_tissue": "bone", "direction": "increase", "delta": 0.1,
                "before_features": before, "after_features": after,
                "label": 1 if i % 2 == 0 else -1,
            }) + "\n")
    model = train_reward_model(str(prefs_path), str(tmp_path / "reward_model.pt"), epochs=20)

    volume, spacing = load_dataset("synthetic")
    env = RewardModelTFEnv(volume=volume, spacing=spacing, reward_model=model, seed=0)
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step([0.3])
    assert -1.0 <= reward <= 1.0
    assert "automatic_mass_fraction_reward" in info
```

Save this to `tests/test_rl_reward_model_env.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rl_reward_model_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.reward_model_env'`

- [ ] **Step 3: Write the implementation**

```python
"""TFEnv variant whose reward comes from a learned reward model over
rendered image features, instead of the automatic mass_fraction metric. See
rl/reward_model.py for the reward model itself, and
docs/superpowers/specs/2026-09-09-rlhf-reward-model-design.md for why."""
from camera import DEFAULT_CAMERA
from rl.env import TFEnv
from rl.reward_model import predict_reward

import render


class RewardModelTFEnv(TFEnv):
    """Reuses TFEnv's sampling/action/observation logic entirely unchanged
    (via super().reset()/super().step()) and only replaces the reward.
    One render per step, not two: each step's "after" features become next
    step's cached "before" features, seeded once at reset()."""

    def __init__(self, volume, spacing, reward_model, seed=None, camera=None):
        super().__init__(seed=seed)
        self.volume = volume
        self.spacing = spacing
        self.reward_model = reward_model
        self.camera = camera or dict(DEFAULT_CAMERA)
        self._prev_features = None

    def _render_features(self, params):
        win = render.render(self.volume, params, self.spacing, self.camera)
        return render.features(render.grab(win))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_features = self._render_features(self.params)
        return obs, info

    def step(self, action):
        obs, automatic_reward, terminated, truncated, info = super().step(action)
        after_features = self._render_features(self.params)
        reward = predict_reward(self.reward_model, self._prev_features, after_features,
                                 self.target_tissue, self.direction)
        self._prev_features = after_features
        info["automatic_mass_fraction_reward"] = automatic_reward
        return obs, reward, terminated, truncated, info
```

Save this to `rl/reward_model_env.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rl_reward_model_env.py -v -m "not slow"` (fast test only)
Expected: PASS (1 test, fast — mocked render, no VTK)

Then run: `.venv/bin/python -m pytest tests/test_rl_reward_model_env.py -v -m slow` (real render + real tiny reward model)
Expected: PASS (takes a few seconds — one real VTK render at reset + one at step)

- [ ] **Step 5: Commit**

```bash
git add rl/reward_model_env.py tests/test_rl_reward_model_env.py
git commit -m "Add rl/reward_model_env.py: TFEnv variant rewarded by the learned reward model"
```

---

## Task 3: `rl/collect_preferences.py`

**Files:**
- Create: `rl/collect_preferences.py`
- Test: `tests/test_rl_collect_preferences.py`

- [ ] **Step 1: Write the failing test**

```python
"""Fast test for rl/collect_preferences.py -- no real console input, no real VTK render."""
import json

import numpy as np

import rl.collect_preferences as collect_preferences


def test_collect_preferences_writes_expected_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_load_dataset(name):
        return np.zeros((4, 4, 4)), (1.0, 1.0, 1.0)

    def fake_render(volume, params, spacing, camera):
        return object()

    def fake_grab(win):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_features(rgb):
        return {"mean": 1.0, "std": 2.0, "coverage": 0.1, "entropy": 0.5}

    def fake_human(before_path, after_path):
        return 1

    monkeypatch.setattr(collect_preferences, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(collect_preferences.render, "render", fake_render)
    monkeypatch.setattr(collect_preferences.render, "grab", fake_grab)
    monkeypatch.setattr(collect_preferences.render, "features", fake_features)
    monkeypatch.setattr(collect_preferences, "human", fake_human)

    collect_preferences.collect_preferences(n_pairs=3, rater_id="tester", seed=0,
                                             out_path="out/rlhf_preferences.jsonl")

    with open("out/rlhf_preferences.jsonl") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 3
    for row in rows:
        assert row["rater_id"] == "tester"
        assert row["label"] == 1
        assert row["before_features"] == {"mean": 1.0, "std": 2.0, "coverage": 0.1, "entropy": 0.5}
        assert "target_tissue" in row
        assert "direction" in row
        assert "delta" in row
```

Save this to `tests/test_rl_collect_preferences.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rl_collect_preferences.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.collect_preferences'`

- [ ] **Step 3: Write the implementation**

```python
"""Collects human better/worse preference labels over rendered transfer-
function before/after pairs, for training the reward model in
rl/reward_model.py. Reuses the same base-case sampling rl/eval.py's offline
comparison uses, and the same console judging prompt the hill-climb search
flow already uses (evaluate.human) -- run this yourself, interactively, it
cannot be delegated to a background agent."""
import argparse
import datetime
import os

import numpy as np
from PIL import Image

from camera import DEFAULT_CAMERA
from datasets import load_dataset
from evaluate import human, jsonl_append
from rl.env import MAX_DELTA
from rl.eval import _build_episodes
from search import propose_step

import render

PREFERENCES_PATH = "out/rlhf_preferences.jsonl"
LABELING_DIR = "out/rlhf_labeling"


def collect_preferences(n_pairs: int = 50, rater_id: str = "default", seed: int = 0,
                         out_path: str = PREFERENCES_PATH) -> None:
    volume, spacing = load_dataset("synthetic")
    camera = dict(DEFAULT_CAMERA)
    cases = _build_episodes(n_pairs, seed)
    rng = np.random.default_rng(seed)

    os.makedirs(LABELING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for i, case in enumerate(cases):
        before_params = case["params"]
        action = float(rng.uniform(-1.0, 1.0))
        delta = action * MAX_DELTA
        after_params = propose_step(before_params, case["peak_idx"], sign=1.0, step=delta)

        before_win = render.render(volume, before_params, spacing, camera)
        before_rgb = render.grab(before_win)
        before_features = render.features(before_rgb)
        after_win = render.render(volume, after_params, spacing, camera)
        after_rgb = render.grab(after_win)
        after_features = render.features(after_rgb)

        before_path = os.path.join(LABELING_DIR, f"{i}_before.png")
        after_path = os.path.join(LABELING_DIR, f"{i}_after.png")
        Image.fromarray(before_rgb).save(before_path)
        Image.fromarray(after_rgb).save(after_path)

        print(f"[{i + 1}/{n_pairs}] target={case['target_tissue']} direction={case['direction']}")
        label = human(before_path, after_path)

        jsonl_append(out_path, {
            "timestamp": datetime.datetime.now().isoformat(),
            "rater_id": rater_id,
            "target_tissue": case["target_tissue"],
            "direction": case["direction"],
            "delta": delta,
            "before_features": before_features,
            "after_features": after_features,
            "label": label,
        })


def main():
    parser = argparse.ArgumentParser(description="Collect human preference labels for the reward model.")
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--rater-id", type=str, default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=PREFERENCES_PATH)
    args = parser.parse_args()
    collect_preferences(args.n_pairs, rater_id=args.rater_id, seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
```

Save this to `rl/collect_preferences.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rl_collect_preferences.py -v`
Expected: PASS (fast — render and human() are both mocked, no real VTK, no blocking console input)

- [ ] **Step 5: Commit**

```bash
git add rl/collect_preferences.py tests/test_rl_collect_preferences.py
git commit -m "Add rl/collect_preferences.py: interactive labeling script for the reward model"
```

---

## Task 4: `rl/reward_model_train.py`

**Files:**
- Create: `rl/reward_model_train.py`
- Test: `tests/test_rl_reward_model_train.py`

- [ ] **Step 1: Write the failing test**

```python
"""Marked slow: exercises a real (tiny) reward-model-driven SAC training
run, including real VTK rendering, to verify eval logging and the gallery
output."""
import json
import os

import pytest

from rl.reward_model import train_reward_model
from rl.reward_model_train import reward_model_train


def _make_tiny_reward_model(tmp_path):
    prefs_path = tmp_path / "preferences.jsonl"
    with open(prefs_path, "w") as f:
        for i in range(10):
            before = {"mean": 50.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            after = {"mean": 60.0 if i % 2 == 0 else 40.0, "std": 10.0, "coverage": 0.2, "entropy": 1.0}
            f.write(json.dumps({
                "timestamp": "2026-01-01T00:00:00", "rater_id": "default",
                "target_tissue": "bone", "direction": "increase", "delta": 0.1,
                "before_features": before, "after_features": after,
                "label": 1 if i % 2 == 0 else -1,
            }) + "\n")
    model_path = tmp_path / "reward_model.pt"
    train_reward_model(str(prefs_path), str(model_path), epochs=20)
    return str(model_path)


@pytest.mark.slow
def test_reward_model_train_writes_eval_log_and_gallery(tmp_path, monkeypatch):
    reward_model_path = _make_tiny_reward_model(tmp_path)
    monkeypatch.chdir(tmp_path)

    reward_model_train(total_timesteps=100, eval_interval=50, seed=42,
                        reward_model_path=reward_model_path,
                        model_path="out/rl_models/test_reward_model_agent.zip")

    run_dir = os.path.join("out", "rl_logs", "reward_model_run_seed42")
    eval_csv_path = os.path.join(run_dir, "eval_progress.csv")
    assert os.path.exists(eval_csv_path)
    with open(eval_csv_path, newline="") as f:
        rows = f.readlines()
    assert rows[0].strip() == "timesteps,mean_reward_model_score,mean_automatic_mass_fraction_reward"
    assert len(rows) == 3  # header + 2 checkpoints (100 / 50)

    gallery_dir = os.path.join(run_dir, "gallery")
    assert os.path.exists(os.path.join(gallery_dir, "0_before.png"))
    assert os.path.exists(os.path.join(gallery_dir, "0_after.png"))

    assert os.path.exists("out/rl_models/test_reward_model_agent.zip")
```

Save this to `tests/test_rl_reward_model_train.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rl_reward_model_train.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.reward_model_train'`

- [ ] **Step 3: Write the implementation**

```python
"""Trains a fresh SAC agent against the learned reward model
(rl/reward_model.py) via RewardModelTFEnv, instead of the automatic
mass_fraction metric -- mirrors rl/online_train.py's chunked-learn()-plus-
checkpoint-eval shape, at a much smaller scale since every step now renders
(see docs/superpowers/specs/2026-09-09-rlhf-reward-model-design.md)."""
import argparse
import csv
import math
import os

import numpy as np
from PIL import Image
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_logger

from camera import DEFAULT_CAMERA
from datasets import load_dataset
from rl.eval import _build_episodes
from rl.reward_model import load_reward_model
from rl.reward_model_env import RewardModelTFEnv

import render

MODEL_PATH = "out/rl_models/sac_tf_reward_model_agent.zip"
LOG_DIR = "out/rl_logs"
EVAL_N_EPISODES = 5
EVAL_SEED = 12345
N_GALLERY_EPISODES = 3
MAX_STEPS = 20


def _seed_episode(env, episode):
    env.params = episode["params"].copy()
    env.target_tissue = episode["target_tissue"]
    env.direction = episode["direction"]
    env.peak_idx = episode["peak_idx"]
    env.step_count = 0
    env._prev_features = env._render_features(env.params)
    return env._obs()


def _run_policy_with_logging(model, env, episode):
    obs = _seed_episode(env, episode)
    reward_scores, automatic_rewards = [], []
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_scores.append(reward)
        automatic_rewards.append(info["automatic_mass_fraction_reward"])
    return reward_scores, automatic_rewards


def reward_model_train(total_timesteps: int = 2000, eval_interval: int = 500, seed: int = 0,
                        reward_model_path: str = "out/rl_models/reward_model.pt",
                        model_path: str = MODEL_PATH) -> SAC:
    run_dir = os.path.join(LOG_DIR, f"reward_model_run_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    volume, spacing = load_dataset("synthetic")
    reward_model = load_reward_model(reward_model_path)
    env = RewardModelTFEnv(volume, spacing, reward_model, seed=seed)
    model = SAC("MlpPolicy", env, seed=seed)
    model.set_logger(configure_logger(run_dir, ["stdout", "csv", "tensorboard"]))

    eval_episodes = _build_episodes(EVAL_N_EPISODES, EVAL_SEED)
    eval_env = RewardModelTFEnv(volume, spacing, reward_model, seed=seed + 1)

    eval_log_path = os.path.join(run_dir, "eval_progress.csv")
    with open(eval_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["timesteps", "mean_reward_model_score", "mean_automatic_mass_fraction_reward"])

    n_chunks = math.ceil(total_timesteps / eval_interval)
    timesteps_done = 0
    for _ in range(n_chunks):
        chunk = min(eval_interval, total_timesteps - timesteps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        timesteps_done += chunk

        all_reward_scores, all_automatic_rewards = [], []
        for ep in eval_episodes:
            reward_scores, automatic_rewards = _run_policy_with_logging(model, eval_env, ep)
            all_reward_scores.extend(reward_scores)
            all_automatic_rewards.extend(automatic_rewards)

        row = [timesteps_done, float(np.mean(all_reward_scores)), float(np.mean(all_automatic_rewards))]
        with open(eval_log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[reward_model_train] timesteps={timesteps_done} "
              f"mean_reward_model_score={row[1]:.4f} mean_automatic_reward={row[2]:.4f}")

    gallery_dir = os.path.join(run_dir, "gallery")
    os.makedirs(gallery_dir, exist_ok=True)
    for i, ep in enumerate(eval_episodes[:N_GALLERY_EPISODES]):
        before_win = render.render(volume, ep["params"], spacing, dict(DEFAULT_CAMERA))
        Image.fromarray(render.grab(before_win)).save(os.path.join(gallery_dir, f"{i}_before.png"))

        obs = _seed_episode(eval_env, ep)
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = eval_env.step(action)
        after_win = render.render(volume, eval_env.params, spacing, dict(DEFAULT_CAMERA))
        Image.fromarray(render.grab(after_win)).save(os.path.join(gallery_dir, f"{i}_after.png"))

    model.save(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SAC against the learned RLHF reward model.")
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-model", type=str, default="out/rl_models/reward_model.pt")
    parser.add_argument("--out", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    reward_model_train(args.timesteps, eval_interval=args.eval_interval, seed=args.seed,
                        reward_model_path=args.reward_model, model_path=args.out)


if __name__ == "__main__":
    main()
```

Save this to `rl/reward_model_train.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rl_reward_model_train.py -v -m slow`
Expected: PASS (takes roughly 1-2 minutes — a real 100-step run with VTK rendering every step, plus gallery renders)

- [ ] **Step 5: Commit**

```bash
git add rl/reward_model_train.py tests/test_rl_reward_model_train.py
git commit -m "Add rl/reward_model_train.py: train SAC against the learned reward model"
```

---

## Task 5: Produce real results

No new code in this task except what emerges from running the pipeline for real.

**Files:** none created or modified in Steps 2+ (Step 1 only creates data/artifact files under `out/`, which is gitignored).

- [ ] **Step 1: Collect real preference labels — run this interactively with the user, not as a background subagent**

The coordinator tells the user to run this themselves (e.g. via the `!` prefix in their own terminal, so blocking `input()` prompts actually reach a real person):

```
python -m rl.collect_preferences --n-pairs 50 --rater-id <name>
```

This takes as long as the person takes to judge 50 pairs — there is no way to estimate or automate this. Expected when done: `out/rlhf_preferences.jsonl` has 50 lines, `out/rlhf_labeling/` has 100 PNGs (50 before + 50 after).

- [ ] **Step 2: Run the full pytest suite (fast tests only) to confirm nothing broke**

Run: `.venv/bin/python -m pytest -m "not slow"`
Expected: PASS, no new failures

- [ ] **Step 3: Train the reward model on the real preference data**

Run: `.venv/bin/python -m rl.reward_model --preferences out/rlhf_preferences.jsonl`
Expected: prints `[reward_model] train_acc=... val_acc=... (n_train=40, n_val=10)`. Report both numbers honestly regardless of value — this is a 50-label pilot, not a claim of a well-generalized model. `out/rl_models/reward_model.pt` exists afterward.

- [ ] **Step 4: Train SAC against the learned reward model**

Run: `.venv/bin/python -m rl.reward_model_train --timesteps 2000 --eval-interval 500`
Expected: prints 4 `[reward_model_train] timesteps=... mean_reward_model_score=... mean_automatic_reward=...` lines, takes roughly 4-6 minutes (real VTK render every step). Afterward: `out/rl_models/sac_tf_reward_model_agent.zip`, `out/rl_logs/reward_model_run_seed0/eval_progress.csv` (5 lines: header + 4 checkpoints), `out/rl_logs/reward_model_run_seed0/gallery/` (3 before/after PNG pairs).

- [ ] **Step 5: Report results — nothing to commit**

`out/` is gitignored (same as the previous phase), so there's nothing to `git add`. Report back:
- The full `eval_progress.csv` table (all checkpoints)
- The reward-model train/val accuracy from Step 3
- Whether `mean_automatic_mass_fraction_reward` trends upward alongside `mean_reward_model_score` (does optimizing the learned reward also move the true metric sensibly, or diverge from it — either answer is a real, reportable finding)
- Send the 3 gallery before/after PNG pairs to the user directly so they can judge the actual visual result, not just the numbers

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers "New module: rl/reward_model.py" + its testing section. Task 2 covers "New module: rl/reward_model_env.py" (including the verified subclass-reuse approach and the caching behavior) + its testing section. Task 3 covers "New module: rl/collect_preferences.py". Task 4 covers "Training + evaluation" (chunked loop, dual-signal logging, qualitative gallery). Task 5 covers the full "Data flow" end-to-end and delivers the actual pilot results the design exists to produce.
- **Process risk called out explicitly:** Task 5 Step 1 cannot be delegated to a background subagent (no one to answer blocking `input()` prompts) — flagged both in the plan header and inline at that step, unlike the previous plan where every step was subagent-safe.
- **Type/signature consistency:** `RewardModelTFEnv(volume, spacing, reward_model, seed=None, camera=None)` is constructed identically in Task 2's tests, Task 4's `reward_model_train`, and Task 4's test. `predict_reward(model, before_features, after_features, target_tissue, direction)` signature matches between Task 1's implementation, Task 2's usage, and both tasks' tests. `featurize`'s 10-dim output order (Δfeatures, tissue one-hot, direction flag) is defined once in Task 1 and never redefined elsewhere.
- **No placeholders:** all steps show complete, runnable code; no TBD/TODO.
- **Scope check:** four small new files (one per responsibility: reward model, reward-driven env, labeling, training/eval) plus one real end-to-end run — no existing file is modified, matching the spec's explicit non-goals (no `rl/env.py`, `rl/eval.py`, `server.py`, or any previous-phase file touched).
