# Preference-Aligned Transfer Functions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a transfer-function policy in two stages — hindsight-goal episodes with no human involvement, then a preference finetune anchored to that policy — and measure whether human raters prefer the finetuned one.

**Architecture:** Tasks 1–2 operate on the preference data already being collected and need no training; they are useful before the 2026-09-22 MVP. Task 3 changes how training episodes are generated (`rl/oneshot_env.py`). Task 4 fits a Bradley-Terry reward model over measured features. Task 5 uses it to rerank policy proposals. Task 6 finetunes the policy against it with an anchor to the stage-1 policy.

**Tech Stack:** Python 3.14, numpy, torch (already a dependency via `visibility.py`), gymnasium, stable-baselines3, pytest. No sklearn — it is not installed and must not be added.

**Spec:** `docs/superpowers/specs/2026-09-17-preference-aligned-tf-design.md`

---

### Task 1: Flag assisted preference rows

Rows rated on 2026-09-17 while Claude commented on the pairs are not independent human judgments. They stay in the file, but must be excluded from any evaluation set.

**Files:**
- Create: `tools/flag_assisted_rows.py`
- Test: `tests/test_flag_assisted_rows.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flag_assisted_rows.py
from tools.flag_assisted_rows import flag_rows


def _row(timestamp, **extra):
    row = {"pair_id": "p", "timestamp": timestamp, "choice": "a"}
    row.update(extra)
    return row


def test_rows_inside_the_window_are_flagged():
    rows = [_row("2026-09-17T13:20:00")]
    assert flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")[0]["assisted"] is True


def test_rows_outside_the_window_are_flagged_false_not_dropped():
    rows = [_row("2026-09-16T09:00:00"), _row("2026-09-17T18:00:00")]
    flagged = flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")
    assert [r["assisted"] for r in flagged] == [False, False]
    assert len(flagged) == 2


def test_an_existing_true_flag_is_never_cleared():
    # Re-running the tool must not un-flag rows flagged by an earlier run with
    # a different window.
    rows = [_row("2026-09-15T09:00:00", assisted=True)]
    assert flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")[0]["assisted"] is True


def test_flagging_is_idempotent():
    rows = [_row("2026-09-17T13:20:00")]
    once = flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")
    twice = flag_rows(once, "2026-09-17T13:00:00", "2026-09-17T16:00:00")
    assert once == twice
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_flag_assisted_rows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.flag_assisted_rows'`

- [ ] **Step 3: Write the implementation**

```python
# tools/flag_assisted_rows.py
"""Mark preference rows that were rated with Claude commenting on the pair.

Those judgments are not independent: the rater saw an argument for one
candidate before choosing. They stay in the dataset -- they are still a
rater's answers -- but anything used to *evaluate* the reward model or a
policy must exclude them, or the evaluation measures the assistant as much as
the rater.
"""
import argparse
import datetime
import json
import os
import tempfile

PREF_PATH = "out/vis_preferences.jsonl"


def _parse(stamp: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(stamp)


def flag_rows(rows: list, start: str, end: str) -> list:
    """Every row gets an explicit `assisted` bool; an existing True is kept."""
    lo, hi = _parse(start), _parse(end)
    flagged = []
    for row in rows:
        row = dict(row)
        inside = lo <= _parse(row["timestamp"]) <= hi
        row["assisted"] = bool(row.get("assisted", False) or inside)
        flagged.append(row)
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=PREF_PATH)
    ap.add_argument("--start", required=True, help="ISO timestamp, inclusive")
    ap.add_argument("--end", required=True, help="ISO timestamp, inclusive")
    args = ap.parse_args()

    with open(args.path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    flagged = flag_rows(rows, args.start, args.end)

    # Written through a temporary file in the same directory: a half-written
    # preference file would lose judgments that cannot be re-collected.
    directory = os.path.dirname(os.path.abspath(args.path))
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as tmp:
        for row in flagged:
            tmp.write(json.dumps(row) + "\n")
        temporary = tmp.name
    os.replace(temporary, args.path)
    print(f"{sum(r['assisted'] for r in flagged)}/{len(flagged)} rows flagged assisted")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_flag_assisted_rows.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Back up the real data, then flag it**

```bash
cp out/vis_preferences.jsonl out/vis_preferences.jsonl.bak
.venv/bin/python -m tools.flag_assisted_rows --start 2026-09-17T13:00:00 --end 2026-09-17T16:30:00
```

Expected: a line like `28/54 rows flagged assisted`. Confirm the count is
plausible before continuing; if it flags everything, the window is wrong.

- [ ] **Step 6: Commit**

```bash
git add tools/flag_assisted_rows.py tests/test_flag_assisted_rows.py out/vis_preferences.jsonl
git commit -m "feat(prefs): flag rows rated with assistance so evaluation can exclude them"
```

---

### Task 2: Metric-versus-human agreement

The first question the preference data answers needs no reward model and no training: does `visibility.py` prefer what the rater prefers? Every row already carries `objective_choice` (the metric's pick) beside `choice` (the rater's).

**Files:**
- Create: `tools/preference_agreement.py`
- Test: `tests/test_preference_agreement.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preference_agreement.py
from tools.preference_agreement import agreement


def _row(choice, objective_choice, assisted=False, kind="relative"):
    return {
        "choice": choice,
        "objective_choice": objective_choice,
        "assisted": assisted,
        "instruction": {"kind": kind, "text": "more bone"},
    }


def test_agreement_counts_only_decided_unassisted_rows():
    rows = [
        _row("a", "a"),              # agree
        _row("b", "a"),              # disagree
        _row("skip", "a"),           # dropped: no judgment
        _row("equal", "a"),          # dropped from the rate, counted separately
        _row("a", "a", assisted=True),  # dropped: not independent
    ]
    result = agreement(rows)
    assert result["n"] == 2
    assert result["agree"] == 1
    assert result["rate"] == 0.5
    assert result["ties"] == 1
    assert result["skipped"] == 1
    assert result["assisted_excluded"] == 1


def test_confidence_interval_brackets_the_rate_and_stays_in_bounds():
    rows = [_row("a", "a") for _ in range(20)]
    result = agreement(rows)
    lo, hi = result["ci95"]
    assert 0.0 <= lo <= result["rate"] <= hi <= 1.0
    assert lo > 0.5  # 20/20 agreement is not consistent with a coin flip


def test_per_kind_breakdown_splits_by_instruction_kind():
    rows = [_row("a", "a", kind="relative"), _row("b", "a", kind="show_only")]
    per_kind = agreement(rows)["per_kind"]
    assert per_kind["relative"]["rate"] == 1.0
    assert per_kind["show_only"]["rate"] == 0.0


def test_empty_input_reports_no_rate_instead_of_dividing_by_zero():
    result = agreement([])
    assert result["n"] == 0
    assert result["rate"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_preference_agreement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.preference_agreement'`

- [ ] **Step 3: Write the implementation**

```python
# tools/preference_agreement.py
"""How often the visibility metric picks the candidate the rater picked.

This is the check the whole preference dataset exists to make: the policy is
trained and scored on `visibility.py`, which is a proxy for "does this render
answer the instruction". If raters disagree with it often, the proxy is the
problem, and that is the motivation for a preference-trained reward. No model
is fitted here -- every row already carries the metric's own pick.
"""
import argparse
import collections
import json
import math

PREF_PATH = "out/vis_preferences.jsonl"
DECIDED = ("a", "b")


def _wilson(agree: int, n: int) -> tuple:
    """Wilson score interval: behaves at 0/n and n/n, where the textbook
    normal approximation gives a zero-width interval."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    p = agree / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _summary(rows: list) -> dict:
    decided = [r for r in rows if r["choice"] in DECIDED]
    agree = sum(r["choice"] == r["objective_choice"] for r in decided)
    n = len(decided)
    return {
        "n": n,
        "agree": agree,
        "rate": (agree / n) if n else None,
        "ci95": _wilson(agree, n),
    }


def agreement(rows: list) -> dict:
    assisted_excluded = sum(bool(r.get("assisted")) for r in rows)
    usable = [r for r in rows if not r.get("assisted")]

    result = _summary(usable)
    result["ties"] = sum(r["choice"] == "equal" for r in usable)
    result["skipped"] = sum(r["choice"] == "skip" for r in usable)
    result["assisted_excluded"] = assisted_excluded

    by_kind = collections.defaultdict(list)
    for row in usable:
        by_kind[row["instruction"]["kind"]].append(row)
    result["per_kind"] = {kind: _summary(kind_rows) for kind, kind_rows in sorted(by_kind.items())}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=PREF_PATH)
    args = ap.parse_args()
    with open(args.path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = agreement(rows)

    rate = "n/a" if result["rate"] is None else f"{result['rate']:.3f}"
    lo, hi = result["ci95"]
    print(f"metric-vs-human agreement: {rate} (95% CI {lo:.3f}-{hi:.3f}) on n={result['n']}")
    print(f"  ties={result['ties']} skipped={result['skipped']} "
          f"assisted excluded={result['assisted_excluded']}")
    for kind, summary in result["per_kind"].items():
        kind_rate = "n/a" if summary["rate"] is None else f"{summary['rate']:.3f}"
        print(f"  {kind:<12} {kind_rate} on n={summary['n']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_preference_agreement.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Run it on the real data**

Run: `.venv/bin/python -m tools.preference_agreement`
Expected: a rate with a confidence interval. With fewer than ~30 unassisted
decided rows the interval will be too wide to conclude anything — that is the
expected state until more rating happens, and the number should not be quoted
in the thesis until the interval excludes 0.5.

- [ ] **Step 6: Commit**

```bash
git add tools/preference_agreement.py tests/test_preference_agreement.py
git commit -m "feat(prefs): metric-vs-human agreement with a Wilson interval"
```

---

### Task 3: Hindsight-goal episodes

`OneShotEnv.reset()` samples an instruction and hopes the volume can answer it. Sampling a reachable *target* instead makes every episode answerable by construction.

**Files:**
- Modify: `rl/oneshot_env.py` (`__init__` signature, `reset`)
- Test: `tests/test_oneshot_env.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_oneshot_env.py
import numpy as np

from rl.baselines import CONTROLLABLE, apply_controllable
from rl.oneshot_env import OneShotEnv


def test_hindsight_episodes_are_solvable_by_the_action_that_made_them():
    # The goal is derived from a target the action space can reach, so the
    # action that produced the target must score near-perfect attainment.
    # Anything less means the goal encoding and the reward disagree.
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    env.reset(seed=0)
    oracle = env.hindsight_action()
    _obs, _reward, _done, _truncated, info = env.step(oracle)
    assert info["attainment"] > 0.8


def test_hindsight_ratio_zero_keeps_sampling_instructions():
    env = OneShotEnv(["synthetic"], hindsight_ratio=0.0)
    _obs, info = env.reset(seed=0)
    assert info["goal_source"] == "instruction"
    assert env.hindsight_action() is None


def test_hindsight_episodes_report_their_source():
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    _obs, info = env.reset(seed=0)
    assert info["goal_source"] == "hindsight"


def test_hindsight_goal_mentions_at_least_one_class():
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    env.reset(seed=1)
    goal = env._instruction["goal"]
    mentioned = goal[4:8]  # the m[4] block of goals.goal_vector
    assert mentioned.sum() >= 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py -k hindsight -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'hindsight_ratio'`

- [ ] **Step 3: Implement the sampler**

In `rl/oneshot_env.py`, add to the imports:

```python
from rl.baselines import CONTROLLABLE, apply_controllable
```

Change `__init__` to accept the ratio and remember it:

```python
    def __init__(self, volume_ids, model_for_volume=visibility.for_volume,
                  hindsight_ratio: float = 0.0):
        # hindsight_ratio is the share of episodes whose goal is derived from a
        # sampled *target* transfer function rather than from a sampled
        # instruction. A hindsight episode is answerable by construction -- the
        # target demonstrates a solution -- which an instruction-sampled
        # episode is not: "a bit more lungs" on a pelvis scan asks for a change
        # of about 0.01% of the pixels.
        self.hindsight_ratio = float(hindsight_ratio)
        self._hindsight_action = None
```

Add these two methods to the class:

```python
    # A hindsight goal mentions one or two classes, matching the sparsity of
    # sampled instructions (a real instruction never asks for all four).
    HINDSIGHT_MAX_MENTIONS = 2

    def hindsight_action(self):
        """The action that produced this episode's target, or None when the
        episode came from an instruction. Test and distillation hook."""
        return None if self._hindsight_action is None else self._hindsight_action.copy()

    def _sample_hindsight_goal(self, rng, model, start_params, start_agg):
        """(instruction-shaped dict, action) from a reachable target."""
        action = rng.uniform(-1.0, 1.0, size=len(CONTROLLABLE))
        target_params = apply_controllable(start_params, action)
        target_agg = goals.aggregate(model.features(target_params))

        deltas = {}
        for goal_class in goals.GOAL_CLASSES:
            start_vis = start_agg["vis"][goal_class]
            target_vis = target_agg["vis"][goal_class]
            # log10, the same units goals.goal_vector and goals.distance use:
            # visibility spans three orders of magnitude, so a fixed absolute
            # change means very different things at each end.
            deltas[goal_class] = math.log10(target_vis + goals.EPSILON) \
                - math.log10(start_vis + goals.EPSILON)

        ranked = sorted(goals.GOAL_CLASSES, key=lambda c: -abs(deltas[c]))
        count = int(rng.integers(1, self.HINDSIGHT_MAX_MENTIONS + 1))
        targets = {c: {"vis": deltas[c]} for c in ranked[:count]}

        instruction = {
            "kind": "hindsight",
            "text": None,
            "targets": targets,
            "goal": goals.goal_vector(targets),
        }
        return instruction, action
```

In `reset`, replace the instruction line:

```python
        if rng.random() < self.hindsight_ratio:
            instruction, self._hindsight_action = self._sample_hindsight_goal(
                rng, model, start_params, start_agg)
        else:
            instruction = goals.sample_instruction(volume, model, start_agg, rng)
            self._hindsight_action = None
```

And in `_info`, add the source so training runs can report attainment per source:

```python
            "goal_source": "hindsight" if self._hindsight_action is not None else "instruction",
```

Add `import math` at the top of the file if it is not already imported.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py -v`
Expected: PASS, including the four new tests and every pre-existing one.

- [ ] **Step 5: Commit**

```bash
git add rl/oneshot_env.py tests/test_oneshot_env.py
git commit -m "feat(rl): hindsight-goal episodes, answerable by construction"
```

---

### Task 4: Bradley-Terry reward model

**Files:**
- Create: `reward_model.py`
- Test: `tests/test_reward_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward_model.py
import numpy as np
import pytest

from reward_model import RewardModel, pair_features


def _agg(skeleton=0.01, lungs=0.01, soft=0.01, vessels=0.001, bright=0.5):
    return {
        "vis": {"skeleton": skeleton, "lungs": lungs, "soft": soft, "vessels": vessels},
        "bright": {"skeleton": bright, "lungs": bright, "soft": bright, "vessels": bright},
        "coverage": 0.5,
    }


def _more_lungs_goal():
    goal = np.zeros(16)
    goal[1] = 0.3   # d[lungs]
    goal[5] = 1.0   # m[lungs]
    return goal


def _pair(better_lungs, worse_lungs, choice):
    return {
        "choice": choice,
        "assisted": False,
        "instruction": {"kind": "relative", "text": "more lungs", "goal": _more_lungs_goal().tolist()},
        "features": {
            "start": _agg(),
            "a": _agg(lungs=better_lungs),
            "b": _agg(lungs=worse_lungs),
        },
    }


def test_pair_features_is_antisymmetric_between_the_two_candidates():
    row = _pair(0.05, 0.01, "a")
    forward = pair_features(row, "a", "b")
    backward = pair_features(row, "b", "a")
    assert np.allclose(forward, -backward)


def test_model_learns_that_more_of_the_requested_class_is_preferred():
    rows = [_pair(0.05, 0.01, "a") for _ in range(20)] + [_pair(0.01, 0.05, "b") for _ in range(20)]
    model = RewardModel().fit(rows, epochs=300)
    goal = _more_lungs_goal()
    better = model.score(goal, _agg(), _agg(lungs=0.05))
    worse = model.score(goal, _agg(), _agg(lungs=0.01))
    assert better > worse


def test_ties_train_toward_no_preference():
    rows = [_pair(0.05, 0.01, "equal") for _ in range(40)]
    model = RewardModel().fit(rows, epochs=300)
    goal = _more_lungs_goal()
    gap = abs(model.score(goal, _agg(), _agg(lungs=0.05))
               - model.score(goal, _agg(), _agg(lungs=0.01)))
    assert gap < 0.5


def test_skipped_and_assisted_rows_are_not_trained_on():
    skipped = _pair(0.05, 0.01, "skip")
    assisted = _pair(0.05, 0.01, "a")
    assisted["assisted"] = True
    with pytest.raises(ValueError, match="no usable preference rows"):
        RewardModel().fit([skipped, assisted])


def test_accuracy_reports_held_out_agreement():
    rows = [_pair(0.05, 0.01, "a") for _ in range(20)]
    model = RewardModel().fit(rows, epochs=300)
    assert model.accuracy(rows) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reward_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reward_model'`

- [ ] **Step 3: Write the implementation**

```python
# reward_model.py
"""A Bradley-Terry preference model over measured visibility features.

The policy is trained on `goals.distance`, a proxy for "does this render
answer the instruction". This model is fitted to what raters actually chose,
so it can serve as the reward in a finetuning stage that the proxy cannot
express.

Inputs are the ~25 numbers already computed per candidate: the goal vector,
the per-class log10 visibility change, the per-class brightness change, and
the coverage change. Not image embeddings: a few hundred pairs cannot fit a
768-dimensional encoder, and a linear model over named features can be read
off afterwards -- which classes a rater rewards is itself a result.

Bradley-Terry: P(a preferred over b) = sigmoid(w . (phi(a) - phi(b))).
"""
import json
import math

import numpy as np
import torch

import goals

PREF_PATH = "out/vis_preferences.jsonl"
DECIDED = ("a", "b")


def candidate_features(goal: np.ndarray, start_agg: dict, candidate_agg: dict) -> np.ndarray:
    """phi: goal (16) + per-class log10 visibility change (4) + per-class
    brightness change (4) + coverage change (1) = 25 values."""
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    vis_change, bright_change = [], []
    for goal_class in goals.GOAL_CLASSES:
        start_vis = start_agg["vis"][goal_class]
        end_vis = candidate_agg["vis"][goal_class]
        vis_change.append(math.log10(end_vis + goals.EPSILON) - math.log10(start_vis + goals.EPSILON))
        bright_change.append(candidate_agg["bright"][goal_class] - start_agg["bright"][goal_class])
    coverage_change = candidate_agg["coverage"] - start_agg["coverage"]
    return np.concatenate([goal, vis_change, bright_change, [coverage_change]])


def pair_features(row: dict, first: str, second: str) -> np.ndarray:
    """phi(first) - phi(second) for one collected pair."""
    goal = np.asarray(row["instruction"]["goal"], dtype=np.float64)
    start = row["features"]["start"]
    return (candidate_features(goal, start, row["features"][first])
            - candidate_features(goal, start, row["features"][second]))


class RewardModel:
    """Linear Bradley-Terry scorer. `fit` trains, `score` evaluates."""

    def __init__(self):
        self.weights = None

    def _usable(self, rows: list) -> list:
        return [r for r in rows
                if not r.get("assisted") and r["choice"] in DECIDED + ("equal",)]

    def fit(self, rows: list, epochs: int = 400, learning_rate: float = 0.05):
        usable = self._usable(rows)
        if not usable:
            raise ValueError("no usable preference rows -- every row was skipped or assisted")

        # A tie contributes both orderings at half weight, which pulls the
        # score difference toward zero instead of inventing a winner.
        features, labels, weights = [], [], []
        for row in usable:
            difference = pair_features(row, "a", "b")
            if row["choice"] == "a":
                features.append(difference); labels.append(1.0); weights.append(1.0)
            elif row["choice"] == "b":
                features.append(difference); labels.append(0.0); weights.append(1.0)
            else:
                features.append(difference); labels.append(1.0); weights.append(0.5)
                features.append(difference); labels.append(0.0); weights.append(0.5)

        x = torch.tensor(np.asarray(features), dtype=torch.float64)
        y = torch.tensor(labels, dtype=torch.float64)
        sample_weight = torch.tensor(weights, dtype=torch.float64)

        w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=sample_weight)
        for _ in range(epochs):
            optimizer.zero_grad()
            # L2 keeps a separable, tiny dataset from driving weights to
            # infinity -- with 300 pairs and 25 features that is the default
            # outcome, not an edge case.
            loss = loss_fn(x @ w, y) + 1e-3 * (w * w).sum()
            loss.backward()
            optimizer.step()

        self.weights = w.detach().numpy()
        return self

    def score(self, goal, start_agg: dict, candidate_agg: dict) -> float:
        if self.weights is None:
            raise ValueError("model is not fitted")
        return float(candidate_features(goal, start_agg, candidate_agg) @ self.weights)

    def accuracy(self, rows: list) -> float:
        """Share of decided pairs whose rater choice the model reproduces."""
        decided = [r for r in self._usable(rows) if r["choice"] in DECIDED]
        if not decided:
            raise ValueError("no decided rows to score")
        correct = 0
        for row in decided:
            predicted = "a" if pair_features(row, "a", "b") @ self.weights > 0 else "b"
            correct += predicted == row["choice"]
        return correct / len(decided)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"weights": self.weights.tolist()}, f)

    @classmethod
    def load(cls, path: str):
        model = cls()
        with open(path) as f:
            model.weights = np.asarray(json.load(f)["weights"], dtype=np.float64)
        return model
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reward_model.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add reward_model.py tests/test_reward_model.py
git commit -m "feat(reward): Bradley-Terry preference model over measured features"
```

---

### Task 5: Split-by-instruction fit and the reranking arm

Splitting by row would let the same instruction appear in both fit and evaluation, which inflates accuracy. Reranking is the baseline that asks whether finetuning is needed at all.

**Files:**
- Create: `tools/fit_reward_model.py`
- Test: `tests/test_fit_reward_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit_reward_model.py
import numpy as np

from tools.fit_reward_model import rerank, split_by_instruction


def _row(text, choice="a"):
    return {"choice": choice, "assisted": False,
            "instruction": {"kind": "relative", "text": text, "goal": [0.0] * 16}}


def test_split_puts_every_row_of_one_instruction_on_the_same_side():
    rows = [_row("more bone") for _ in range(6)] + [_row("less lungs") for _ in range(6)]
    fit, held_out = split_by_instruction(rows, held_out_fraction=0.5, seed=0)
    fit_texts = {r["instruction"]["text"] for r in fit}
    held_texts = {r["instruction"]["text"] for r in held_out}
    assert fit_texts.isdisjoint(held_texts)
    assert len(fit) + len(held_out) == 12


def test_rerank_picks_the_candidate_the_model_scores_highest():
    class StubModel:
        def score(self, goal, start_agg, candidate_agg):
            return candidate_agg["vis"]["lungs"]

    start = {"vis": {"lungs": 0.01}, "bright": {"lungs": 0.5}, "coverage": 0.5}
    candidates = [
        (np.zeros(24), {"vis": {"lungs": 0.02}, "bright": {"lungs": 0.5}, "coverage": 0.5}),
        (np.ones(24), {"vis": {"lungs": 0.09}, "bright": {"lungs": 0.5}, "coverage": 0.5}),
    ]
    best_params, best_index = rerank(StubModel(), np.zeros(16), start, candidates)
    assert best_index == 1
    assert np.allclose(best_params, np.ones(24))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fit_reward_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.fit_reward_model'`

- [ ] **Step 3: Write the implementation**

```python
# tools/fit_reward_model.py
"""Fit the preference model with an honest split, and rerank with it.

Splitting by row would put the same instruction on both sides: the collector
samples a small vocabulary, so "a bit less soft tissue" appears many times,
and a model that memorised it would score well on rows it effectively trained
on. The split is by instruction text instead.
"""
import argparse
import json

import numpy as np

from reward_model import PREF_PATH, RewardModel


def split_by_instruction(rows: list, held_out_fraction: float = 0.3, seed: int = 0):
    texts = sorted({row["instruction"]["text"] for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(texts)
    held_out_count = max(1, int(round(len(texts) * held_out_fraction)))
    held_out_texts = set(texts[:held_out_count])
    held_out = [r for r in rows if r["instruction"]["text"] in held_out_texts]
    fit = [r for r in rows if r["instruction"]["text"] not in held_out_texts]
    return fit, held_out


def rerank(model, goal, start_agg: dict, candidates: list):
    """(params, index) of the candidate the model scores highest.

    `candidates` is a list of (params, aggregate) pairs, as produced by
    sampling the policy K times and measuring each proposal."""
    scores = [model.score(goal, start_agg, aggregate) for _params, aggregate in candidates]
    best_index = int(np.argmax(scores))
    return candidates[best_index][0], best_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=PREF_PATH)
    ap.add_argument("--out", default="out/reward_model.json")
    ap.add_argument("--held-out-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    usable = [r for r in rows if not r.get("assisted") and r["choice"] in ("a", "b", "equal")]
    fit_rows, held_out_rows = split_by_instruction(usable, args.held_out_fraction, args.seed)

    model = RewardModel().fit(fit_rows)
    model.save(args.out)
    print(f"fitted on {len(fit_rows)} rows from "
          f"{len({r['instruction']['text'] for r in fit_rows})} instructions")
    print(f"train accuracy   {model.accuracy(fit_rows):.3f}")
    try:
        print(f"held-out accuracy {model.accuracy(held_out_rows):.3f} on {len(held_out_rows)} rows")
    except ValueError:
        print("held-out accuracy n/a -- no decided rows in the held-out split")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fit_reward_model.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Commit**

```bash
git add tools/fit_reward_model.py tests/test_fit_reward_model.py
git commit -m "feat(reward): instruction-level split and reranking helper"
```

---

### Task 6: Stage-2 finetune with an anchor to the stage-1 policy

A few hundred judgments from one non-expert rater must be able to tilt the policy, never rebuild it. The anchor is a penalty on moving away from the stage-1 policy's action.

**Files:**
- Create: `rl/preference_env.py`
- Test: `tests/test_preference_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preference_env.py
import numpy as np

from rl.preference_env import PreferenceEnv


class StubReward:
    """Scores a candidate by how much lung visibility it produced."""

    def score(self, goal, start_agg, candidate_agg):
        return float(candidate_agg["vis"]["lungs"])


def test_reward_comes_from_the_preference_model_not_attainment():
    env = PreferenceEnv(["synthetic"], reward_model=StubReward(),
                         anchor_policy=None, anchor_beta=0.0, hindsight_ratio=1.0)
    env.reset(seed=0)
    _obs, reward, _done, _truncated, info = env.step(np.zeros(len(env.action_space.low)))
    assert reward == info["preference_score"]


def test_anchor_penalises_actions_far_from_the_stage_one_policy():
    class StubPolicy:
        def predict(self, observation, deterministic=True):
            return np.zeros(12), None

    env = PreferenceEnv(["synthetic"], reward_model=StubReward(),
                         anchor_policy=StubPolicy(), anchor_beta=1.0, hindsight_ratio=1.0)
    env.reset(seed=0)
    near = env.step(np.zeros(12))[1]
    env.reset(seed=0)
    far = env.step(np.ones(12))[1]
    assert near > far


def test_anchor_beta_zero_disables_the_penalty():
    class StubPolicy:
        def predict(self, observation, deterministic=True):
            return np.zeros(12), None

    env = PreferenceEnv(["synthetic"], reward_model=StubReward(),
                         anchor_policy=StubPolicy(), anchor_beta=0.0, hindsight_ratio=1.0)
    env.reset(seed=0)
    _obs, reward, _done, _truncated, info = env.step(np.ones(12))
    assert reward == info["preference_score"]
    assert info["anchor_penalty"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preference_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl.preference_env'`

- [ ] **Step 3: Write the implementation**

```python
# rl/preference_env.py
"""Stage-2 environment: the preference model's score is the reward.

Same observation, same action space and the same episode generation as
`OneShotEnv` -- only the reward changes, plus a penalty for straying from the
stage-1 policy. That penalty is what makes a few hundred judgments from one
rater safe to train on: they adjust a competent policy instead of replacing
it.

The penalty is measured in action space (squared distance to the stage-1
policy's deterministic action), not as a true KL between stochastic policies.
It is the same idea -- stay near the reference -- and it needs nothing from
the SB3 internals, which keeps this environment independent of the algorithm
used to train it. Report it as an action-space anchor, not as KL.
"""
import numpy as np

import goals
from rl.oneshot_env import OneShotEnv


class PreferenceEnv(OneShotEnv):
    def __init__(self, volume_ids, reward_model, anchor_policy=None, anchor_beta: float = 0.1,
                  **kwargs):
        super().__init__(volume_ids, **kwargs)
        self.reward_model = reward_model
        self.anchor_policy = anchor_policy
        self.anchor_beta = float(anchor_beta)
        self._observation = None

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._observation = observation
        return observation, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        observation, _attainment_reward, done, truncated, info = super().step(action)

        final_agg = goals.aggregate(self._model.features(self._params))
        score = float(self.reward_model.score(
            self._instruction["goal"], self._start_agg, final_agg))

        penalty = 0.0
        if self.anchor_policy is not None and self.anchor_beta > 0.0:
            reference, _state = self.anchor_policy.predict(self._observation, deterministic=True)
            penalty = self.anchor_beta * float(np.mean((action - np.asarray(reference)) ** 2))

        info["preference_score"] = score
        info["anchor_penalty"] = penalty
        self._observation = observation
        return observation, score - penalty, done, truncated, info
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preference_env.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: every test passes, including the 549 that existed before this plan.

- [ ] **Step 6: Commit**

```bash
git add rl/preference_env.py tests/test_preference_env.py
git commit -m "feat(rl): stage-2 preference environment with an anchor to stage 1"
```

---

## What this plan does not cover

Deliberately out of scope, each needing its own plan once the pieces above exist:

- **Training runs.** Stage-1 training with `hindsight_ratio` set, longer than the 150k steps the v3 seeds stopped at, and the stage-2 finetune with a beta sweep. These are cluster jobs, not code changes.
- **The blind A/B.** Serving π₁ against π₂ through the existing collector and reporting a win rate with a binomial interval.
- **The differentiable visibility baseline.** Porting `transfer_tables` to torch so attainment has an analytic gradient. It re-baselines the whole results table and deserves separate treatment.
- **Per-source attainment reporting** in the training loop, which the spec requires as the distribution-shift check. It belongs with the training-run plan, using the `goal_source` field Task 3 adds.
