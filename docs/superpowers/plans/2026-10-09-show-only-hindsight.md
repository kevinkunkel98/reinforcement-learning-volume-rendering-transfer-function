# `show_only` Hindsight Demonstrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the RL training loop continuous, in-training-time hindsight
demonstrations shaped like `show_only` instructions (one explicit target per
canonical class: 1-2 "shown" classes positive, every other class negative),
mirroring the existing `--hindsight-brightness` mechanism.

**Architecture:** `OneShotEnv._sample_hindsight_show_only()` constructs an
action that pushes 1-2 random classes' height parameter toward a randomized
high value and every other class's height toward a randomized low value,
applies it, measures the *real* resulting visibility change for every
class, and uses that measurement as the goal -- same "construct, apply,
measure, that's the oracle" pattern `_sample_hindsight_brightness` already
uses, generalized from one controllable group to all eight. Gated by a new
`hindsight_show_only_share`, wired through the CLI and pretrain-collection
contract exactly where `hindsight_brightness_share` already is.

**Tech Stack:** Python, NumPy, Gymnasium, pytest.

Reference: `docs/superpowers/specs/2026-10-09-show-only-hindsight-design.md`

---

### Task 1: `OneShotEnv._sample_hindsight_show_only`

**Files:**
- Modify: `rl/oneshot_env.py:83` (new constants, after the brightness
  constants), `rl/oneshot_env.py:191-229` (new constructor param +
  validation), `rl/oneshot_env.py:378-407` (new method + wiring into
  `_sample_hindsight_goal`)
- Test: `tests/test_oneshot_env.py`

- [x] **Step 1: Write the failing tests**

Add to `tests/test_oneshot_env.py`, right after
`test_brightness_share_zero_keeps_visibility_hindsight` (ends at line 411,
immediately before `test_hindsight_ratio_zero_keeps_sampling_instructions`):

```python
def test_hindsight_show_only_targets_every_class_with_correct_signs():
    # "synthetic" can't be used here: it uses visibility.py's intensity-only
    # fallback, under which heart/vessels/liver/kidneys/spleen read exactly
    # 0.0 visibility regardless of any TF parameter (the fallback can't
    # identify them), so a test requiring all 8 classes to get a meaningful
    # target could never pass against it. fake_a + _StubModel (already used
    # by ~20 other tests in this file) has full 8-class coverage instead.
    env = OneShotEnv(["fake_a"], model_for_volume=lambda name: _StubModel(),
                     hindsight_ratio=1.0, hindsight_show_only_share=1.0)
    for seed in range(5):
        env.reset(seed=seed)
        targets = env._instruction["targets"]
        assert set(targets) == set(goals.GOAL_CLASSES)
        assert all(set(t) == {"vis"} for t in targets.values())
        shown = {c for c, t in targets.items() if t["vis"] > 0}
        hidden = {c for c, t in targets.items() if t["vis"] < 0}
        assert 1 <= len(shown) <= 2
        assert shown | hidden == set(goals.GOAL_CLASSES)


def test_hindsight_show_only_action_matches_measured_deltas():
    # Same invariant as test_hindsight_goal_is_exactly_what_the_target_achieved,
    # generalized to every class instead of just the mentioned ones: the
    # goal this hindsight kind hands the policy is exactly what the action
    # that produced it actually measured, for every single class.
    env = OneShotEnv(["fake_a"], model_for_volume=lambda name: _StubModel(),
                     hindsight_ratio=1.0, hindsight_show_only_share=1.0)
    for seed in range(5):
        env.reset(seed=seed)
        targets = env._instruction["targets"]
        target_agg = goals.aggregate(env._model.features(
            apply_controllable(env._start_params, env.hindsight_action())))
        for goal_class in goals.GOAL_CLASSES:
            measured = (math.log10(target_agg["vis"][goal_class] + goals.EPSILON)
                        - math.log10(env._start_agg["vis"][goal_class] + goals.EPSILON))
            assert targets[goal_class]["vis"] == pytest.approx(measured, abs=1e-9)


def test_show_only_share_zero_keeps_visibility_hindsight():
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    for seed in range(5):
        env.reset(seed=seed)
        # The generic sampler only ever mentions 1-2 classes; show_only
        # mentions all 8. share=0.0 (the default) must never produce the
        # show_only shape.
        assert len(env._instruction["targets"]) <= 2
```

- [x] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py::test_hindsight_show_only_targets_every_class_with_correct_signs tests/test_oneshot_env.py::test_hindsight_show_only_action_matches_measured_deltas tests/test_oneshot_env.py::test_show_only_share_zero_keeps_visibility_hindsight -v`
Expected: FAIL with `TypeError: OneShotEnv.__init__() got an unexpected keyword argument 'hindsight_show_only_share'`

- [x] **Step 3: Read the current code**

Re-read `rl/oneshot_env.py:78-85` (brightness constants),
`rl/oneshot_env.py:191-229` (constructor), and `rl/oneshot_env.py:378-407`
(`_sample_hindsight_brightness` and `_sample_hindsight_goal`) for the exact
current text -- the `Edit` tool requires an exact match, and this plan's
line numbers may have drifted slightly since this plan was written.

- [x] **Step 4: Add the show-only hindsight constants, in `rl/oneshot_env.py`**

Replace:

```python
HINDSIGHT_BRIGHTNESS_NOISE = 0.8
HINDSIGHT_BRIGHTNESS_MIN_DELTA = 0.05
# Brightness is only measured on visible tissue (visibility.features reports
# 0 below 1e-4), so a class needs this much visibility to be demonstrable.
HINDSIGHT_BRIGHTNESS_MIN_VIS = 1e-3
```

with:

```python
HINDSIGHT_BRIGHTNESS_NOISE = 0.8
HINDSIGHT_BRIGHTNESS_MIN_DELTA = 0.05
# Brightness is only measured on visible tissue (visibility.features reports
# 0 below 1e-4), so a class needs this much visibility to be demonstrable.
HINDSIGHT_BRIGHTNESS_MIN_VIS = 1e-3

# Show-only hindsight: push 1-2 classes' height *up from wherever it
# already is* and every other class's height *down from wherever it
# already is* -- a relative nudge, mirroring brightness's own pattern, not
# an absolute snap to a fixed region. An absolute region can't work here:
# transfer.ANATOMICAL_HEIGHTS ranges from 0.05 (lungs) to 0.60 (skeleton),
# so any single fixed "low" region sits above most classes' own starting
# height, which would push a "hidden" class brighter instead of dimmer.
# Randomized magnitude, not a fixed snap to one offset, since overlapping
# anatomy means a literal full separation isn't always reachable or
# realistic.
HINDSIGHT_SHOW_ONLY_PUSH_RANGE = (0.4, 1.2)
HINDSIGHT_SHOW_ONLY_MIN_DELTA = 0.05
```

- [x] **Step 5: Add the constructor param and validation, in `rl/oneshot_env.py`**

Replace:

```python
    def __init__(self, volume_ids, model_for_volume=visibility.for_volume,
                 hindsight_ratio: float = 0.0, policy_version: str = POLICY_VERSION,
                 action_mode: str = "absolute", reward_mode: str = "attainment",
                 balance_classes: bool = False, hindsight_brightness_share: float = 0.0):
```

with:

```python
    def __init__(self, volume_ids, model_for_volume=visibility.for_volume,
                 hindsight_ratio: float = 0.0, policy_version: str = POLICY_VERSION,
                 action_mode: str = "absolute", reward_mode: str = "attainment",
                 balance_classes: bool = False, hindsight_brightness_share: float = 0.0,
                 hindsight_show_only_share: float = 0.0):
```

Then replace:

```python
        if not 0.0 <= hindsight_brightness_share <= 1.0:
            raise ValueError("hindsight_brightness_share must be between 0 and 1")
        # Share of hindsight episodes whose goal is a brightness change
        # (`_sample_hindsight_brightness`) rather than a visibility change.
        self.hindsight_brightness_share = float(hindsight_brightness_share)
```

with:

```python
        if not 0.0 <= hindsight_brightness_share <= 1.0:
            raise ValueError("hindsight_brightness_share must be between 0 and 1")
        if not 0.0 <= hindsight_show_only_share <= 1.0:
            raise ValueError("hindsight_show_only_share must be between 0 and 1")
        # Share of hindsight episodes whose goal is a brightness change
        # (`_sample_hindsight_brightness`) rather than a visibility change.
        self.hindsight_brightness_share = float(hindsight_brightness_share)
        # Share of hindsight episodes whose goal is a show-only change
        # (`_sample_hindsight_show_only`) rather than a visibility change.
        self.hindsight_show_only_share = float(hindsight_show_only_share)
```

- [x] **Step 6: Add `_sample_hindsight_show_only` and wire it in, in `rl/oneshot_env.py`**

Replace:

```python
    def _sample_hindsight_goal(self, rng, model, start_params, start_agg):
        """(instruction-shaped dict, action) from a reachable target."""
        if self.hindsight_brightness_share and rng.random() < self.hindsight_brightness_share:
            sampled = self._sample_hindsight_brightness(rng, start_params, start_agg)
            if sampled is not None:
                return sampled
        start_controllable = np.asarray(self._controllable_values(start_params))
```

with:

```python
    def _sample_hindsight_show_only(self, rng, start_params, start_agg):
        """(instruction-shaped dict, action) for a show-only goal: 1-2
        classes pushed toward high visibility, every other class pushed
        toward low -- the one hindsight shape with an explicit target on
        every class, matching goals._sample_show_only's real-instruction
        shape."""
        n = min(int(rng.integers(1, 3)), len(goals.GOAL_CLASSES))
        start_controllable = np.asarray(self._controllable_values(start_params))
        for _ in range(self.HINDSIGHT_MAX_TRIES):
            shown = set(rng.choice(goals.GOAL_CLASSES, size=n, replace=False))
            hidden = set(goals.GOAL_CLASSES) - shown
            action = start_controllable.copy()
            for goal_class in goals.GOAL_CLASSES:
                base = goals.PEAK_INDEX[goal_class] * transfer.PARAMS_PER_PEAK
                group = CONTROLLABLE.index((base + 2,))
                sign = 1.0 if goal_class in shown else -1.0
                push = rng.uniform(*HINDSIGHT_SHOW_ONLY_PUSH_RANGE)
                action[group] = np.clip(action[group] + sign * push, -1.0, 1.0)
            target_agg = goals.aggregate(self._features(apply_controllable(start_params, action)),
                                         self._active_layers)
            deltas = {c: math.log10(target_agg["vis"][c] + goals.EPSILON)
                         - math.log10(start_agg["vis"][c] + goals.EPSILON)
                      for c in goals.GOAL_CLASSES}
            shown_ok = min(deltas[c] for c in shown) >= HINDSIGHT_SHOW_ONLY_MIN_DELTA
            hidden_ok = max(deltas[c] for c in hidden) <= -HINDSIGHT_SHOW_ONLY_MIN_DELTA
            if shown_ok and hidden_ok:
                targets = {c: {"vis": deltas[c]} for c in goals.GOAL_CLASSES}
                return ({"kind": "hindsight", "text": None, "targets": targets,
                         "goal": goals.goal_vector(targets)}, action)
        return None

    def _sample_hindsight_goal(self, rng, model, start_params, start_agg):
        """(instruction-shaped dict, action) from a reachable target."""
        if self.hindsight_brightness_share and rng.random() < self.hindsight_brightness_share:
            sampled = self._sample_hindsight_brightness(rng, start_params, start_agg)
            if sampled is not None:
                return sampled
        if self.hindsight_show_only_share and rng.random() < self.hindsight_show_only_share:
            sampled = self._sample_hindsight_show_only(rng, start_params, start_agg)
            if sampled is not None:
                return sampled
        start_controllable = np.asarray(self._controllable_values(start_params))
```

- [x] **Step 7: Run the tests to confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py -v -k "show_only or hindsight or brightness"`
Expected: PASS, all hindsight/brightness tests and the three new show-only tests.

- [x] **Step 8: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS. Note: `tests/test_pretrain.py::test_collect_is_deterministic_and_independent_of_workers` (and its search-collection sibling) have known, pre-existing, full-suite-only flakiness from torch intra-op threading nondeterminism, unrelated to this change (see
`docs/experiments/2026-10-07-search-distilled-pretraining-results.md`). If
ONLY those fail with a ~1e-7-scale floating point mismatch, rerun in
isolation to confirm they pass alone and treat that as the known flake, not
a bug. Any other failure is real and must be fixed.

- [x] **Step 9: Commit**

```bash
git add rl/oneshot_env.py tests/test_oneshot_env.py
git commit -m "feat(rl): add show_only-shaped hindsight demonstrations"
```

---

### Task 2: Wire `--hindsight-show-only` through the CLI and pretrain collection

**Files:**
- Modify: `rl/oneshot_train.py:196-324` (new CLI flag, pretrain-collection
  contract, live-training `env_kwargs`)
- Test: `tests/test_oneshot_train.py`

- [x] **Step 1: Write the failing test**

Add to `tests/test_oneshot_train.py`, right after
`test_parse_args_accepts_v7_short_experiment_options` (ends at line 229,
immediately before `test_parse_args_accepts_v8_layer_policy_options`):

```python
def test_parse_args_accepts_hindsight_show_only_option():
    args = oneshot_train.parse_args([])
    assert args.hindsight_show_only == pytest.approx(0.0)

    args = oneshot_train.parse_args(["--hindsight-show-only", "0.1"])
    assert args.hindsight_show_only == pytest.approx(0.1)
```

- [x] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_oneshot_train.py::test_parse_args_accepts_hindsight_show_only_option -v`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'hindsight_show_only'`

- [x] **Step 3: Read the current code**

Re-read `rl/oneshot_train.py:196-324` for the exact current text.

- [x] **Step 4: Add the CLI flag, in `rl/oneshot_train.py`**

Replace:

```python
    parser.add_argument("--hindsight-brightness", type=float, default=0.0,
                        help="share of hindsight episodes whose goal is a brightness change")
```

with:

```python
    parser.add_argument("--hindsight-brightness", type=float, default=0.0,
                        help="share of hindsight episodes whose goal is a brightness change")
    parser.add_argument("--hindsight-show-only", type=float, default=0.0,
                        help="share of hindsight episodes whose goal is a show-only change")
```

- [x] **Step 5: Thread it through pretrain collection, in `rl/oneshot_train.py`**

Replace:

```python
    expected = {**observation_metadata(**contract), "episodes": args.pretrain_episodes,
                "seed": args.seed, "split": "train",
                "hindsight_brightness_share": args.hindsight_brightness}
    if os.path.exists(args.pretrain_data):
        data, stored = pretrain.load_dataset(args.pretrain_data)
        # Datasets collected before brightness hindsight existed had none.
        stored.setdefault("hindsight_brightness_share", 0.0)
        mismatched = {key for key in expected if stored.get(key) != expected[key]}
        if mismatched:
            raise ValueError(f"{args.pretrain_data} does not match this run: {sorted(mismatched)}")
    else:
        data = pretrain.collect_hindsight(datasets.volumes_for_split("train"),
                                          {**contract, "hindsight_brightness_share": args.hindsight_brightness},
                                          args.pretrain_episodes, args.seed, args.pretrain_workers)
        pretrain.save_dataset(data, args.pretrain_data, expected)
```

with:

```python
    expected = {**observation_metadata(**contract), "episodes": args.pretrain_episodes,
                "seed": args.seed, "split": "train",
                "hindsight_brightness_share": args.hindsight_brightness,
                "hindsight_show_only_share": args.hindsight_show_only}
    if os.path.exists(args.pretrain_data):
        data, stored = pretrain.load_dataset(args.pretrain_data)
        # Datasets collected before brightness/show-only hindsight existed had none.
        stored.setdefault("hindsight_brightness_share", 0.0)
        stored.setdefault("hindsight_show_only_share", 0.0)
        mismatched = {key for key in expected if stored.get(key) != expected[key]}
        if mismatched:
            raise ValueError(f"{args.pretrain_data} does not match this run: {sorted(mismatched)}")
    else:
        data = pretrain.collect_hindsight(datasets.volumes_for_split("train"),
                                          {**contract, "hindsight_brightness_share": args.hindsight_brightness,
                                           "hindsight_show_only_share": args.hindsight_show_only},
                                          args.pretrain_episodes, args.seed, args.pretrain_workers)
        pretrain.save_dataset(data, args.pretrain_data, expected)
```

- [x] **Step 6: Thread it through live-training `env_kwargs`, in `rl/oneshot_train.py`**

Replace:

```python
                          env_kwargs={"policy_version": args.policy_version,
                                      "action_mode": args.action_mode,
                                      "hindsight_ratio": args.hindsight_ratio,
                                      "balance_classes": args.balance_classes,
                                      "hindsight_brightness_share": args.hindsight_brightness,
                                      "reward_mode": args.reward_mode},
```

with:

```python
                          env_kwargs={"policy_version": args.policy_version,
                                      "action_mode": args.action_mode,
                                      "hindsight_ratio": args.hindsight_ratio,
                                      "balance_classes": args.balance_classes,
                                      "hindsight_brightness_share": args.hindsight_brightness,
                                      "hindsight_show_only_share": args.hindsight_show_only,
                                      "reward_mode": args.reward_mode},
```

- [x] **Step 7: Run the test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/test_oneshot_train.py -v -k "hindsight or parse_args"`
Expected: PASS, all existing parse_args/hindsight tests and the new one.

- [x] **Step 8: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS (same known-flake caveat as Task 1 Step 8).

- [x] **Step 9: Commit**

```bash
git add rl/oneshot_train.py tests/test_oneshot_train.py
git commit -m "feat(rl): wire --hindsight-show-only through training and pretrain collection"
```

---

### Task 3: Record the change in the training-setup reference doc

**Files:**
- Modify: `docs/rl-training-setup.typ`

- [x] **Step 1: Add the new flag to the "Hindsight episodes" section**

Read `docs/rl-training-setup.typ`'s `== Hindsight episodes` section. After
its existing sentence ending "...since brightness instructions are
otherwise the weakest-supervised kind.", add:

```typst
A further share (`--hindsight-show-only`) make that demonstration a
show-only change: 1--2 classes pushed toward high visibility, every other
canonical class toward low, mirroring `goals._sample_show_only`'s
real-instruction shape -- the only hindsight kind with an explicit target
on every class (`OneShotEnv._sample_hindsight_show_only`).
```

- [x] **Step 2: Compile to confirm no syntax errors**

Run: `typst compile docs/rl-training-setup.typ /tmp/rl-training-setup-check.pdf`
Expected: exits 0, no errors. Then `rm /tmp/rl-training-setup-check.pdf`.

- [x] **Step 3: Commit**

```bash
git add docs/rl-training-setup.typ
git commit -m "docs: record --hindsight-show-only in the training-setup note"
```
