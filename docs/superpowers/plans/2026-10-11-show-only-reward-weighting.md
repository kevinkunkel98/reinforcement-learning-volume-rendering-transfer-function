# Show-Only Group-Balanced Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the measured `show_only` reward asymmetry (hidden-group classes outweigh the shown-group 3-7x in `goals.distance` just by having more terms) by giving the shown and hidden groups equal total weight, applied consistently everywhere a `show_only` instruction is scored.

**Architecture:** `goals.distance`/`attainment`/`scoring` gain an optional `weights: dict` parameter (default `None`, zero behavior change for every existing caller). A new `goals.show_only_weights(shown, classes)` helper computes `{class: 1/group_size}` for the shown and hidden groups. The three places that build a `show_only`-shaped instruction (`goals._sample_show_only`, `goals._goal_show_only`, `OneShotEnv._sample_hindsight_show_only`) attach the result as a new `"weights"` key on the instruction dict. Every scoring call site across the codebase threads `instruction.get("weights")` through.

**Tech Stack:** Python, pytest, numpy. No new dependencies.

**Reference:** `docs/superpowers/specs/2026-10-10-show-only-reward-weighting-design.md` (approved design).

---

### Task 1: `goals.distance`/`attainment`/`scoring` accept an optional `weights` parameter

**Files:**
- Modify: `goals.py:176-217` (`distance`, `scoring`, `attainment`)
- Test: `tests/test_goals.py`

- [x] **Step 1: Write the failing tests**

Add after `test_distance_penalises_hiding_behind_unlabeled_material` (around line 260, before `test_attainment_is_one_when_reached_and_negative_when_worse`) in `tests/test_goals.py`:

```python
def test_distance_applies_weights_to_mentioned_class_terms():
    start = _aggregated(skeleton=0.01, lungs=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}, "lungs": {"vis": -0.3}})
    new_skeleton = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    # Give skeleton 0 error (goal met exactly) and lungs a known nonzero error
    # (goal unmet) so only lungs' weight shows up in the total.
    current = _aggregated(skeleton=new_skeleton, lungs=0.05)
    c_lungs_err = abs(0.0 - (-0.3))  # lungs stayed put: c=0, d=-0.3

    unweighted = goals.distance(goal, start, current)
    weighted = goals.distance(goal, start, current, weights={"skeleton": 1.0, "lungs": 0.5})

    assert unweighted == pytest.approx(c_lungs_err)
    assert weighted == pytest.approx(0.5 * c_lungs_err)


def test_distance_weights_default_to_one_for_classes_missing_from_the_dict():
    start = _aggregated(skeleton=0.01)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    current = _aggregated(skeleton=0.01)  # no progress -> full error

    without_weights = goals.distance(goal, start, current)
    with_partial_weights = goals.distance(goal, start, current, weights={"lungs": 0.5})

    assert with_partial_weights == pytest.approx(without_weights)


def test_distance_weights_do_not_affect_unmentioned_keep_tolerance():
    """A weight entry for a class that the goal never mentions must not
    disturb its keep-tolerance treatment -- weights only ever scale the
    mentioned-class term, never substitute for the mentioned flag."""
    start = _aggregated(skeleton=0.01, lungs=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}})
    new_skeleton = 10 ** (0.3 + math.log10(0.01 + goals.EPSILON)) - goals.EPSILON
    with_drift = _aggregated(skeleton=new_skeleton, lungs=0.08)

    without_weights = goals.distance(goal, start, with_drift)
    # lungs is unmentioned (m[i]=0 for lungs); a weight entry for it must be ignored.
    with_weights = goals.distance(goal, start, with_drift, weights={"lungs": 0.1})

    assert with_weights == pytest.approx(without_weights)


def test_attainment_accepts_weights_and_forwards_them():
    start = _aggregated(skeleton=0.01, lungs=0.05)
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}, "lungs": {"vis": -0.3}})
    current = _aggregated(skeleton=0.01, lungs=0.05)  # no progress on either
    weights = {"skeleton": 1.0, "lungs": 0.5}

    expected = 1.0 - goals.distance(goal, start, current, weights=weights) / \
        goals.distance(goal, start, start, weights=weights)
    assert goals.attainment(goal, start, current, weights=weights) == pytest.approx(expected)
    # Omitting weights must reproduce today's unweighted behavior exactly.
    assert goals.attainment(goal, start, current) == pytest.approx(
        1.0 - goals.distance(goal, start, current) / goals.distance(goal, start, start))
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goals.py -k "weights" -v`
Expected: FAIL with `TypeError: distance() got an unexpected keyword argument 'weights'` (and similarly for `attainment`).

- [x] **Step 3: Implement**

In `goals.py`, replace the `distance` function (currently lines 176-205):

```python
def distance(goal: np.ndarray, start: dict, current: dict, weights: dict = None) -> float:
    """How far a state is from the goal.

    sum m*w*|c - d| + KAPPA*sum n*w*|b - e|
      + LAMBDA_KEEP*sum (1-m)*max(0, |c| - KEEP_TOLERANCE)
      + LAMBDA_KEEP*KAPPA*sum (1-n)*max(0, |b| - KEEP_TOLERANCE)

    Moving one peak inevitably shifts what occludes what elsewhere in the
    volume, so an unmentioned class always drifts a little even when the
    instruction is followed well; a keep term that punishes any drift at
    all makes leaving the transfer function untouched score better than
    acting on the instruction. KEEP_TOLERANCE gives that small, unavoidable
    side effect for free and only penalises drift beyond it.

    `weights` (goal_class -> multiplier, default None = every class 1.0) only
    ever scales a *mentioned* class's own term -- it has no effect on a class
    the goal doesn't mention, whose treatment is entirely the keep-tolerance
    branch above. This is how `show_only` gives its shown-class group and
    hidden-class group equal total weight despite one side having far fewer
    classes than the other (see `show_only_weights`).
    """
    c, b = progress(start, current)
    n_classes = len(GOAL_CLASSES)
    d, m, e, n = goal[0:n_classes], goal[n_classes:2 * n_classes], \
        goal[2 * n_classes:3 * n_classes], goal[3 * n_classes:4 * n_classes]
    total = 0.0
    for i, goal_class in enumerate(GOAL_CLASSES):
        w = 1.0 if weights is None else weights.get(goal_class, 1.0)
        total += m[i] * w * abs(c[goal_class] - d[i]) + KAPPA * n[i] * w * abs(b[goal_class] - e[i])
        total += LAMBDA_KEEP * (1.0 - m[i]) * max(0.0, abs(c[goal_class]) - KEEP_TOLERANCE)
        total += LAMBDA_KEEP * KAPPA * (1.0 - n[i]) * max(0.0, abs(b[goal_class]) - KEEP_TOLERANCE)
    # "other" (unlabeled tissue) can never be named by an instruction, so it is
    # always the unmentioned case -- without this, a transfer function can
    # satisfy "show only X" by rendering an opaque wall of unclassified
    # material instead of X, since "other" is not a goal class and is never
    # charged by the mentioned-class terms.
    total += LAMBDA_KEEP * max(0.0, abs(c["other"]) - KEEP_TOLERANCE)
    return float(total)


def scoring(goal, start_features, final_features, active_layers=None, weights: dict = None) -> float:
    """Score final model features against start features using effective layers."""
    start = aggregate(start_features, active_layers)
    final = aggregate(final_features, active_layers)
    return attainment(goal, start, final, weights=weights)


def attainment(goal, start, final, active_layers=None, weights: dict = None) -> float:
    """1 - distance(final) / distance(start); 1 = goal reached, < 0 = worse."""
    return 1.0 - distance(goal, start, final, weights=weights) / distance(goal, start, start, weights=weights)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goals.py -v`
Expected: all PASS, including the 4 new tests and every pre-existing test in the file unchanged.

- [x] **Step 5: Commit**

```bash
git add goals.py tests/test_goals.py
git commit -m "feat(goals): accept an optional per-class weights multiplier in distance/attainment/scoring"
```

---

### Task 2: `goals.show_only_weights` helper, wired into both real-instruction construction sites

**Files:**
- Modify: `goals.py:422-434` (`_sample_show_only`), `goals.py:554-564` (`_goal_show_only`)
- Test: `tests/test_goals.py`

- [x] **Step 1: Write the failing tests**

Add directly after `EPSILON`/`KAPPA`/etc. constant block usage is irrelevant here -- add these new tests right after `test_show_only_hides_the_other_classes` (around line 409) in `tests/test_goals.py`:

```python
def test_show_only_weights_gives_each_group_equal_total_weight():
    classes = list(goals.GOAL_CLASSES)  # 8 classes
    weights = goals.show_only_weights({"skeleton"}, classes)
    assert weights["skeleton"] == pytest.approx(1.0)
    hidden = [c for c in classes if c != "skeleton"]
    assert len(hidden) == 7
    for c in hidden:
        assert weights[c] == pytest.approx(1.0 / 7)
    assert sum(weights[c] for c in hidden) == pytest.approx(1.0)


def test_show_only_weights_with_two_shown_classes():
    classes = list(goals.GOAL_CLASSES)
    weights = goals.show_only_weights({"skeleton", "lungs"}, classes)
    assert weights["skeleton"] == pytest.approx(0.5)
    assert weights["lungs"] == pytest.approx(0.5)
    hidden = [c for c in classes if c not in ("skeleton", "lungs")]
    assert len(hidden) == 6
    for c in hidden:
        assert weights[c] == pytest.approx(1.0 / 6)


def test_sampled_show_only_instruction_carries_group_balanced_weights(monkeypatch):
    monkeypatch.setattr(goals.totalseg, "classes_present", lambda name: _all_classes_present())
    monkeypatch.setattr(goals.totalseg, "is_contrast", lambda name: True)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    rng = np.random.default_rng(2)
    found = False
    for _ in range(400):
        instruction = goals.sample_instruction("fake", model, start, rng)
        if instruction["kind"] != "show_only":
            continue
        found = True
        shown = {c for c, t in instruction["targets"].items() if t["vis"] > 0.0}
        expected = goals.show_only_weights(shown, instruction["targets"].keys())
        assert instruction["weights"] == expected
    assert found
```

Add after `test_goal_from_command_show_only_single_string_target` (end of file section, around line 534) in `tests/test_goals.py`:

```python
def test_goal_from_command_show_only_carries_group_balanced_weights(monkeypatch):
    _patch_contrast_volume(monkeypatch)
    model = _StubModel()
    start = _aggregated(skeleton=0.05, lungs=0.05, soft=0.05, vessels=0.05)
    cmd = {"target": ["skeleton", "lungs"], "attribute": "opacity", "direction": "show_only", "strength": None}
    result = goals.goal_from_command(cmd, model, start, volume="fake")
    expected = goals.show_only_weights({"skeleton", "lungs"}, result["targets"].keys())
    assert result["weights"] == expected
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goals.py -k "show_only_weights or group_balanced" -v`
Expected: FAIL -- `AttributeError: module 'goals' has no attribute 'show_only_weights'`, and `KeyError: 'weights'` once that's fixed but the construction sites aren't wired yet.

- [x] **Step 3: Implement**

In `goals.py`, add the new helper directly above `_sample_show_only` (currently line 422):

```python
def show_only_weights(shown: set, classes) -> dict:
    """Equal total weight for the shown group and the hidden group,
    regardless of how many classes are in each -- 1/n_shown per shown class,
    1/n_hidden per hidden class. Without this, `distance`'s flat per-class
    weighting lets a show_only goal's 6-7 hidden classes outweigh its 1-2
    shown ones purely by having more terms."""
    shown = set(shown)
    hidden = [c for c in classes if c not in shown]
    weights = {c: 1.0 / len(shown) for c in shown}
    weights.update({c: 1.0 / len(hidden) for c in hidden})
    return weights
```

Change `_sample_show_only` (currently lines 422-434) to attach weights to its returned tuple's targets won't work directly since it returns `(targets, text)`, not the full instruction -- the weights need to flow into the instruction dict built by `sample_instruction`. Update both:

```python
def _sample_show_only(classes: list, rng, preferred=None) -> tuple:
    n = min(int(rng.integers(1, 3)), len(classes))
    shown = ([preferred] if preferred in classes else [])
    remaining = [c for c in classes if c not in shown]
    if len(shown) < n:
        shown.extend(list(rng.choice(remaining, size=n - len(shown), replace=False)))
    targets = {goal_class: {"vis": HIDE_STRENGTH} for goal_class in shown}
    for goal_class in classes:
        if goal_class not in shown:
            targets[goal_class] = {"vis": -HIDE_STRENGTH}
    words = [_class_word(goal_class, rng) for goal_class in shown]
    text = "show only the " + " and the ".join(words)
    return targets, text, show_only_weights(set(shown), classes)
```

This changes `_sample_show_only`'s return arity from 2 to 3. Update its only caller, `sample_instruction` (currently lines 454-482):

```python
def sample_instruction(name, model, start_features, rng, active_layers=None,
                       class_counts=None, balance_classes=False) -> dict:
    """One instruction: {"kind", "text", "targets", "goal"}, plus "weights"
    for show_only instructions (see `show_only_weights`).

    `targets` is the goal_vector input; `text` is the spoken form. Absolute
    levels use ABSOLUTE_LEVEL x model.solo_max aggregated for the goal class,
    turned into a requested change from the start state, so "high skeleton"
    means the same thing on every volume.
    """
    classes = (reachable_goal_classes(name, model, active_layers)
               if active_layers is not None else reachable_goal_classes(name, model))
    preferred = None
    if balance_classes and class_counts:
        minimum = min(class_counts.get(goal_class, 0) for goal_class in classes)
        least_seen = [goal_class for goal_class in classes
                      if class_counts.get(goal_class, 0) == minimum]
        preferred = str(rng.choice(least_seen))
    kind = _sample_kind(rng)
    weights = None
    if kind == "relative":
        targets, text = _sample_relative(classes, rng, preferred)
    elif kind == "compound":
        targets, text = _sample_compound(classes, rng, preferred)
    elif kind == "show_only":
        targets, text, weights = _sample_show_only(classes, rng, preferred)
    elif kind == "absolute":
        targets, text = _sample_absolute(classes, model, start_features, rng, preferred)
    else:
        targets, text = _sample_brightness(classes, rng, preferred)
    instruction = {"kind": kind, "text": text, "targets": targets, "goal": goal_vector(targets)}
    if weights is not None:
        instruction["weights"] = weights
    return instruction
```

Change `_goal_show_only` (currently lines 554-564) and `goal_from_command`'s caller `_goal_from_command`:

```python
def _goal_show_only(command: dict, volume) -> dict:
    shown = command["target"] if isinstance(command["target"], list) else [command["target"]]
    for goal_class in shown:
        _check_class_supported(goal_class, volume)
    classes = goal_classes_for_volume(volume) if volume is not None else list(GOAL_CLASSES)
    targets = {goal_class: {"vis": HIDE_STRENGTH} for goal_class in shown}
    for goal_class in classes:
        if goal_class not in shown:
            targets[goal_class] = {"vis": -HIDE_STRENGTH}
    text = _show_only_command_text(shown)
    return {"kind": "show_only", "text": text, "targets": targets, "goal": goal_vector(targets),
            "weights": show_only_weights(set(shown), classes)}
```

(`_goal_relative`, `_goal_absolute`, `_goal_brightness` and every other `_goal_*` function are unchanged -- they never set `"weights"`.)

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goals.py -v`
Expected: all PASS, including every pre-existing test (in particular `test_every_sampled_instruction_has_text_and_a_goal_vector` and `test_sampling_is_deterministic_for_a_seed`, which iterate over every kind including `show_only` and must still pass unchanged).

- [x] **Step 5: Commit**

```bash
git add goals.py tests/test_goals.py
git commit -m "feat(goals): attach group-balanced weights to show_only instructions"
```

---

### Task 3: Wire `weights` into `OneShotEnv`'s hindsight show_only sampler and its reward/attainment computation

**Files:**
- Modify: `rl/oneshot_env.py:355-358` (`_attainment`), `rl/oneshot_env.py:430-459` (`_sample_hindsight_show_only`), `rl/oneshot_env.py:530` (`reset`), `rl/oneshot_env.py:564-566` (`step`)
- Test: `tests/test_oneshot_env.py`

- [x] **Step 1: Write the failing tests**

Add after `test_hindsight_show_only_action_matches_measured_deltas` (around line 450) in `tests/test_oneshot_env.py`:

```python
def test_hindsight_show_only_targets_carry_group_balanced_weights():
    env = OneShotEnv(["fake_a"], model_for_volume=lambda name: _StubModel(),
                     hindsight_ratio=1.0, hindsight_show_only_share=1.0)
    for seed in range(5):
        env.reset(seed=seed)
        targets = env._instruction["targets"]
        shown = {c for c, t in targets.items() if t["vis"] > 0}
        expected = goals.show_only_weights(shown, goals.GOAL_CLASSES)
        assert env._instruction["weights"] == expected
```

Add a new test near `test_hindsight_episodes_report_their_source` (end of file) verifying the reward actually reflects the weighting -- this is the integration check that the plumbing (not just the instruction dict) is correct:

```python
def test_show_only_reward_reflects_group_balanced_weights(monkeypatch):
    """A show_only episode's reward (reward_mode="target") must use the same
    group-balanced weights as the instruction's own "weights" field -- not
    just carry the field around unused."""
    _patch_totalseg(monkeypatch)
    env = OneShotEnv(["fake_a"], model_for_volume=lambda name: _StubModel(),
                     action_mode="residual", reward_mode="target")
    env.reset(seed=0)
    # Force a known show_only instruction directly, bypassing sampling, so
    # the expected reward can be hand-computed.
    shown = {"skeleton"}
    targets = {c: {"vis": goals.HIDE_STRENGTH if c in shown else -goals.HIDE_STRENGTH}
               for c in goals.GOAL_CLASSES}
    weights = goals.show_only_weights(shown, goals.GOAL_CLASSES)
    env._instruction = {"kind": "show_only", "text": "test", "targets": targets,
                        "goal": goals.goal_vector(targets), "weights": weights}
    env._start_distance = goals.distance(env._instruction["goal"], env._start_agg,
                                         env._start_agg, weights=weights)

    action = np.zeros(ACTION_SIZE, dtype=np.float64)  # no-op residual action
    _obs, reward, _term, _trunc, info = env.step(action)

    final_agg = env._start_agg  # no-op action -> final state equals start state
    expected_distance = goals.distance(env._instruction["goal"], env._start_agg,
                                       final_agg, weights=weights)
    expected_progress = (env._start_distance - expected_distance) / env._start_distance
    assert info["target_progress"] == pytest.approx(expected_progress)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py -k "group_balanced" -v`
Expected: FAIL -- `KeyError: 'weights'` on the first test (hindsight sampler doesn't set it yet); the second test currently passes by coincidence only if weights don't change the no-op case's distance ratio, so verify it fails once weights legitimately change a nonzero case -- if it passes before Step 3, that means the no-op/all-zero-progress case doesn't exercise the weighting; re-check against a nonzero-progress action before proceeding. (A no-op residual action keeps `final_agg == start_agg`, so both numerator and denominator distances are evaluated at identical input, meaning this specific test's main value is confirming `weights` is threaded into both calls without a `TypeError`/`KeyError`, not that the ratio changed -- that's covered by Task 1's `distance` tests. Keep it as a plumbing check.)

- [x] **Step 3: Implement**

In `rl/oneshot_env.py`, change `_attainment` (currently lines 355-358):

```python
    def _attainment(self, agg: dict) -> float:
        if self._start_distance <= _ATTAINMENT_FLOOR:
            return 0.0
        return goals.attainment(self._instruction["goal"], self._start_agg, agg,  # pyright: ignore[reportOptionalSubscript]
                                weights=self._instruction.get("weights"))  # pyright: ignore[reportOptionalSubscript]
```

Change `_sample_hindsight_show_only`'s return (currently lines 455-458, inside the method at lines 430-459):

```python
            if shown_ok and hidden_ok:
                targets = {c: {"vis": deltas[c]} for c in goals.GOAL_CLASSES}
                weights = goals.show_only_weights(shown, goals.GOAL_CLASSES)
                return ({"kind": "hindsight", "text": None, "targets": targets,
                         "goal": goals.goal_vector(targets), "weights": weights}, action)
```

Change `reset`'s `start_distance` line (currently line 530):

```python
        start_distance = goals.distance(instruction["goal"], start_agg, start_agg,
                                        weights=instruction.get("weights"))
```

Change `step`'s `reward_mode == "target"` branch (currently lines 564-566):

```python
        if self.reward_mode == "target":
            weights = self._instruction.get("weights")  # pyright: ignore[reportOptionalSubscript]
            start_distance = goals.distance(self._instruction["goal"], self._start_agg,  # pyright: ignore[reportOptionalSubscript]
                                            self._start_agg, weights=weights)
            final_distance = goals.distance(self._instruction["goal"], self._start_agg,  # pyright: ignore[reportOptionalSubscript]
                                            final_agg, weights=weights)
```

(The `drift` calculation just below stays unchanged -- `drift` only ever sums over classes `not in mentioned`, and `mentioned = set(self._instruction["targets"])` is already all 8 classes for `show_only`, so `drift` is always 0 for this kind regardless of weights, exactly as before this change.)

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py -v`
Expected: all PASS, including every pre-existing test in the file.

- [x] **Step 5: Commit**

```bash
git add rl/oneshot_env.py tests/test_oneshot_env.py
git commit -m "feat(rl): apply show_only's group-balanced weights to OneShotEnv's reward and attainment"
```

---

### Task 4: Wire `weights` into `rl/baselines.py`'s `hill_climb` search objective

**Files:**
- Modify: `rl/baselines.py:200-240` (`_evaluate`, `hill_climb`)
- Test: `tests/test_baselines.py`

- [x] **Step 1: Write the failing tests**

In `tests/test_baselines.py`, update the existing `_show_only_instruction` helper (currently lines 78-82) to carry weights, matching production instructions:

```python
def _show_only_instruction(shown: list) -> dict:
    targets = {}
    for goal_class in goals.GOAL_CLASSES:
        targets[goal_class] = {"vis": goals.HIDE_STRENGTH if goal_class in shown else -goals.HIDE_STRENGTH}
    weights = goals.show_only_weights(set(shown), goals.GOAL_CLASSES)
    return {"kind": "show_only", "text": "test", "targets": targets, "goal": goals.goal_vector(targets),
            "weights": weights}
```

Add after `test_hill_climb_passes_active_layers_to_effective_scoring` (around line 263) in `tests/test_baselines.py`:

```python
def test_hill_climb_applies_show_only_weights_to_its_own_objective():
    """hill_climb must search against the *same* weighted objective the RL
    policy is trained and judged on for show_only -- otherwise the search
    teacher used for distillation targets a different goal than the policy."""
    model = _StubModel()
    start = _start_params()
    instruction = _show_only_instruction(["skeleton"])
    start_agg = goals.aggregate(model.features(start))

    result = baselines.hill_climb(model, start, instruction, evaluations=80)

    unweighted_after = goals.distance(instruction["goal"], start_agg, goals.aggregate(model.features(result)))
    weighted_after = goals.distance(instruction["goal"], start_agg, goals.aggregate(model.features(result)),
                                    weights=instruction["weights"])
    unweighted_before = goals.distance(instruction["goal"], start_agg, start_agg)
    weighted_before = goals.distance(instruction["goal"], start_agg, start_agg, weights=instruction["weights"])
    # The search must have actually reduced the *weighted* distance (what it
    # optimizes), not just coincidentally reduced the unweighted one.
    assert weighted_after <= weighted_before
    assert unweighted_after <= unweighted_before


def test_evaluate_applies_weights():
    model = _StubModel()
    start = _start_params()
    instruction = _show_only_instruction(["skeleton"])
    start_agg = goals.aggregate(model.features(start))

    unweighted = baselines._evaluate(model, instruction["goal"], start_agg, start)
    weighted = baselines._evaluate(model, instruction["goal"], start_agg, start,
                                   weights=instruction["weights"])

    assert weighted != pytest.approx(unweighted)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baselines.py -k "weights" -v`
Expected: FAIL -- `TypeError: _evaluate() got an unexpected keyword argument 'weights'`.

- [x] **Step 3: Implement**

In `rl/baselines.py`, change `_evaluate` and `hill_climb` (currently lines 200-240):

```python
def _evaluate(model, goal, start_agg, params, active_layers=None, weights: dict = None) -> float:
    features = goals.features(model, params, active_layers)
    return goals.distance(goal, start_agg, goals.aggregate(features, active_layers), weights=weights)


def hill_climb(model, start_params, instruction, evaluations: int = 200,
               initial_step: float = 0.2, active_layers=None) -> np.ndarray:
    """Coordinate search on `goals.distance`: try +-step on each of the 12
    controllable values, keep improvements, halve the step once a full sweep
    finds none. `evaluations` bounds the number of `model.features` calls
    (one to score the start state, one per proposal after that). Applies
    `instruction`'s own `weights` (set for show_only; None otherwise) so this
    search optimizes the identical objective the RL policy is trained and
    judged on."""
    goal = instruction["goal"]
    weights = instruction.get("weights")
    start_agg = goals.aggregate(goals.features(model, start_params, active_layers), active_layers)
    used = 1

    best_params = start_params.copy()
    best_distance = goals.distance(goal, start_agg, start_agg, weights=weights)
    step = initial_step

    while used < evaluations and step > 1e-3:
        improved = False
        for dim in CONTROLLABLE:
            for sign in (1.0, -1.0):
                if used >= evaluations:
                    break
                candidate = best_params.copy()
                for idx in dim:
                    candidate[idx] = float(np.clip(candidate[idx] + sign * step, -1.0, 1.0))
                candidate_distance = _evaluate(model, goal, start_agg, candidate, active_layers, weights=weights)
                used += 1
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_params = candidate
                    improved = True
            if used >= evaluations:
                break
        if not improved:
            step /= 2.0
    return best_params
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baselines.py -v`
Expected: all PASS, including every pre-existing test in the file.

- [x] **Step 5: Commit**

```bash
git add rl/baselines.py tests/test_baselines.py
git commit -m "feat(rl): apply show_only's group-balanced weights to hill_climb's search objective"
```

---

### Task 5: Wire `weights` into `rl/vis_eval.py`'s scoring

**Files:**
- Modify: `rl/vis_eval.py:110-163` (`_frozen_episode_env`, `_frozen_one_shot_env`), `rl/vis_eval.py:269-277` (`_attainment_or_zero`)
- Test: `tests/test_vis_eval.py`

- [x] **Step 1: Write the failing test**

Add this self-contained unit test anywhere in `tests/test_vis_eval.py` (it exercises `_attainment_or_zero` directly as a pure function, independent of that file's episode/env fixtures):

```python
def test_attainment_or_zero_applies_weights():
    start = {"vis": {c: 0.0 for c in goals.GOAL_CLASSES}, "bright": {c: 0.0 for c in goals.GOAL_CLASSES}}
    start["vis"]["skeleton"] = 0.01
    final = {"vis": dict(start["vis"]), "bright": dict(start["bright"])}
    goal = goals.goal_vector({"skeleton": {"vis": 0.3}, "lungs": {"vis": -0.3}})
    weights = {"skeleton": 1.0, "lungs": 0.5}

    unweighted = vis_eval._attainment_or_zero(goal, start, final)
    weighted = vis_eval._attainment_or_zero(goal, start, final, weights=weights)

    assert weighted != pytest.approx(unweighted)
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_vis_eval.py -k "attainment_or_zero_applies_weights" -v`
Expected: FAIL -- `TypeError: _attainment_or_zero() got an unexpected keyword argument 'weights'`.

- [x] **Step 3: Implement**

In `rl/vis_eval.py`, change `_attainment_or_zero` (currently lines 269-277):

```python
def _attainment_or_zero(goal, start_agg: dict, final_agg: dict, weights: dict = None) -> float:
    # Same floor as VisibilityTFEnv._attainment: when the start state
    # already meets the goal, distance(start) is ~0 and attainment's ratio
    # is meaningless -- report 0 instead of dividing by ~0, so the policy
    # and the baselines are floored identically.
    start_distance = goals.distance(goal, start_agg, start_agg, weights=weights)
    if start_distance <= _ATTAINMENT_FLOOR:
        return 0.0
    return goals.attainment(goal, start_agg, final_agg, weights=weights)
```

Update its one call site, in `run_baseline` (currently line 300):

```python
            attainment = _attainment_or_zero(instruction["goal"], start_agg, final_agg,
                                             weights=instruction.get("weights"))
```

Change `_frozen_episode_env`'s `start_distance` line (currently line 120):

```python
    start_distance = goals.distance(instruction["goal"], start_agg, start_agg,
                                    weights=instruction.get("weights"))
```

Change `_frozen_one_shot_env`'s `start_distance` line (currently line 149) identically:

```python
    start_distance = goals.distance(instruction["goal"], start_agg, start_agg,
                                    weights=instruction.get("weights"))
```

(Both functions already store `env._instruction = instruction` and compute `env._start_distance`/observations from `OneShotEnv`/`VisibilityTFEnv`'s own `_attainment`, which Tasks 3 and 6 make weights-aware via `self._instruction.get("weights")` -- so only the one direct `goals.distance` call in each of these two functions needs updating here; everything downstream of it already gets weights through the env's own methods.)

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vis_eval.py -v`
Expected: all PASS, including every pre-existing test in the file.

- [x] **Step 5: Commit**

```bash
git add rl/vis_eval.py tests/test_vis_eval.py
git commit -m "feat(rl): apply show_only's group-balanced weights in vis_eval scoring"
```

---

### Task 6: Wire `weights` into `rl/vis_env.py` (the legacy multi-step env)

**Files:**
- Modify: `rl/vis_env.py:140-143` (`_attainment`), `rl/vis_env.py:161` (`reset`), `rl/vis_env.py:188` (`step`)
- Test: `tests/test_vis_env.py`

- [x] **Step 1: Write the failing test**

`tests/test_vis_env.py` already has a `_make_env(monkeypatch, volume_ids=...)` helper and a `_StubModel` with full 8-class coverage (`MEASURED = {name: name for name in CANONICAL_CLASSES}`). Add this test using that helper:

```python
def test_show_only_attainment_reflects_group_balanced_weights(monkeypatch):
    env = _make_env(monkeypatch, volume_ids=("fake_a",))
    env.reset(seed=0)
    shown = {"skeleton"}
    targets = {c: {"vis": goals.HIDE_STRENGTH if c in shown else -goals.HIDE_STRENGTH}
               for c in goals.GOAL_CLASSES}
    weights = goals.show_only_weights(shown, goals.GOAL_CLASSES)
    env._instruction = {"kind": "show_only", "text": "test", "targets": targets,
                        "goal": goals.goal_vector(targets), "weights": weights}
    env._start_distance = goals.distance(env._instruction["goal"], env._start_agg,
                                         env._start_agg, weights=weights)
    # skeleton (shown) makes no progress; every hidden class is perfectly
    # suppressed -- unweighted scoring lets the many suppressed hidden
    # classes swamp the one unmet shown class; weighted scoring must not.
    agg = {"vis": dict(env._start_agg["vis"]), "bright": dict(env._start_agg["bright"]),
           "coverage": env._start_agg["coverage"]}
    for c in goals.GOAL_CLASSES:
        if c not in shown:
            agg["vis"][c] = 0.0
    expected = goals.attainment(env._instruction["goal"], env._start_agg, agg, weights=weights)
    assert env._attainment(agg) == pytest.approx(expected)
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_vis_env.py -k "show_only" -v`
Expected: FAIL (new test not yet passing against unmodified code -- `_attainment` ignores `"weights"` until Step 3).

- [x] **Step 3: Implement**

In `rl/vis_env.py`, change `_attainment` (currently lines 140-143):

```python
    def _attainment(self, agg: dict) -> float:
        if self._start_distance <= _ATTAINMENT_FLOOR:
            return 0.0
        return goals.attainment(self._instruction["goal"], self._start_agg, agg,  # pyright: ignore[reportOptionalSubscript]
                                weights=self._instruction.get("weights"))  # pyright: ignore[reportOptionalSubscript]
```

Change `reset`'s `start_distance` line (currently line 161):

```python
        start_distance = goals.distance(instruction["goal"], start_agg, start_agg,
                                        weights=instruction.get("weights"))
```

Change `step`'s `distance` line (currently line 188):

```python
        distance = goals.distance(self._instruction["goal"], self._start_agg, agg,  # pyright: ignore[reportOptionalSubscript]
                                  weights=self._instruction.get("weights"))  # pyright: ignore[reportOptionalSubscript]
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vis_env.py -v`
Expected: all PASS, including every pre-existing test in the file.

- [x] **Step 5: Commit**

```bash
git add rl/vis_env.py tests/test_vis_env.py
git commit -m "feat(rl): apply show_only's group-balanced weights in VisibilityTFEnv"
```

---

### Task 7: Wire `weights` into `rl/candidates.py` and `server.py` (production inference path)

**Files:**
- Modify: `rl/candidates.py:167-168`, `server.py:452`, `server.py:551-552`
- Test: `tests/test_candidates.py`, `tests/test_server.py`

- [x] **Step 1: Write the failing tests, and fix one existing test that this change will otherwise break**

`tests/test_candidates.py:114-124`'s existing `test_objective_choice_matches_the_lower_distance` samples a random instruction per seed (via `sample_item` -> `goals.sample_instruction`) and independently recomputes `a_distance`/`b_distance` *without* weights to check against `item["objective_choice"]`. Once this task makes `objective_choice` use `instruction.get("weights")`, any seed that happens to sample a `show_only` instruction (~15% of kinds) will disagree with this test's unweighted recomputation. Fix it to thread weights through, matching what the implementation will do:

```python
def test_objective_choice_matches_the_lower_distance(monkeypatch):
    model = _make_model(monkeypatch)
    for seed in range(10):
        item = sample_item("fake_a", model, np.random.default_rng(seed), policy=_StubPolicy())
        start_agg = goals.aggregate(model.features(item["start_params"]))
        a_agg = goals.aggregate(model.features(item["a"]["params"]))
        b_agg = goals.aggregate(model.features(item["b"]["params"]))
        weights = item["instruction"].get("weights")
        a_distance = goals.distance(item["instruction"]["goal"], start_agg, a_agg, weights=weights)
        b_distance = goals.distance(item["instruction"]["goal"], start_agg, b_agg, weights=weights)
        expected = "a" if a_distance <= b_distance else "b"
        assert item["objective_choice"] == expected
```

Add a dedicated test proving the weighting actually changes which arm wins (not just that the comparison runs without error) directly after it:

```python
def test_objective_choice_uses_show_only_weights_not_flat_distance(monkeypatch):
    """Construct a case where the unweighted and weighted comparisons
    disagree: arm A is perfect on the single shown class but makes no
    progress on any hidden class; arm B is the reverse (perfect on every
    hidden class, no progress on the shown one). Flat per-class weighting
    picks B (6 improved hidden terms beat 1 improved shown term); weighted
    scoring must pick A instead."""
    shown = {"skeleton"}
    classes = list(goals.GOAL_CLASSES)
    targets = {c: {"vis": goals.HIDE_STRENGTH if c in shown else -goals.HIDE_STRENGTH} for c in classes}
    weights = goals.show_only_weights(shown, classes)
    instruction = {"kind": "show_only", "text": "test", "targets": targets,
                   "goal": goals.goal_vector(targets), "weights": weights}

    start_agg = {"vis": {c: 0.05 for c in classes}, "bright": {c: 0.0 for c in classes}}
    a_agg = {"vis": {**start_agg["vis"], "skeleton": 0.5}, "bright": start_agg["bright"]}  # shown fixed, hidden untouched
    b_agg = {"vis": {c: (0.5 if c != "skeleton" else 0.05) for c in classes}, "bright": start_agg["bright"]}  # hidden fixed, shown untouched

    weighted_a = goals.distance(instruction["goal"], start_agg, a_agg, weights=weights)
    weighted_b = goals.distance(instruction["goal"], start_agg, b_agg, weights=weights)
    unweighted_a = goals.distance(instruction["goal"], start_agg, a_agg)
    unweighted_b = goals.distance(instruction["goal"], start_agg, b_agg)

    assert weighted_a < weighted_b   # weighted scoring favors fixing the shown class
    assert unweighted_b < unweighted_a  # flat scoring favors fixing the many hidden classes instead
```

(This second test checks `goals.distance` itself, already implemented in Task 1/2 -- it documents and locks in *why* `rl/candidates.py`'s `objective_choice` needs the `weights=` passthrough this task adds, as a readable regression test independent of `sample_item`'s randomness.)

In `tests/test_server.py`, add a focused test for `compare_arms`'s `"attainment"` field using a `show_only` command (the existing `_compare_inputs` helper always uses `"more bone"`, a `relative` instruction never carrying `"weights"`, so it can't exercise this path):

```python
def test_compare_arms_scores_show_only_with_group_balanced_weights(ts_session):
    s = ts_session
    model = s.model_for_volume(server._dataset_name)
    params = np.array(s.history[s.cursor]["params"], dtype=np.float64)
    camera = dict(s.history[s.cursor]["camera"])
    start_agg = goals.aggregate(model.features(params))
    cmd, _ = server.parse_command_with_meta("show only bone", parser="rule")
    instruction = goals.goal_from_command(cmd, model, start_agg, volume=server._dataset_name)
    assert "weights" in instruction  # sanity: this command really is show_only-shaped

    arms = server.compare_arms(model, policy_stub(), cmd, instruction, params, camera, cheap=4)

    for name, arm in arms.items():
        replayed = np.array(arm["params"], dtype=np.float64)
        expected = goals.attainment(instruction["goal"], start_agg,
                                    goals.aggregate(model.features(replayed)),
                                    weights=instruction["weights"])
        assert arm["attainment"] == pytest.approx(expected), name
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_candidates.py tests/test_server.py -k "weight or objective_choice" -v`
Expected: `test_objective_choice_uses_show_only_weights_not_flat_distance` PASSes already (it only exercises `goals.distance`, already implemented); `test_compare_arms_scores_show_only_with_group_balanced_weights` FAILs against unmodified `server.py` (its `arm["attainment"]` was computed without weights); `test_objective_choice_matches_the_lower_distance` continues to pass for now (its fix anticipates Step 3's change, so it won't fail until *after* Step 3 if you skip the fix -- apply the fix in this step regardless, so the suite stays green through Step 3).

- [x] **Step 3: Implement**

In `rl/candidates.py`, change the objective-choice comparison (currently lines 167-168):

```python
    weights = instruction.get("weights")
    a_distance = goals.distance(instruction["goal"], start_agg, a_agg, weights=weights)
    b_distance = goals.distance(instruction["goal"], start_agg, b_agg, weights=weights)
```

In `server.py`, change the `compare_arms`/`_finish` attainment field (currently line 452):

```python
            "attainment": float(goals.attainment(instruction["goal"], start_agg, final_agg,
                                                 weights=instruction.get("weights"))),
```

Change `sweep_arms`'s `_score` closure (currently lines 550-552):

```python
        def _score(params):
            return float(goals.attainment(
                instruction["goal"], start_agg, goals.aggregate(model.features(params)),
                weights=instruction.get("weights")))
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_candidates.py tests/test_server.py -v`
Expected: all PASS, including every pre-existing test in both files.

- [x] **Step 5: Commit**

```bash
git add rl/candidates.py server.py tests/test_candidates.py tests/test_server.py
git commit -m "feat: apply show_only's group-balanced weights in candidate comparison and production inference"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: every test passes (same pass count as the pre-change baseline plus the new tests added in Tasks 1-7; 0 failures, 0 errors). If the suite has a `slow`-marked subset normally excluded, also run: `.venv/bin/python -m pytest -q -m slow` if that marker exists in this project (check `pytest.ini`/`pyproject.toml` for a `slow` marker definition first).

- [ ] **Step 2: Grep for any remaining unweighted show_only-relevant call site**

Run: `grep -rn "goals\.distance(\|goals\.attainment(\|goals\.scoring(" --include="*.py" . | grep -v "/tests/\|def distance\|def attainment\|def scoring\|\.worktrees"`

Expected: every remaining line already passes `weights=` (either `weights=instruction.get("weights")`, `weights=weights`, or an explicit test value) -- confirms no call site was missed per the spec's table. If any line lacks a `weights=` argument and its surrounding function has an `instruction` (or already-extracted `weights`) in scope, that is a gap Task 7 or an earlier task should have covered -- fix it and re-run Step 1.

- [ ] **Step 3: No commit needed for this task** (verification only; if Step 2 finds a gap, fix it as a new commit and re-run Step 1).
