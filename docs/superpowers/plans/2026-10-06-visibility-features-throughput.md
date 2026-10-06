# Visibility `features()` throughput — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut RL training wall-clock cost by removing redundant tensor work
inside `VisibilityModel.features()` (currently computed twice per call, once
"plain" and once "effective", even when no anatomy layers are active) and by
giving the hindsight-goal retry loop in `rl/oneshot_env.py` a cheap
visibility-only path instead of the full feature computation it doesn't need.

**Architecture:** Split `visibility.VisibilityModel._weights` into three
reusable pieces (`_alpha_luminance`, `_layer_mask`, `_composite`) so
`features()` computes the expensive layer-independent half exactly once per
call, and add a new `vis_only()` method built from the same pieces that skips
luminance/brightness/coverage entirely. `goals.py` gets the same split
(`_aggregate_vis` shared by `aggregate` and a new `aggregate_vis`), and
`OneShotEnv` gets a `_vis()` helper (mirroring the existing `_features()`)
that the hindsight visibility search loop uses in place of
`_features()` + `goals.aggregate()`.

**Tech Stack:** Python, PyTorch (CPU tensors), pytest. No new dependencies.

Reference: `docs/superpowers/specs/2026-10-06-visibility-features-throughput-design.md`

---

### Task 1: Split `VisibilityModel._weights` into reusable pieces, stop duplicating the layer-independent work

**Files:**
- Modify: `visibility.py:184-245` (`_weights`, `features`)
- Test: `tests/test_visibility.py`

- [ ] **Step 1: Add a regression test pinning "effective equals plain when no layers are given"**

Add to `tests/test_visibility.py`, right after `test_explicit_layer_opacity_populates_effective_fields` (currently ending at line 148):

```python
def test_no_layers_effective_fields_equal_plain_fields():
    """`features(params)` (no `layers` arg) must report `effective_*` equal
    to the plain fields -- this is the invariant the features()/vis_only()
    fast path for `layers=None` relies on. Compared per-class rather than
    dict-equality: `bright` has no "other" key (nothing ever targets its
    colour) while `effective_bright["other"]` always defaults to 0.0 via
    `.get` -- that asymmetry is pre-existing and out of scope here, so this
    test doesn't assert the two dicts are equal as a whole, only that every
    key `bright` actually has matches its `effective_bright` counterpart."""
    model = _model(_slab_volume(-1000.0, 900.0))
    params = _params([0.0, 0.0, 0.0, 0.9])
    result = model.features(params)
    for name in visibility.CLASSES:
        assert result["effective_vis"][name] == result["vis"][name]
        assert result["effective_bright"][name] == result["bright"][name]
    assert result["effective_vis"]["other"] == result["vis"]["other"]
    assert result["effective_coverage"] == pytest.approx(result["coverage"])
```

- [ ] **Step 2: Run it to confirm it already passes**

Run: `.venv/bin/python -m pytest tests/test_visibility.py::test_no_layers_effective_fields_equal_plain_fields -v`
Expected: PASS (today's code already computes an equal value the slow way —
this test pins that invariant before the refactor changes *how* it's
computed, not *what* the result is).

- [ ] **Step 3: Read the current `_weights`/`features` code**

Re-read `visibility.py:184-245` to have the exact current text in front of you (the `Edit` tool requires an exact match).

- [ ] **Step 4: Replace `_weights` with three smaller methods, in `visibility.py`**

Replace:

```python
    def _weights(self, params, layers=None, transfer_mode=transfer.DEFAULT_TRANSFER_MODE):
        """(contribution per sample, luminance per sample, accumulated opacity per ray)."""
        opacity_table, luminance_table = label_transfer_tables(params, transfer_mode)
        index = self._label_index()
        alpha = torch.from_numpy(opacity_table.reshape(-1))[index]
        alpha = 1.0 - (1.0 - alpha) ** (self.step_mm / 1.0)
        if layers is not None:
            normalized = normalize_layers(layers)
            layer_opacity = torch.ones_like(alpha)
            for class_index, name in enumerate(CLASSES):
                layer_opacity = torch.where(
                    self._labels == class_index,
                    layer_opacity.new_tensor(normalized[name]["opacity"]),
                    layer_opacity,
                )
            alpha = alpha * layer_opacity
        luminance = torch.from_numpy(luminance_table.reshape(-1))[index]
        transparency = 1.0 - alpha
        ones = torch.ones_like(transparency[:, :1])
        before = torch.cat([ones, torch.cumprod(transparency, dim=1)[:, :-1]], dim=1)
        return before * alpha, luminance, 1.0 - transparency.prod(dim=1)
```

with:

```python
    def _alpha_luminance(self, params, transfer_mode):
        """(alpha per sample, luminance per sample), before any layer mask --
        the layer-independent half of the old `_weights`: LUT build, index
        lookup, and the `(1 - alpha) ** step_mm` term. The expensive part,
        so callers that need both a plain and a layer-masked composite
        (`features`) compute this once and branch after, instead of paying
        for it twice."""
        opacity_table, luminance_table = label_transfer_tables(params, transfer_mode)
        index = self._label_index()
        alpha = torch.from_numpy(opacity_table.reshape(-1))[index]
        alpha = 1.0 - (1.0 - alpha) ** (self.step_mm / 1.0)
        luminance = torch.from_numpy(luminance_table.reshape(-1))[index]
        return alpha, luminance

    def _layer_mask(self, alpha, layers):
        """Per-sample layer-opacity multiplier for `layers` (already
        normalized). `alpha` is only used for its shape/dtype/device."""
        layer_opacity = torch.ones_like(alpha)
        for class_index, name in enumerate(CLASSES):
            layer_opacity = torch.where(
                self._labels == class_index,
                layer_opacity.new_tensor(layers[name]["opacity"]),
                layer_opacity,
            )
        return layer_opacity

    def _composite(self, alpha):
        """(contribution per sample, accumulated opacity per ray) from alpha."""
        transparency = 1.0 - alpha
        ones = torch.ones_like(transparency[:, :1])
        before = torch.cat([ones, torch.cumprod(transparency, dim=1)[:, :-1]], dim=1)
        return before * alpha, 1.0 - transparency.prod(dim=1)

    def _weights(self, params, layers=None, transfer_mode=transfer.DEFAULT_TRANSFER_MODE):
        """(contribution per sample, luminance per sample, accumulated opacity per ray)."""
        alpha, luminance = self._alpha_luminance(params, transfer_mode)
        if layers is not None:
            alpha = alpha * self._layer_mask(alpha, normalize_layers(layers))
        weights, accumulated = self._composite(alpha)
        return weights, luminance, accumulated
```

`_weights` keeps its exact old signature and return shape, so `per_view_visibility` (its only other caller, further down in this file) needs no change.

- [ ] **Step 5: Run the full visibility test file to confirm nothing broke**

Run: `.venv/bin/python -m pytest tests/test_visibility.py -v`
Expected: PASS (all tests, including the new one from Step 1)

- [ ] **Step 6: Commit**

```bash
git add visibility.py tests/test_visibility.py
git commit -m "refactor(visibility): split _weights into alpha/layer-mask/composite"
```

- [ ] **Step 7: Rewrite `features()` to stop computing "effective" twice when no layers are given**

Read the current `features` method again (it did not change in Step 4, so it is still exactly as it was). Replace:

```python
    def features(self, params, layers=None,
                 transfer_mode=transfer.DEFAULT_TRANSFER_MODE) -> dict:
        """{"vis": {class: float, "other": float}, "bright": {class: float}, "coverage": float}.

        "other" is voxels the label volume (or the intensity fallback)
        assigned to none of the eight classes -- fat, connective tissue,
        partial-volume edges. A transfer function can still put opacity
        there, so it is measured the same way the eight classes are, instead
        of silently vanishing: `goals.distance` charges for it under the same
        keep-tolerance any unmentioned class gets, or a search/policy could
        satisfy "show only X" by rendering an opaque wall of unclassified
        material instead of X (see goals.py's OTHER keep term)."""
        layers = normalize_layers(layers or {})
        weights, luminance, accumulated = self._weights(params, transfer_mode=transfer_mode)
        effective_weights, _, effective_accumulated = self._weights(params, layers, transfer_mode)
        vis, bright = {}, {}
        effective_vis, effective_bright = {}, {}
        for index, name in enumerate(CLASSES):
            masked = torch.where(self._labels == index, weights, torch.zeros(()))
            total = float(masked.sum())
            vis[name] = total / self._rays
            bright[name] = (float((masked * luminance).sum() / total)
                            if total / self._rays > 1e-4 else 0.0)
            effective = torch.where(self._labels == index, effective_weights, torch.zeros(()))
            effective_total = float(effective.sum())
            effective_vis[name] = effective_total / self._rays
            effective_bright[name] = (
                float((effective * luminance).sum() / effective_total)
                if effective_total / self._rays > 1e-4 else 0.0
            )
        other_masked = torch.where(self._labels == -1, weights, torch.zeros(()))
        vis["other"] = float(other_masked.sum()) / self._rays
        effective_vis["other"] = vis["other"]
        effective_bright["other"] = bright.get("other", 0.0)
        coverage = float((accumulated >= COVERAGE_THRESHOLD).to(torch.float32).mean())
        effective_coverage = float((effective_accumulated >= COVERAGE_THRESHOLD).to(torch.float32).mean())
        return {"vis": vis, "bright": bright, "effective_vis": effective_vis,
                "effective_bright": effective_bright,
                "effective_class_contribution": dict(effective_vis),
                "coverage": coverage, "effective_coverage": effective_coverage}
```

with:

```python
    def features(self, params, layers=None,
                 transfer_mode=transfer.DEFAULT_TRANSFER_MODE) -> dict:
        """{"vis": {class: float, "other": float}, "bright": {class: float}, "coverage": float}.

        "other" is voxels the label volume (or the intensity fallback)
        assigned to none of the eight classes -- fat, connective tissue,
        partial-volume edges. A transfer function can still put opacity
        there, so it is measured the same way the eight classes are, instead
        of silently vanishing: `goals.distance` charges for it under the same
        keep-tolerance any unmentioned class gets, or a search/policy could
        satisfy "show only X" by rendering an opaque wall of unclassified
        material instead of X (see goals.py's OTHER keep term).

        When `layers` is None (every non-v8 policy), `effective_*` equals
        the plain fields by construction -- skip computing it a second time
        rather than re-deriving an already-known answer."""
        alpha, luminance = self._alpha_luminance(params, transfer_mode)
        weights, accumulated = self._composite(alpha)
        if layers is None:
            effective_weights, effective_accumulated = weights, accumulated
        else:
            effective_alpha = alpha * self._layer_mask(alpha, normalize_layers(layers))
            effective_weights, effective_accumulated = self._composite(effective_alpha)
        vis, bright = {}, {}
        effective_vis, effective_bright = {}, {}
        for index, name in enumerate(CLASSES):
            masked = torch.where(self._labels == index, weights, torch.zeros(()))
            total = float(masked.sum())
            vis[name] = total / self._rays
            bright[name] = (float((masked * luminance).sum() / total)
                            if total / self._rays > 1e-4 else 0.0)
            if layers is None:
                effective_vis[name] = vis[name]
                effective_bright[name] = bright[name]
            else:
                effective = torch.where(self._labels == index, effective_weights, torch.zeros(()))
                effective_total = float(effective.sum())
                effective_vis[name] = effective_total / self._rays
                effective_bright[name] = (
                    float((effective * luminance).sum() / effective_total)
                    if effective_total / self._rays > 1e-4 else 0.0
                )
        other_masked = torch.where(self._labels == -1, weights, torch.zeros(()))
        vis["other"] = float(other_masked.sum()) / self._rays
        effective_vis["other"] = vis["other"]
        effective_bright["other"] = bright.get("other", 0.0)
        coverage = float((accumulated >= COVERAGE_THRESHOLD).to(torch.float32).mean())
        effective_coverage = (coverage if layers is None else
                              float((effective_accumulated >= COVERAGE_THRESHOLD).to(torch.float32).mean()))
        return {"vis": vis, "bright": bright, "effective_vis": effective_vis,
                "effective_bright": effective_bright,
                "effective_class_contribution": dict(effective_vis),
                "coverage": coverage, "effective_coverage": effective_coverage}
```

- [ ] **Step 8: Run the full visibility test file again**

Run: `.venv/bin/python -m pytest tests/test_visibility.py -v`
Expected: PASS, all tests including `test_no_layers_effective_fields_equal_plain_fields`,
`test_explicit_layer_opacity_populates_effective_fields`,
`test_hidden_foreground_reveals_background_in_effective_visibility`,
`test_features_report_hu_and_effective_layer_contributions_separately`,
`test_layer_isolation_metrics_measure_target_and_cross_class_leakage`, and the
determinism check `assert model.features(params) == model.features(params)`.

- [ ] **Step 9: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS (no regressions anywhere that touches `visibility.features`,
e.g. `tests/test_goals.py`, `tests/test_oneshot_env.py`, `tests/test_oneshot_train.py`)

- [ ] **Step 10: Commit**

```bash
git add visibility.py
git commit -m "perf(visibility): compute features() effective fields only when layers are given"
```

---

### Task 2: Add `VisibilityModel.vis_only()` — a cheap vis-only path

**Files:**
- Modify: `visibility.py` (new method, placed right after `features`)
- Test: `tests/test_visibility.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_visibility.py`, after `test_no_layers_effective_fields_equal_plain_fields`:

```python
def test_vis_only_matches_features_vis_without_layers():
    model = _model(_slab_volume(-1000.0, 900.0))
    params = _params([0.0, 0.9, 0.0, 0.9])
    assert model.vis_only(params) == model.features(params)["vis"]


def test_vis_only_matches_features_effective_vis_with_layers():
    model = _model(_slab_volume(-1000.0, 900.0))
    params = _params([0.0, 0.0, 0.0, 0.9])
    layers = {"skeleton": {"opacity": 0.0}}
    assert model.vis_only(params, layers) == model.features(params, layers)["effective_vis"]
```

- [ ] **Step 2: Run it to confirm it fails (the method doesn't exist yet)**

Run: `.venv/bin/python -m pytest tests/test_visibility.py::test_vis_only_matches_features_vis_without_layers -v`
Expected: FAIL with `AttributeError: 'VisibilityModel' object has no attribute 'vis_only'`

- [ ] **Step 3: Implement `vis_only`, in `visibility.py` right after `features`**

```python
    def vis_only(self, params, layers=None,
                 transfer_mode=transfer.DEFAULT_TRANSFER_MODE) -> dict:
        """Per-class vis only (`{class: float, "other": float}`) -- skips
        the luminance lookup, `bright`, and `coverage` that `features()`
        also computes. For search loops that only ever compare visibility,
        e.g. the hindsight goal sampler in `rl.oneshot_env`."""
        alpha, _ = self._alpha_luminance(params, transfer_mode)
        if layers is not None:
            alpha = alpha * self._layer_mask(alpha, normalize_layers(layers))
        weights, _ = self._composite(alpha)
        vis = {}
        for index, name in enumerate(CLASSES):
            masked = torch.where(self._labels == index, weights, torch.zeros(()))
            vis[name] = float(masked.sum()) / self._rays
        vis["other"] = float(torch.where(self._labels == -1, weights, torch.zeros(())).sum()) / self._rays
        return vis
```

- [ ] **Step 4: Run the tests again to confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_visibility.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add visibility.py tests/test_visibility.py
git commit -m "perf(visibility): add vis_only(), a cheap visibility-only feature path"
```

---

### Task 3: `goals.py` — share the vis-aggregation logic, add `aggregate_vis` and a `vis()` dispatcher

**Files:**
- Modify: `goals.py:68-100` (`aggregate`)
- Test: `tests/test_goals.py`

- [ ] **Step 1: Write the tests**

Add to `tests/test_goals.py`, right after `test_aggregate_keeps_promoted_organs_out_of_soft` (ends at line 96):

```python
def test_aggregate_vis_matches_aggregate_vis_field():
    features = _features(skeleton=(0.3, 0.9), organs=(0.2, 0.4), muscle=(0.6, 0.8), other=0.1)
    assert goals.aggregate_vis(features["vis"]) == goals.aggregate(features)["vis"]


def test_vis_dispatches_to_vis_only_when_available():
    class Model:
        def __init__(self):
            self.calls = []

        def vis_only(self, params, *args):
            self.calls.append(args)
            return {"skeleton": 0.3, "other": 0.0}

    model = Model()
    assert goals.vis(model, "params") == {"skeleton": 0.3, "other": 0.0}
    assert model.calls == [()]


def test_vis_falls_back_to_features_without_vis_only():
    class Model:
        def features(self, params):
            return {"vis": {"skeleton": 0.4}, "bright": {"skeleton": 0.1}, "coverage": 0.2}

    assert goals.vis(Model(), "params") == {"skeleton": 0.4}
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_goals.py::test_aggregate_vis_matches_aggregate_vis_field tests/test_goals.py::test_vis_dispatches_to_vis_only_when_available tests/test_goals.py::test_vis_falls_back_to_features_without_vis_only -v`
Expected: FAIL — `AttributeError: module 'goals' has no attribute 'aggregate_vis'` (and then, once that's added, `has no attribute 'vis'`)

- [ ] **Step 3: Read the current `aggregate` function**

Re-read `goals.py:68-100` for the exact current text.

- [ ] **Step 4: Extract the shared vis logic and add `aggregate_vis`, in `goals.py`**

Replace:

```python
def aggregate(features: dict, active_layers=None) -> dict:
    """Measured classes -> goal classes.

    {"vis": {goal class: float}, "bright": {goal class: float}, "coverage": float}
    Each canonical goal class maps to its matching measured class. Legacy raw
    feature dictionaries using `organs` and `muscle` are read as generic
    `soft`; canonical models use the shared anatomy registry directly.
    """
    source_vis = features.get("effective_vis", features["vis"]) if active_layers is not None else features["vis"]
    source_bright = (features.get("effective_bright", features["bright"])
                     if active_layers is not None else features["bright"])
    vis, bright = {}, {}
    for goal_class in GOAL_CLASSES:
        measured = MEASURED_FOR_GOAL[goal_class]
        # Read old synthetic feature fixtures while canonical models use only
        # registry names. Legacy organs/muscle remain soft, never promoted.
        if goal_class == "soft" and "soft" not in source_vis:
            sources = ("organs", "muscle")
        else:
            sources = measured
        total_vis = sum(source_vis.get(m, 0.0) for m in sources)
        vis[goal_class] = total_vis
        if total_vis > 0.0:
            bright[goal_class] = (sum(source_vis.get(m, 0.0) * source_bright.get(m, 0.0)
                                   for m in sources) / total_vis)
        else:
            bright[goal_class] = 0.0
    # Not a goal class -- no instruction ever names it -- but distance() still
    # needs it to tell "hidden" apart from "hidden behind unlabeled material".
    vis["other"] = source_vis.get("other", 0.0)
    coverage = (features.get("effective_coverage", features["coverage"])
                if active_layers is not None else features["coverage"])
    return {"vis": vis, "bright": bright, "coverage": coverage}
```

with:

```python
def _goal_sources(goal_class: str, source_vis: dict) -> tuple:
    """Measured-class keys backing one goal class, in `source_vis`. Read old
    synthetic feature fixtures while canonical models use only registry
    names: legacy `organs`+`muscle` are read as generic `soft`, never
    promoted, when the canonical `soft` key itself is absent."""
    if goal_class == "soft" and "soft" not in source_vis:
        return ("organs", "muscle")
    return MEASURED_FOR_GOAL[goal_class]


def _aggregate_vis(source_vis: dict) -> dict:
    """Measured-class vis -> goal-class vis. Shared by `aggregate` (which
    also needs `bright`/`coverage`) and `aggregate_vis` (which doesn't)."""
    vis = {goal_class: sum(source_vis.get(m, 0.0) for m in _goal_sources(goal_class, source_vis))
           for goal_class in GOAL_CLASSES}
    # Not a goal class -- no instruction ever names it -- but distance() still
    # needs it to tell "hidden" apart from "hidden behind unlabeled material".
    vis["other"] = source_vis.get("other", 0.0)
    return vis


def aggregate(features: dict, active_layers=None) -> dict:
    """Measured classes -> goal classes.

    {"vis": {goal class: float}, "bright": {goal class: float}, "coverage": float}
    Each canonical goal class maps to its matching measured class. Legacy raw
    feature dictionaries using `organs` and `muscle` are read as generic
    `soft`; canonical models use the shared anatomy registry directly.
    """
    source_vis = features.get("effective_vis", features["vis"]) if active_layers is not None else features["vis"]
    source_bright = (features.get("effective_bright", features["bright"])
                     if active_layers is not None else features["bright"])
    vis = _aggregate_vis(source_vis)
    bright = {}
    for goal_class in GOAL_CLASSES:
        sources = _goal_sources(goal_class, source_vis)
        total_vis = vis[goal_class]
        if total_vis > 0.0:
            bright[goal_class] = (sum(source_vis.get(m, 0.0) * source_bright.get(m, 0.0)
                                   for m in sources) / total_vis)
        else:
            bright[goal_class] = 0.0
    coverage = (features.get("effective_coverage", features["coverage"])
                if active_layers is not None else features["coverage"])
    return {"vis": vis, "bright": bright, "coverage": coverage}


def aggregate_vis(vis: dict) -> dict:
    """Like `aggregate`, but for a vis-only measured-class dict such as
    `visibility.VisibilityModel.vis_only`'s output: no `bright`/`coverage`,
    for callers that never read them (the hindsight visibility search loop
    in `rl.oneshot_env`)."""
    return _aggregate_vis(vis)


def vis(model, params, active_layers=None) -> dict:
    """Per-class visibility only, via the model's cheap `vis_only` when it
    has one (the real `VisibilityModel`); a model without it (test stubs)
    falls back to `features()`, exactly mirroring `aggregate`'s own
    plain-vs-effective selection so the fallback can never disagree with
    what `aggregate(features(...))` would have picked."""
    if not hasattr(model, "vis_only"):
        raw = features(model, params, active_layers)
        return raw.get("effective_vis", raw["vis"]) if active_layers is not None else raw["vis"]
    if active_layers is None:
        return model.vis_only(params)
    try:
        return model.vis_only(params, active_layers)
    except TypeError:
        return model.vis_only(params)
```

- [ ] **Step 5: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_goals.py -v`
Expected: PASS, all tests (including every existing `aggregate` test —
`_aggregate_vis`/`_goal_sources` must reproduce the exact old inline logic)

- [ ] **Step 6: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add goals.py tests/test_goals.py
git commit -m "perf(goals): add aggregate_vis/vis, a cheap vis-only path alongside aggregate"
```

---

### Task 4: Use the cheap vis-only path in the hindsight visibility search loop

**Files:**
- Modify: `rl/oneshot_env.py:321-322` (add `_vis`, next to `_features`)
- Modify: `rl/oneshot_env.py:381-421` (`_sample_hindsight_goal`)
- Test: `tests/test_oneshot_env.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_oneshot_env.py`, right after `test_hindsight_goal_mentions_at_least_one_class` (the file's last test):

```python
def test_hindsight_goal_deltas_match_full_features_path():
    """The hindsight visibility search loop now reads `_vis()` (the cheap
    vis_only path) instead of `_features()` + `goals.aggregate()` -- this
    pins that both paths produce the identical goal for the same seed, i.e.
    the fast path is not just faster but exactly equivalent."""
    env = OneShotEnv(["synthetic"], hindsight_ratio=1.0)
    for seed in range(10):
        env.reset(seed=seed)
        target_params = apply_controllable(env._start_params, env.hindsight_action())
        via_features = goals.aggregate(env._model.features(target_params), env._active_layers)
        for goal_class, target in env._instruction["targets"].items():
            assert target["vis"] == pytest.approx(
                math.log10(via_features["vis"][goal_class] + goals.EPSILON)
                - math.log10(env._start_agg["vis"][goal_class] + goals.EPSILON))
```

- [ ] **Step 2: Run it to confirm it already passes (behavioral pin, not a new feature yet)**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py::test_hindsight_goal_deltas_match_full_features_path -v`
Expected: PASS (today's `_sample_hindsight_goal` already computes exactly
this via the slow path — this test pins the *result*, which Step 4 below
must keep producing from the fast path)

- [ ] **Step 3: Read the current code**

Re-read `rl/oneshot_env.py:321-322` and `rl/oneshot_env.py:381-421` for the exact current text.

- [ ] **Step 4: Add `self._vis()`, right after `self._features()` in `rl/oneshot_env.py`**

Replace:

```python
    def _features(self, params):
        return goals.features(self._model, params, getattr(self, "_active_layers", None))
```

with:

```python
    def _features(self, params):
        return goals.features(self._model, params, getattr(self, "_active_layers", None))

    def _vis(self, params):
        return goals.vis(self._model, params, getattr(self, "_active_layers", None))
```

- [ ] **Step 5: Use it in `_sample_hindsight_goal`, in `rl/oneshot_env.py`**

Replace:

```python
            target_params = apply_controllable(start_params, action)
            target_agg = goals.aggregate(self._features(target_params), self._active_layers)

            deltas = {}
            for goal_class in goals.GOAL_CLASSES:
                start_vis = start_agg["vis"][goal_class]
                target_vis = target_agg["vis"][goal_class]
```

with:

```python
            target_params = apply_controllable(start_params, action)
            target_vis_agg = goals.aggregate_vis(self._vis(target_params))

            deltas = {}
            for goal_class in goals.GOAL_CLASSES:
                start_vis = start_agg["vis"][goal_class]
                target_vis = target_vis_agg[goal_class]
```

- [ ] **Step 6: Run the new test and the full hindsight test block**

Run: `.venv/bin/python -m pytest tests/test_oneshot_env.py -v -k hindsight`
Expected: PASS, all hindsight tests, including:
- `test_hindsight_goal_is_exactly_what_the_target_achieved`
- `test_hindsight_oracle_beats_a_random_action_by_a_clear_margin`
- `test_hindsight_goal_mentions_at_least_one_class`
- `test_hindsight_goal_deltas_match_full_features_path` (new)

- [ ] **Step 7: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add rl/oneshot_env.py tests/test_oneshot_env.py
git commit -m "perf(rl): use the cheap vis-only path in the hindsight visibility search loop"
```

---

### Task 5: Re-measure and record the throughput win

**Files:**
- None in the repo — this re-runs the scratch profiling script used during
  design and reports the new numbers in the PR/commit, per the spec's
  Testing section.

- [ ] **Step 1: Re-run the same profiling script from the design phase**

The script lives at
`/private/tmp/claude-501/-Users-macbook-Documents-Dev-Uni-reinforcement-learning-volume-rendering-transfer-function/2360da7c-4ab3-4ebf-8bca-c0c26beaad9b/scratchpad/profile_hindsight.py`
(not part of the repo). If that scratch directory no longer exists, recreate
it with this content before running:

```python
"""Diagnostic profiling of OneShotEnv.reset() cost, plain vs hindsight."""
import cProfile
import io
import pstats
import sys
import time

sys.path.insert(0, "/Users/macbook/Documents/Dev/Uni/reinforcement-learning-volume-rendering-transfer-function")

import datasets
from rl.oneshot_env import OneShotEnv

VOLUME = datasets.volumes_for_split("train")[0]
N = 60


def time_resets(env, n):
    env.reset(seed=0)
    start = time.perf_counter()
    for i in range(n):
        env.reset(seed=i)
    return time.perf_counter() - start


print(f"volume={VOLUME}")

plain = OneShotEnv([VOLUME], hindsight_ratio=0.0, reward_mode="attainment")
t_plain = time_resets(plain, N)
print(f"hindsight_ratio=0.0: {N / t_plain:.1f} resets/s ({1000 * t_plain / N:.1f} ms/reset)")

hindsight = OneShotEnv([VOLUME], hindsight_ratio=1.0, reward_mode="attainment")
t_hindsight = time_resets(hindsight, N)
print(f"hindsight_ratio=1.0: {N / t_hindsight:.1f} resets/s ({1000 * t_hindsight / N:.1f} ms/reset)")
```

Run: `.venv/bin/python /private/tmp/claude-501/-Users-macbook-Documents-Dev-Uni-reinforcement-learning-volume-rendering-transfer-function/2360da7c-4ab3-4ebf-8bca-c0c26beaad9b/scratchpad/profile_hindsight.py`

- [ ] **Step 2: Compare against the "before" numbers**

Before (recorded in the spec): `hindsight_ratio=0.0` → 16.4 resets/s (61.1
ms/reset); `hindsight_ratio=1.0` → 9.6 resets/s (103.6 ms/reset). Report the
new numbers next to these — expect roughly a 1.5–2x improvement on both,
since Task 1 removes the dominant redundant computation from every
`features()` call regardless of hindsight.

- [ ] **Step 3: Decide on Approach C (deferred in the spec)**

If the new `hindsight_ratio=1.0` number is still far below
`hindsight_ratio=0.0` (i.e. the retry loop itself, not just each
`features()` call's fixed cost, is now the bottleneck), profile *how many*
`vis_only` calls the loop makes per reset on average (count calls inside
`_sample_hindsight_goal`, e.g. with a monkeypatched counter, across a
few hundred resets). Only if that average is high (several, not ~1) does
batching the retry loop (Approach C in the spec) become worth the added
risk — open a new spec for it at that point rather than extending this one.

- [ ] **Step 4: Report the numbers to the user**

No commit for this task — it's a measurement, not a code change. State the
before/after resets/s for both configs, and whether Approach C now looks
warranted per Step 3.
