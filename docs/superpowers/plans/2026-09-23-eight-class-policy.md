# Eight-Class Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the renderer and one-shot RL policy from four broad anatomical goals to eight TotalSegmentator-backed goals with consistent commands, visibility scoring, caching, and evaluation.

**Architecture:** Define one canonical eight-class registry and use it across TotalSegmentator preprocessing, transfer peaks, visibility aggregation, goals, commands, and baselines. Keep rendering HU-based with fixed peak centres; use segmentation labels for visibility attribution and per-volume reachability. Regenerate versioned labels/caches and train a new policy with 48-value transfer functions, 24-value actions, and 97-value observations.

**Tech Stack:** Python, NumPy, PyTorch, Gymnasium, Stable-Baselines3 SAC, VTK, nibabel, pytest.

---

## File Map

- Create `anatomy.py`: canonical class names, structure rules, peak order, layout version, and shared mappings.
- Modify `tools/select_totalseg.py`: build eight-class label volumes and manifest metadata.
- Modify `totalseg.py`: validate/read label layout metadata.
- Modify `transfer.py`: define eight fixed peaks and 48-value transfer vectors.
- Modify `visibility.py`: use eight measured classes and versioned caches.
- Modify `goals.py`: aggregate eight goals, sample new instructions, and expose reachability.
- Modify `commands.py`: parse new organ vocabulary and map it to canonical goals.
- Modify `scene_schema.py`: validate expanded canonical tissue names.
- Modify `rl/baselines.py`: use shared eight-peak mappings.
- Modify `rl/oneshot_env.py`: expose 97-value observations and 24-value actions.
- Modify `policy.py`, `rl/oneshot_train.py`, `rl/vis_eval.py`, and
  `plots/qualitative.py` for checkpoint layout metadata and dimensions.
- Add focused tests under `tests/` for anatomy, transfer, visibility, goals, commands, and policy dimensions.
- Update `README.md`, `COMMANDS.md`, and relevant docs after behavior is working.

## Task 1: Add Canonical Anatomy Registry

**Files:**
- Create: `anatomy.py`
- Test: `tests/test_anatomy.py`

- [ ] **Step 1: Write failing registry tests**

```python
from anatomy import CLASS_NAMES, CLASS_LAYOUT_VERSION, class_for_structure


def test_eight_class_order_is_stable():
    assert CLASS_NAMES == (
        "skeleton", "lungs", "heart", "vessels",
        "liver", "kidneys", "spleen", "soft",
    )
    assert CLASS_LAYOUT_VERSION == "anatomy-v2"


def test_structure_rules_promote_organs_before_soft():
    assert class_for_structure("liver") == "liver"
    assert class_for_structure("kidney_left") == "kidneys"
    assert class_for_structure("heart") == "heart"
    assert class_for_structure("aorta") == "vessels"
    assert class_for_structure("pancreas") == "soft"
    assert class_for_structure("table") is None
```

- [ ] **Step 2: Run test and verify failure**

Run: `python -m pytest tests/test_anatomy.py -q`

Expected: FAIL because `anatomy.py` does not exist.

- [ ] **Step 3: Implement registry**

Define `CLASS_NAMES`, `CLASS_LAYOUT_VERSION`, `STRUCTURE_RULES`, `class_for_structure()`, `MEASURED_CLASSES` (the eight classes plus `other`), and `PROMOTED_ORGAN_CLASSES`. Keep matching deterministic and preserve the existing skeleton/lung/vessel prefixes. Map all non-promoted recognized organs and muscle to `soft`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_anatomy.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add anatomy.py tests/test_anatomy.py
git commit -m "feat: add canonical anatomy registry"
```

## Task 2: Expand TotalSegmentator Labels and Manifest Versioning

**Files:**
- Modify: `tools/select_totalseg.py`
- Modify: `totalseg.py`
- Test: `tests/test_totalseg.py`

- [ ] **Step 1: Add failing label-layout tests**

Test that `build_label_volume()` writes IDs according to `anatomy.CLASS_NAMES`, promoted organs are not written as `soft`, and `totalseg` rejects a manifest with a missing or incompatible `label_layout_version`.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest tests/test_totalseg.py -q`

Expected: FAIL on new layout assertions.

- [ ] **Step 3: Replace local class rules**

Import the registry in `tools/select_totalseg.py`; remove duplicated `CLASS_NAMES`, `CLASS_RULES`, and `class_for_structure()` definitions. Build label IDs from the shared eight-class order. Add `label_layout_version` to the manifest root and each subject entry.

- [ ] **Step 4: Validate metadata in readers**

Make `totalseg.has_labels()` and `totalseg.load_labels()` verify that a subject manifest uses the current layout version. Raise a clear `ValueError` for stale labels instead of silently loading them.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_totalseg.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/select_totalseg.py totalseg.py tests/test_totalseg.py
git commit -m "feat: version eight-class segmentations"
```

- [ ] **Step 7: Regenerate local manifest when source archive exists**

Run: `python -m tools.select_totalseg`

Expected: 30 subjects selected and `data/totalseg_manifest.json` contains `anatomy-v2`. If the source zip is absent, record the exact download command from the tool's error and continue code validation without fabricating generated data.

## Task 3: Expand Transfer Function to Eight Peaks

**Files:**
- Modify: `transfer.py`
- Test: `tests/test_transfer.py`

- [ ] **Step 1: Add failing dimension and mapping tests**

```python
import numpy as np
import transfer


def test_eight_peak_transfer_shape_and_centres():
    params = transfer.anatomical_params()
    assert transfer.N_PEAKS == 8
    assert params.shape == (48,)
    assert set(transfer.ANATOMICAL_PEAK_INDEX) == {
        "skeleton", "lungs", "heart", "vessels", "liver", "kidneys", "spleen", "soft"
    }
    assert np.all(np.isfinite(params))
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_transfer.py -q`

Expected: FAIL because current transfer function has four peaks.

- [ ] **Step 3: Implement eight peak constants**

Import the canonical registry. Set `N_PEAKS = 8`, add one centre/width/height/colour per class, and make `anatomical_params()` iterate the shared order. Choose initial centres from CT distributions, with organ peaks kept narrow enough to reduce cross-tissue bleed. Keep `default_params()` delegating to `anatomical_params()`.

- [ ] **Step 4: Update transfer tests and round trips**

Assert `vector_to_vtk()` accepts 48 values and `peak_internal()` returns finite values for all eight peaks. Test show-only width capping on skeleton and organ peaks.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_transfer.py tests/test_goals.py -q`

Expected: initial transfer tests PASS; goal tests may remain failing until Task 4.

- [ ] **Step 6: Commit**

```bash
git add transfer.py tests/test_transfer.py
git commit -m "feat: add eight anatomical transfer peaks"
```

## Task 4: Expand Visibility and Goals

**Files:**
- Modify: `visibility.py`
- Modify: `goals.py`
- Test: `tests/test_goals.py`
- Test: `tests/test_visibility.py`

- [ ] **Step 1: Add failing eight-class aggregation tests**

Test that liver, kidneys, spleen, heart, and vessels remain distinct, kidneys combine left/right measured labels through the label builder, promoted organs are not included in `soft`, goal vectors contain 32 values, and `distance()` handles all eight classes plus `other`.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_goals.py tests/test_visibility.py -q`

Expected: FAIL on four-class assumptions and dimensions.

- [ ] **Step 3: Expand visibility classes**

Use the shared measured class order in `visibility.CLASSES`. Update intensity fallback to populate only skeleton, lungs, and soft-compatible classes. Increase cache version and include layout version in cache keys. Ensure `VisibilityModel.features()`, `solo_max()`, and cache serialization handle all eight class names.

- [ ] **Step 4: Expand goal aggregation and vectors**

Set `GOAL_CLASSES` from the registry. Define measured-to-goal mappings, with soft excluding promoted organs and kidneys combining both kidney labels. Generate 32-value goal vectors and iterate all classes in progress, distance, attainment, hindsight, and reachability code.

- [ ] **Step 5: Update per-volume support checks**

Use manifest label presence and `solo_max()` thresholds. For fallback volumes return only supported classes. Keep vessels gated by contrast where required, but allow heart as a separate goal only when its peak is reachable.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_goals.py tests/test_visibility.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add visibility.py goals.py tests/test_goals.py tests/test_visibility.py
git commit -m "feat: score eight anatomical goals"
```

## Task 5: Expand Commands, Scene Schema, and Baselines

**Files:**
- Modify: `commands.py`
- Modify: `scene_schema.py`
- Modify: `rl/baselines.py`
- Test: `tests/test_commands.py`
- Test: `tests/test_transfer.py`

- [ ] **Step 1: Add failing command tests**

Add parser cases for `show more liver`, `hide the kidneys`, `brighten the spleen`, `show heart`, and `reduce aorta`. Assert canonical targets are `liver`, `kidneys`, `spleen`, `heart`, and `vessels`.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_commands.py -q`

Expected: FAIL because new nouns are not canonical.

- [ ] **Step 3: Update parser vocabulary and prompts**

Import the shared registry, add aliases and examples, update the LLM system
prompt class descriptions, and remove private four-class assumptions in
`commands.py`. Preserve existing command normalization and fallback behavior.

- [ ] **Step 4: Expand scene validation**

Make `scene_schema._TISSUES` derive from the canonical registry. Add tests proving new command and goal targets normalize and stale unknown targets remain rejected.

- [ ] **Step 5: Expand baselines**

Make `rl.baselines` derive peak indices and controllable groups from `transfer.N_PEAKS`. Update current executor, occlusion rule, and hill climb to operate on all eight classes without private lists.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_commands.py tests/test_transfer.py tests/test_scene_schema.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add commands.py scene_schema.py rl/baselines.py tests/test_commands.py tests/test_transfer.py tests/test_scene_schema.py
git commit -m "feat: support eight-class commands"
```

## Task 6: Expand One-Shot Policy Space and Callers

**Files:**
- Modify: `rl/oneshot_env.py`
- Modify: `rl/candidates.py`
- Modify: `policy.py`
- Modify: `rl/oneshot_train.py`
- Modify: `rl/vis_eval.py`
- Modify: `plots/qualitative.py`
- Test: `tests/test_oneshot_train.py`

- [ ] **Step 1: Add failing dimension tests**

Assert `OBSERVATION_SIZE == 97`, `ACTION_SIZE == 24`, the observation space has shape `(97,)`, the action space has shape `(24,)`, and `build_observation()` returns 97 finite float32 values.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_oneshot_train.py -q`

Expected: FAIL on current 57/12 dimensions.

- [ ] **Step 3: Expand observation construction**

Build goal, histogram, eight visibility values, eight brightness values, eight solo-max values, 24 controllable start values, and coverage. Keep one shared builder used by training and candidate generation.

- [ ] **Step 4: Expand action application and reset logic**

Derive controllable groups from eight peaks. Preserve RGB offsets when applying each scalar RGB action. Update comments, bounds, and hindsight target sampling to use the dynamic group count.

- [ ] **Step 5: Update policy callers and metadata**

Ensure `rl/candidates.py`, `rl/oneshot_train.py`, and `rl/vis_eval.py` use the
97/24 environment contract. Make `policy.py` select a new checkpoint path and
validate checkpoint metadata before loading. Update `plots/qualitative.py` to
load only the new policy layout and report class metadata. Old checkpoints must
fail with a clear incompatibility error rather than being silently used.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_oneshot_train.py tests/test_candidates.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add rl/oneshot_env.py rl/candidates.py tests/test_oneshot_train.py tests/test_candidates.py
git commit -m "feat: expand one-shot policy spaces"
```

## Task 7: Rebuild Caches and Validate Rendering

**Files:**
- Modify: `tools/validate_visibility.py` if class output is hard-coded.
- Modify: `README.md`, `COMMANDS.md`, and relevant docs.
- Generated: `data/totalseg_manifest.json`, `out/cache/visibility/*` when source data is available.

- [ ] **Step 1: Run fast tests before data work**

Run: `python -m pytest -q -m "not slow"`

Expected: PASS.

- [ ] **Step 2: Regenerate labels and manifest**

Run: `python -m tools.select_totalseg`

Expected: versioned eight-class manifest and labels for all selected subjects.

- [ ] **Step 3: Rebuild visibility caches**

Run: `python -m tools.build_visibility_cache`

Expected: new cache files whose keys include the new cache/layout version; stale caches are not loaded.

- [ ] **Step 4: Validate real renders**

Run: `python -m tools.validate_visibility`

Expected: validation completes and reports per-class correlations for supported classes without treating unsupported fallback classes as populated.

- [ ] **Step 5: Update user documentation**

Document new commands and the approximate HU-based limitation. Update dimensions and data instructions. Do not claim perfect organ isolation.

- [ ] **Step 6: Commit**

```bash
git add README.md COMMANDS.md tools/validate_visibility.py data/totalseg_manifest.json
git commit -m "docs: document eight-class anatomy"
```

## Task 8: Train and Evaluate New Policy

**Files:**
- Modify: `rl/oneshot_train.py`, `rl/vis_eval.py`, `tools/per_class_eval.py`,
  and `plots/qualitative.py` for class metadata and per-class reports.
- Generated: new policy checkpoints and evaluation JSON under `out/rl_v3/`.

- [ ] **Step 1: Run a smoke training job**

Run the repository's existing one-shot training command with a short episode/step budget and the new train split.

Expected: environment initialization succeeds, SAC receives `(97,)` observations and `(24,)` actions, and a checkpoint is written.

- [ ] **Step 2: Run full multi-seed training**

Use the existing reproducible training command for all configured seeds, writing to a new `out/rl_v3/` namespace so v4 artifacts remain historical.

Expected: all seeds complete without dimension or cache mismatch errors.

- [ ] **Step 3: Evaluate held-out subjects**

Run the existing held-out evaluation on TotalSegmentator test subjects and the four out-of-source CT volumes. Include aggregate attainment, reliability, evaluation cost, and per-class metrics with unsupported/unreachable counts.

- [ ] **Step 4: Compare baselines**

Compare new policy against do-nothing, current executor, random, and hill-climb baselines on the same fixed episodes. Report new-class performance separately from legacy classes.

- [ ] **Step 5: Verify policy render colors**

Render candidate outputs for each new supported class and assert RGB is not collapsed to grayscale unless the configured colour itself is grayscale.

- [ ] **Step 6: Commit metadata/scripts if changed**

```bash
git add rl tools plots docs
git commit -m "feat: train eight-class policy"
```

## Final Verification

- [ ] Run: `python -m pytest -q -m "not slow"`
- [ ] Run: `python -m pytest -q`
- [ ] Run: `git diff --check`
- [ ] Confirm `git status --short` contains only intentional generated outputs or unrelated pre-existing user changes.
- [ ] Confirm old checkpoints are not selected by active policy loader.
- [ ] Confirm spec requirements map to Tasks 1 through 8: canonical classes, transfer peaks, visibility/goals, commands, policy dimensions, cache versioning, tests, docs, and evaluation.
