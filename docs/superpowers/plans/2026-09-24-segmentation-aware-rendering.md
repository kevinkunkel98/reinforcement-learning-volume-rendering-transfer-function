# Segmentation-Aware Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic segmentation-aware anatomical layers to server VTK and browser vtk.js rendering while preserving the existing HU transfer function and unlabeled fallback.

**Architecture:** Introduce a validated renderer-neutral anatomy-layer state and versioned uint8 label transport. Keep the existing 48-value HU transfer function unchanged. Server VTK will use labels for exact class contribution; browser support will initially expose label metadata/chunks and apply the same layer state through a label-aware rendering path with fallback when unavailable. Policy improvements remain a separate post-renderer phase.

**Tech Stack:** Python, NumPy, VTK, PyTorch visibility model, FastAPI, JavaScript, vtk.js, pytest.

---

## File Map

- Create `anatomy_layers.py`: canonical layer defaults, schema validation, serialization, renderer version.
- Modify `datasets.py` and `totalseg.py`: expose optional labels and label transport metadata.
- Modify `server.py`: persist layer state, expose label metadata/chunks, apply layer-aware rendering.
- Modify `render.py`: add label-aware VTK rendering while preserving HU-only API.
- Modify `visibility.py`: score effective layer visibility and cache label-aware data.
- Modify `scene_schema.py`: validate optional `anatomy_layers` and label metadata.
- Modify `commands.py`: direct layer-aware commands for supported anatomy classes.
- Modify `static/viewer.js`: fetch/reconstruct labels and apply layer state with fallback.
- Modify `static/app.js`, `static/index.html`, `static/style.css`: anatomical layer controls and state handling.
- Add tests under `tests/` for schema, transport, VTK isolation, API behavior, browser contracts, fallback, and leakage.
- Add policy-training changes only after renderer phase passes; keep outputs under a new policy namespace.

## Task 1: Define Layer Contract

**Files:**
- Create: `anatomy_layers.py`
- Modify: `scene_schema.py`
- Test: `tests/test_anatomy_layers.py`
- Test: `tests/test_scene_schema.py`

- [ ] Write failing tests for eight canonical layer names, defaults, opacity range, RGB range, unknown-key rejection, and missing-layer defaulting.
- [ ] Run `./.venv/bin/pytest -q tests/test_anatomy_layers.py tests/test_scene_schema.py` and confirm failure because the contract does not exist.
- [ ] Implement `LAYER_LAYOUT_VERSION = "anatomy-layers-v1"`, `DEFAULT_LAYERS`, `default_layers()`, and `normalize_layers(value, available_classes=None)`.
- [ ] Require every canonical class in normalized output, preserve defaults for omitted classes, reject non-finite values, reject opacity outside `[0, 1]`, reject RGB outside `[0, 1]`, and reject unknown classes.
- [ ] Add optional `anatomy_layers` and `label_layout` handling to scene normalization without changing valid legacy scenes.
- [ ] Run focused tests; expected PASS.
- [ ] Commit `feat: add anatomy layer contract`.

## Task 2: Label Metadata and Chunk Transport

**Files:**
- Modify: `totalseg.py`
- Modify: `datasets.py`
- Modify: `server.py`
- Modify: `static/viewer.js`
- Test: `tests/test_totalseg.py`
- Test: `tests/test_datasets.py`
- Test: `tests/test_server.py`
- Test: `tests/test_static_viewer_contract.py`

- [ ] Add failing tests for labeled dataset metadata: dimensions, uint8 scalar type, Fortran order, anatomy layout version, class availability, chunk descriptors, and missing-label fallback.
- [ ] Run focused tests and confirm the new metadata/chunk endpoints are absent.
- [ ] Add `label_metadata(name)` and `iter_label_chunks(labels)` using the existing chunk size and Fortran ordering; validate uint8, finite, non-empty, three-dimensional labels.
- [ ] Add `get_label_chunk(name, index)` and a metadata route parallel to volume metadata/chunks. Return a clear 404/availability response for unlabeled datasets.
- [ ] Include label layout version, dimensions, class IDs, and dataset version in metadata and cache keys.
- [ ] Add browser contract checks for label metadata/chunk URLs, uint8 reconstruction, Fortran ordering, and graceful no-label handling.
- [ ] Run focused tests and Ruff; expected PASS.
- [ ] Commit `feat: add anatomical label transport`.

## Task 3: Server VTK Layer Rendering

**Files:**
- Modify: `render.py`
- Modify: `server.py`
- Test: `tests/test_render.py`
- Test: `tests/test_server.py`

- [ ] Write failing synthetic tests: two overlapping HU-identical labeled regions must respond independently to layer opacity; HU-only render must remain unchanged when labels are absent.
- [ ] Run tests and confirm current scalar-only renderer cannot isolate equal-HU labels.
- [ ] Add a label-aware rendering function with signature `render(volume, params, spacing, camera, frame_bounds, labels=None, layers=None)` while preserving old positional callers.
- [ ] Build one anatomical contribution per label class using validated layer opacity/color settings, composite with the existing HU appearance, and ensure label ID `0` remains background/other.
- [ ] Keep label-aware code versioned and explicit; do not interpolate labels. Use nearest-neighbor sampling for any resampled label path.
- [ ] Add leakage tests proving hidden liver does not contribute to the anatomical layer output while an equal-HU neighboring class remains visible when enabled.
- [ ] Run render/server tests; expected PASS.
- [ ] Commit `feat: render anatomical segmentation layers`.

## Task 4: Visibility and Scene Integration

**Files:**
- Modify: `visibility.py`
- Modify: `server.py`
- Modify: `scene_schema.py`
- Modify: `provenance.py`
- Test: `tests/test_visibility.py`
- Test: `tests/test_scene_schema.py`
- Test: `tests/test_server.py`

- [ ] Add failing tests for layer-aware feature output and scene round trips containing `anatomy_layers`.
- [ ] Extend visibility features with effective layer contribution and keep existing HU contribution fields for comparisons.
- [ ] Version visibility caches with layer layout/renderer version and label metadata; stale caches must regenerate or fail clearly.
- [ ] Add `anatomy_layers` to `_render_step`, persisted session state, scene transitions, compare arms, and provenance metadata.
- [ ] Ensure missing layers in old sessions normalize to defaults, while stale label layouts remain rejected.
- [ ] Add per-class isolation and cross-class leakage metrics to validation output.
- [ ] Run focused and full tests; expected PASS.
- [ ] Commit `feat: integrate layer visibility state`.

## Task 5: Layer-Aware Commands and UI Controls

**Files:**
- Modify: `commands.py`
- Modify: `server.py`
- Modify: `static/index.html`
- Modify: `static/app.js`
- Modify: `static/viewer.js`
- Modify: `static/style.css`
- Test: `tests/test_commands.py`
- Test: `tests/test_static_viewer_contract.py`
- Test: `tests/test_server.py`

- [ ] Add failing command tests for `show only liver`, `hide the kidneys`, `show heart and spleen`, and `restore all anatomy` producing layer changes without corrupting the HU vector.
- [ ] Implement deterministic layer commands: positive target opacity increases, hidden target opacity becomes zero, show-only sets selected layers visible and others hidden, and reset restores defaults.
- [ ] Keep exact legacy HU commands working and include layer state in returned state/scene payloads.
- [ ] Add UI controls for each available class: opacity slider, color display/control, disabled state for unsupported classes, and clear label-unavailable text.
- [ ] Browser fetches label metadata/chunks for labeled datasets, reconstructs uint8 Fortran labels, and uses the label-aware path. On failure, retain local HU rendering or PNG fallback.
- [ ] Add browser contract tests for all eight classes, layer controls, label endpoints, and fallback behavior.
- [ ] Run full tests and browser smoke test on labeled and unlabeled datasets.
- [ ] Commit `feat: add anatomical layer controls`.

## Task 6: Browser/Server Render Agreement

**Files:**
- Modify: `tools/validate_visibility.py`
- Create: `tools/compare_layer_renders.py`
- Test: `tests/test_compare_layer_renders.py`

- [ ] Add failing tests for identical layer defaults and class availability across server metadata and browser-facing metadata.
- [ ] Implement a deterministic comparison utility that renders the same labeled volume/layer state through server reference output and browser-compatible sampled output, reporting image difference, per-class visibility difference, and leakage.
- [ ] Define tolerances in the command output and fail nonzero when supported classes exceed them.
- [ ] Validate at least liver, kidneys, spleen, heart, vessels, and skeleton on representative labeled subjects.
- [ ] Run the comparison tool and record results without committing large generated images.
- [ ] Commit `test: validate server browser anatomy layers`.

## Task 7: Policy Improvement After Renderer Validation

**Files:**
- Modify: `goals.py`
- Modify: `rl/oneshot_env.py`
- Modify: `rl/oneshot_train.py`
- Modify: `tools/per_class_eval.py`
- Test: `tests/test_goals.py`
- Test: `tests/test_oneshot_env.py`
- Generated: `out/rl_v4/` ignored policy artifacts.

- [ ] Establish expanded hill-climb baseline on fixed episodes before policy changes.
- [ ] Add class-balanced reachable instruction sampling and configurable `hindsight_ratio`.
- [ ] Add residual-action mode behind an explicit policy version; do not alter the existing v3 checkpoint contract.
- [ ] Reweight reward so target-layer/target-class progress dominates unrelated drift while retaining empty-render protection.
- [ ] Run short experiments with fixed seeds and 10k–30k steps before any long run.
- [ ] Select the best configuration by per-class median attainment and share-positive, not aggregate mean alone.
- [ ] Train new checkpoints under `out/rl_v4/`; report unsupported/unreachable counts and compare against do-nothing, executor, hill-climb, and v3 policy.
- [ ] Commit only code/config/report changes; keep model artifacts ignored.

## Task 8: Full Application Smoke Test

**Files:**
- Modify: none unless verification finds a defect.

- [ ] Run `./.venv/bin/pytest -q` and require zero failures.
- [ ] Run `./.venv/bin/ruff check .` and require clean output.
- [ ] Start server on port 8000 and verify `/api/state` returns 48 HU parameters, layer state, label availability, and current dataset metadata.
- [ ] Test labeled dataset: show liver, hide kidneys, show heart and spleen, inspect render and stats.
- [ ] Test unlabeled dataset: confirm HU-only rendering and explicit unavailable layer state.
- [ ] Test browser label loading, fallback, compare mode, and mobile layout.
- [ ] Run layer render comparison and policy per-class evaluation.
- [ ] Run `git diff --check` on intended files and leave existing documentation changes untouched.

## Task 9: Commit Intended Changes

- [ ] Review `git status --short`; stage only `.gitignore`, implementation files, tests, specs, plans, and intentional reports.
- [ ] Commit `.gitignore` with the implementation changes only if they belong to this work; do not stage `docs/system-documentation.pdf`, `docs/system-documentation.typ`, `.idea/`, or visual companion artifacts.
- [ ] Record final test, smoke-test, renderer-agreement, and policy metrics in the completion summary.
