# Anatomy Stats UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-class viewer telemetry with responsive grouped anatomy stats covering all eight classes, compare mode, and transfer-curve labels.

**Architecture:** Add one shared frontend anatomy metadata registry in `static/app.js`. Use it for telemetry rows, compare bars, and transfer-curve peak markers. Keep backend payloads unchanged; missing legacy fields render as unavailable. Replace the compact four-item strip with three always-expanded anatomy groups styled responsively.

**Tech Stack:** Vanilla JavaScript, HTML, CSS, VTK.js, pytest contract tests.

---

## Files

- Modify `static/index.html`: grouped anatomy stats markup.
- Modify `static/app.js`: shared class metadata, telemetry, compare rendering, curve markers.
- Modify `static/style.css`: grouped rows, bars, colors, desktop/mobile layout.
- Modify `tests/test_static_viewer_contract.py`: frontend contract assertions.
- Add or extend JS-free contract tests for metadata and class coverage.

## Task 1: Define Shared Anatomy Metadata

**Files:**
- Modify: `static/app.js`
- Test: `tests/test_static_viewer_contract.py`

- [ ] Add a contract assertion that the frontend defines all eight canonical classes and the three groups.
- [ ] Run `./.venv/bin/pytest -q tests/test_static_viewer_contract.py` and observe failure.
- [ ] Add one `ANATOMY_CLASSES` array with canonical name, display label, group, UI accent, and transfer peak colour metadata.
- [ ] Define `CLASS_ORDER`, `CLASS_CENTER_HU`, and `CLASS_ANATOMICAL_RGB` from that registry instead of separate four-class lists.
- [ ] Run the contract test and Ruff-equivalent checks; expected PASS.
- [ ] Commit `feat: add frontend anatomy metadata`.

## Task 2: Replace Telemetry Strip With Grouped Stats

**Files:**
- Modify: `static/index.html`
- Modify: `static/app.js`
- Modify: `static/style.css`
- Test: `tests/test_static_viewer_contract.py`

- [ ] Add failing assertions for group headings `Thoracic`, `Abdominal`, and `Structural`, and all eight telemetry IDs.
- [ ] Run the contract tests and observe failure against the four-item strip.
- [ ] Replace the four telemetry items with three always-expanded group sections. Each row must contain label, visibility value, brightness value, and a bar element.
- [ ] Implement `updateTelemetry(classVisibility, classBrightness)` from `ANATOMY_CLASSES`; format percentages and show muted `not measurable` for null/missing values.
- [ ] Pass `state.current.class_brightness` from `refresh()` into the telemetry update.
- [ ] Add CSS for two-column desktop grouping, one-column mobile stacking, row bars, unavailable state, and class accents.
- [ ] Run frontend contract tests and full suite; expected PASS.
- [ ] Commit `feat: display grouped anatomy telemetry`.

## Task 3: Reuse Metadata in Compare Mode

**Files:**
- Modify: `static/app.js`
- Modify: `static/style.css`
- Test: `tests/test_static_viewer_contract.py`

- [ ] Add contract assertions that compare rendering iterates the shared anatomy registry and supports both visibility and brightness channels.
- [ ] Run the contract tests and observe failure if compare code still uses a private four-class list.
- [ ] Update `classBars()` and compare rendering to render grouped eight-class rows using `ANATOMY_CLASSES`, preserving unavailable values and current arm labels.
- [ ] Add responsive compare-bar styles that avoid horizontal overflow on mobile.
- [ ] Run full tests and inspect generated compare markup for all eight class names.
- [ ] Commit `feat: expand anatomy compare stats`.

## Task 4: Label Eight Transfer Peaks

**Files:**
- Modify: `static/app.js`
- Modify: `static/style.css`
- Test: `tests/test_static_viewer_contract.py`

- [ ] Add contract assertions for eight peak labels and collision-aware label handling.
- [ ] Run contract tests and observe failure against current marker comments/implementation.
- [ ] Update `drawTfCurve()` to use metadata centres and colours, draw all eight marker ticks, and place labels only when their x-distance from the previous visible label exceeds a responsive spacing threshold.
- [ ] Keep all ticks visible even when labels are suppressed; preserve near-white skeleton outlines.
- [ ] Add CSS/canvas sizing behavior for narrow screens so labels do not force layout overflow.
- [ ] Run full tests and browser smoke check against the running viewer.
- [ ] Commit `feat: label eight transfer peaks`.

## Task 5: End-to-End Verification

**Files:**
- Modify: none unless verification finds a defect.

- [ ] Run `./.venv/bin/pytest -q` and require zero failures.
- [ ] Run `./.venv/bin/ruff check .` and require clean output.
- [ ] Start/restart `server.py` and verify `/api/state` still returns 48 transfer parameters.
- [ ] Open the viewer and verify all three groups are expanded, all eight rows exist, unsupported classes display an unavailable state, and a liver/heart/kidney command updates the matching row.
- [ ] Verify compare mode displays all eight classes and does not overflow at desktop or mobile width.
- [ ] Verify transfer chart shows eight ticks and readable non-overlapping labels where space permits.
- [ ] Run `git diff --check` and confirm unrelated `docs/system-documentation.*` and `.idea/` changes remain untouched.
