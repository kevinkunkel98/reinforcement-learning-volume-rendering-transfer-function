# Segmentation-Aware Rendering Design

## Status

Approved design. Implementation not included in this document.

## Goal

Use TotalSegmentator labels to provide exact anatomical layer control in both
server-side VTK renders and the browser vtk.js viewer, while preserving the
existing HU transfer function and unlabeled-dataset fallback.

## Architecture

Add a renderer-neutral `anatomy_layers` record alongside the existing 48-value
HU transfer function. Each canonical class has opacity and RGB settings:

```json
{
  "skeleton": {"opacity": 1.0, "rgb": [0.95, 0.95, 0.90]},
  "lungs": {"opacity": 1.0, "rgb": [0.55, 0.70, 0.95]},
  "heart": {"opacity": 1.0, "rgb": [0.85, 0.25, 0.25]},
  "vessels": {"opacity": 1.0, "rgb": [0.90, 0.45, 0.40]},
  "liver": {"opacity": 1.0, "rgb": [0.75, 0.45, 0.25]},
  "kidneys": {"opacity": 1.0, "rgb": [0.70, 0.40, 0.35]},
  "spleen": {"opacity": 1.0, "rgb": [0.55, 0.30, 0.45]},
  "soft": {"opacity": 1.0, "rgb": [0.85, 0.35, 0.35]}
}
```

The first phase is deterministic. Existing SAC policy action dimensions remain
unchanged. Commands can manipulate anatomical layers directly; policy layer
actions are deferred until renderer correctness is established.

## Data Flow

1. Dataset loading returns CT volume, spacing, and an optional canonical label
   volume.
2. Scene records contain the HU transfer function, `anatomy_layers`, and label
   layout metadata.
3. Server VTK consumes scalar HU and labels.
4. API exposes CT chunks, optional `uint8` label chunks, label metadata, class
   availability, and layer state.
5. Browser requests labels only for labeled datasets, reconstructs them in
   Fortran order, and applies the same layer state.
6. Visibility scoring uses the same canonical class IDs and layer state.

Label transport uses the existing chunk contract with:

```text
scalar_type: uint8
order: F
layout: anatomy-v2
```

Label cache keys include dataset version, label layout, dimensions, spacing, and
renderer-layer version. Unlabeled datasets do not send label data.

## Rendering

### Server VTK

Build label-aware anatomical contributions and composite them with the scalar
HU transfer appearance. Labeled datasets use exact class masks; unlabeled
datasets use the current HU-only path.

### Browser vtk.js

Fetch and reconstruct label chunks using the same dimensions, order, and layout
metadata. Apply anatomical layer settings in the local renderer. If label
loading or WebGL layer rendering fails, preserve the existing local render or
PNG fallback.

Both renderers must agree on class opacity/color semantics within a documented
visibility and image tolerance.

## Policy and Training

Phase 1 does not change the current 24-action SAC policy:

- HU transfer remains policy-controlled.
- Anatomical layer settings are deterministic command controls.
- Layer state and measured layer visibility appear in scenes, candidates, and
  evaluation metadata.

Phase 2, after renderer validation, may add 16 layer actions: opacity and
brightness/color controls for eight classes. It requires a new policy version,
observation/action contract, class-balanced reachable instruction sampling,
hindsight targets, and a new checkpoint namespace.

Layer-aware reward is based on actual composited contribution:

```text
target-layer progress
+ target brightness progress
- unmentioned-layer drift
- HU transfer drift
- empty-render penalty
```

## Validation and Migration

- Validate class names, opacity range, RGB range, label layout, dimensions, and
  layer schema version.
- Add synthetic volume/label tests for exact class isolation and overlap.
- Compare server and browser rendering behavior on labeled volumes.
- Add per-class isolation and cross-class leakage metrics.
- Missing `anatomy_layers` defaults to canonical layer settings.
- Existing 48-value transfer functions remain valid.
- Old four-class scene records remain readable; absent classes are unavailable.
- Stale label versions fail clearly and are never reinterpreted.
- Old policy checkpoints remain HU-only and cannot be treated as layer-aware.

## UI

Add anatomical layer controls beside HU transfer controls. Show class
availability, opacity, and color. Unsupported classes are disabled with a clear
explanation. Existing stats and fallback rendering remain available.

## Scope Boundaries

- First implementation includes deterministic layer rendering and UI controls.
- Policy layer actions are deferred until deterministic rendering passes.
- No new segmentation dataset.
- No silent label interpolation or class-ID reinterpretation.

## Success Criteria

1. “Show liver” isolates liver on labeled CT even when HU overlaps nearby tissue.
2. Server VTK and browser vtk.js use the same label/layer contract.
3. Unlabeled datasets retain current behavior.
4. Existing scenes and transfer functions migrate safely.
5. Tests cover schema, transport, rendering, fallback, isolation, leakage, and
   browser/server agreement.
