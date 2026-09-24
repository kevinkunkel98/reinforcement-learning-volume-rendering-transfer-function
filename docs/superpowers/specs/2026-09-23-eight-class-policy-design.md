# Eight-Class Anatomical Policy Design

## Status

Approved design. Implementation not included in this document.

## Goal

Expand the current TotalSegmentator-based rendering and one-shot RL policy from
four broad goal classes to eight medically meaningful classes. The system should
support richer commands and policy actions while preserving the existing fast,
HU-based transfer-function renderer.

## Scope

### In scope

- Promote selected TotalSegmentator structures into eight anatomical classes.
- Add command vocabulary for heart, vessels, liver, kidneys, and spleen.
- Expand the fixed-centre transfer function from four anatomical peaks to eight.
- Expand the one-shot policy action and observation spaces accordingly.
- Update visibility aggregation, goal sampling, baselines, caches, tests, and
  evaluation outputs.
- Retrain the policy from scratch with the expanded spaces.

### Out of scope

- New datasets or modalities.
- Per-voxel segmentation-driven opacity rendering.
- Perfect anatomical isolation when tissues overlap in HU space.
- Compatibility with old policy checkpoints. Existing checkpoints are kept as
  historical artifacts but are not loadable by the new policy architecture.

## Anatomical Classes

The canonical goal classes become:

```text
skeleton  ribs, vertebrae, skull, limbs, pelvis
lungs    left and right lungs
heart    heart and atrial appendages
vessels  aorta, vena cava, pulmonary and other selected vessels
liver    liver
kidneys  left and right kidneys combined
spleen   spleen
soft     muscle and organs not promoted to a dedicated class
```

Unmatched voxels remain `other` and continue to receive a keep penalty so a
policy cannot satisfy a show-only goal with unclassified material.

The TotalSegmentator preprocessing manifest must use explicit label IDs for the
eight classes. Structure-to-class rules must be deterministic, and overlap
precedence must be documented and tested. Existing manifests must either be
regenerated or rejected with a clear version mismatch; silently interpreting old
five-class label IDs as eight-class IDs is forbidden.

Datasets without anatomical labels retain the intensity fallback. They support
only the classes that can be estimated from HU bands, normally skeleton, lungs,
and soft. Unsupported classes are excluded from instruction sampling and policy
goals for those volumes.

## Transfer Function

The transfer function will contain eight fixed-centre peaks, ordered by the
canonical class-to-peak mapping:

```text
lungs, soft, liver, kidneys, spleen, heart, vessels, skeleton
```

Each peak retains six normalized parameters: centre, width, height, and RGB.
Centres remain fixed during policy execution; width, height, and colour remain
controllable. The vector therefore grows from 24 to 48 values, and the policy
action grows from 12 to 24 scalar groups.

Peak centres and widths must be calibrated against the selected CT volumes.
They are approximate HU bands, not anatomical guarantees. The implementation
must record the chosen values in one source of truth used by transfer-function
construction, visibility probes, goals, and baselines.

Show-only behavior continues to cap wide peaks to reduce neighboring tissue
bleed. The cap must be applied consistently to all applicable organ peaks and
tested against the existing skeleton behavior.

## Goals and Visibility

Visibility continues to attribute rendered contribution using segmentation
labels when available and intensity classes otherwise. The measured class set
and label mapping expand to include the eight canonical goals plus `other`.

Goal aggregation maps measured labels to canonical goals. `soft` combines
muscle and only labels not promoted to `heart`, `liver`, `kidneys`, or `spleen`;
promoted labels must not be double-counted as soft. `kidneys` combines left and
right kidney labels.

Goal vectors expand from 16 values to 32 values:

```text
8 visibility deltas
8 visibility-mentioned flags
8 brightness deltas
8 brightness-mentioned flags
```

Distance, attainment, keep penalties, hindsight goals, and instruction
sampling must iterate over the canonical class list rather than hard-coded four

Instruction sampling must require both label presence and practical
reachability. A class is reachable only when its peak can produce a visibility
ceiling above the configured threshold on that volume. This avoids training or
evaluation examples such as a spleen instruction on a scan without visible
spleen.

## Commands and Baselines

The parser must recognize canonical names and useful synonyms:

- `heart`: heart, cardiac
- `vessels`: vessel, vessels, blood vessel, aorta, vena cava
- `liver`: liver
- `kidneys`: kidney, kidneys, renal
- `spleen`: spleen

Existing skeleton, lungs, soft tissue, and vessel phrases remain supported.
Narrow structure synonyms such as `aorta` map to the `vessels` goal unless a
future design introduces a separate aorta class.

Command normalization, scene schema validation, exact execution, current
executor, occlusion baseline, hill climb, and random baseline must use the same
canonical class and peak mappings. No caller may maintain a private class list.

## Policy Observation and Training

The one-shot observation becomes 97 values:

```text
goal vector              32
histogram                16
start visibility          8
start brightness          8
solo-max ceilings         8
start controllable params 24
coverage                  1
```

The action becomes 24 values in `[-1, 1]`, one value for each width, height,
and RGB group across eight peaks. RGB offsets must be preserved when applying a
scalar brightness action; policy renders must not collapse to grayscale.

The new policy is trained from scratch. Old checkpoints are incompatible and
must not be loaded under the new observation/action dimensions. Training,
evaluation, candidate generation, and policy metadata must report the class
layout and space dimensions.

Instruction mixing must give new organ classes enough exposure while retaining
common skeleton, lungs, and soft-tissue tasks. Per-class coverage and
reachability statistics must be recorded so sparse-organ performance is visible
instead of hidden by aggregate attainment.

## Cache and Data Versioning

Visibility cache version must change when class IDs or transfer peak layout
changes. Cache keys must prevent an old five-class cache from loading into the
new model. The regenerated manifest and label volumes must carry a layout
version or equivalent metadata.

Old policy checkpoints and old visibility caches may remain on disk for
historical comparisons, but the active code must select only matching versions.

## Testing and Validation

Add or update tests for:

- Structure-to-class mapping and deterministic overlap precedence.
- Eight-class label IDs and manifest layout-version validation.
- Fallback behavior for unlabeled CT datasets.
- Eight-peak transfer vectors, fixed centres, and parameter round trips.
- Goal aggregation, vector packing, distance, attainment, and reachability for
  all eight classes.
- Parser synonyms and exact command execution for new classes.
- Baselines using the expanded peak mapping.
- 97-value observations and 24-value actions.
- RGB offset preservation.
- Cache invalidation across label and peak-layout versions.
- Per-class evaluation output and held-out reporting.

Before retraining, run the existing fast test suite and a visibility validation
against real renders. After retraining, evaluate held-out TotalSegmentator
subjects and the existing out-of-source CT set with aggregate and per-class
metrics. Report reliability, median attainment, evaluation cost, and cases where
the requested class was unsupported or unreachable.

## Success Criteria

The work is successful when:

1. Eight canonical classes flow consistently from TotalSegmentator labels
   through visibility, goals, commands, baselines, and policy inputs/outputs.
2. New organ commands produce valid, non-grey rendered candidates on supported
   scans.
3. Unsupported or unreachable classes are not sampled as valid training goals.
4. The retrained policy improves over do-nothing and current-executor baselines
   on reachable new-class tasks.
5. Existing four-class commands remain functional on supported legacy datasets.
6. Tests, cache versioning, and evaluation metadata prevent silent mixing of
   old and new model layouts.
