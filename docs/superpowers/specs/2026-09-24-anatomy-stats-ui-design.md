# Anatomy Stats UI Design

## Status

Approved design. Implementation not included in this document.

## Goal

Expose all eight anatomical classes in the existing volume-rendering viewer
without overcrowding the render, history, or transfer-function panels.

## Design

Replace the current four-item telemetry strip with grouped anatomy details. All
groups are expanded by default on desktop and mobile.

### Groups

```text
Thoracic   lungs, heart, vessels
Abdominal  liver, kidneys, spleen
Structural skeleton, soft tissue
```

Each class row displays:

- Class label.
- Current visibility percentage.
- Brightness value when available.
- Horizontal visibility bar.
- Muted unavailable state when the class is unsupported or not measurable.
- Accent color matching the anatomical transfer-function peak.

Unsupported classes remain visible in the panel and show `not measurable` or
`--`; they are never silently omitted.

## Layout

- Desktop: grouped sections occupy a responsive two-column stats area.
- Mobile: sections stack into one column and remain expanded.
- Existing render, command history, and transfer-function chart stay in place.
- Compare mode reuses the same grouped class-row concept for visibility and
  brightness values.
- Transfer-function chart exposes all eight peak markers with collision-aware
  labels so dense HU ranges do not create unreadable overlapping text.

## Data Flow

The backend already returns `class_visibility` and `class_brightness` maps. The
frontend will define one shared eight-class metadata structure containing:

- Canonical class name.
- Display label.
- Group.
- Accent color.
- Transfer-function peak color where applicable.

Telemetry rendering, compare bars, and transfer-curve labels consume this
metadata rather than maintaining separate four-class lists.

Values are formatted consistently: visibility as a percentage, brightness as a
numeric value, and missing values as an unavailable state. The UI must tolerate
missing legacy fields without throwing or preventing the volume viewer from
loading.

## Interaction

All groups are expanded by default. Group headers remain semantic sections and
may support native responsive styling, but this change does not require a new
collapse state or persistence layer.

## Validation

Tests must verify:

- All eight classes are represented in the frontend metadata.
- New commands update the corresponding class row.
- Unsupported vessels and other unavailable classes render a clear muted state.
- Compare mode can render all eight classes for visibility and brightness.
- Desktop and mobile layouts do not overflow horizontally.
- Eight transfer-curve peak markers are generated without duplicate class lists.
- Legacy four-class payloads fail gracefully without breaking viewer rendering.

## Scope Limits

- No backend API redesign.
- No changes to segmentation, policy, or renderer behavior.
- No new stats beyond visibility and brightness in this iteration.
