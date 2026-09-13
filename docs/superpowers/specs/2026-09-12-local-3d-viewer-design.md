# Local 3D Viewer Design

## Goal

Make browser preference collection reflect the project's real target: local
interactive 3D volume rendering that can later share scene state with a `vrui`
client for VR glasses. Replace browser dependence on server-rendered PNG frames
with a local `vtk.js` viewer while preserving the existing VTK/Python path as a
fallback and offline reference renderer.

## Scope

First slice includes:

- Local browser volume rendering with `vtk.js`
- Existing NRRD datasets exposed through server metadata and typed data chunks
- Browser-owned camera and transfer-function updates
- Dataset selection and scene reset
- Renderer-neutral scene-state logging for web and future `vrui` clients
- Human preference/branch metadata compatible with `rl/extract_pairs.py`

Parser, command grammar, Python VTK rendering, existing PNG UI fallback, and
RL training remain unchanged in first slice. VR hardware integration is not part
of first slice; scene records must be usable by `vrui` later.

## Architecture

```text
NRRD registry and cache
        |
        v
server metadata + typed volume chunks
        |
        v
browser vtk.js volume renderer <---- browser camera controls
        |
        +---- transfer-function adapter (shared 24-value vector semantics)
        +---- scene transition logger
        +---- human verdict / branch metadata

future: same scene records and volume contract -> vrui client
```

The browser downloads a volume once per dataset, reconstructs its typed array,
and renders locally. Camera changes and transfer-function changes do not require
server frame-render requests. Server handles dataset transport, session
persistence, canonical scene logs, and preference records.

## Volume Transport

Existing public NRRD files remain source of truth. Server-side dataset metadata
reports dimensions, spacing, scalar type, intensity range, orientation/version,
and chunk descriptors. Volume data is served as typed-array chunks so the
browser does not need NRRD parsing.

Initial transport uses `float32` data matching `datasets.load_dataset()` after
its existing orientation and MRI rescaling rules. Chunk responses are binary;
metadata identifies byte order, scalar type, chunk index, and total byte size.
Client validates dimensions, chunk lengths, and dataset/version before creating
the vtk.js image data. Incomplete or invalid data fails visibly and leaves the
existing server-rendered fallback available.

Transport must not silently change voxel orientation or spacing. The same
dataset metadata is logged with each scene so web and VR renders can be
reproduced.

## Browser Renderer

Use `vtk.js` volume rendering for first slice. The adapter maps current transfer
function control points to vtk.js color and opacity transfer functions. Existing
24-value transfer-function semantics remain authoritative; parser and command
grammar do not change.

Browser scene state owns a renderer-neutral camera representation:

```json
{
  "position": [0.0, 0.0, 1.0],
  "focal_point": [0.0, 0.0, 0.0],
  "view_up": [0.0, 1.0, 0.0],
  "zoom": 1.0
}
```

vtk.js converts this state to its camera object. Future `vrui` converts the same
state to headset pose/controllers. Client-specific render settings may be
added under a separate field and must not replace the shared camera state.

The current PNG path remains selectable during migration. This permits testing
the new viewer against Python VTK renders and avoids blocking the existing web
UI while browser rendering matures.

## Scene Contract

Every committed scene transition uses a renderer-neutral record:

```json
{
  "scene_id": "session-123:step-7",
  "parent_scene_id": "session-123:step-6",
  "session_id": "session-123",
  "client": "web",
  "dataset": "ct_cardio",
  "dataset_version": "sha256:...",
  "volume": {
    "dimensions": [512, 512, 300],
    "spacing": [0.7, 0.7, 1.0],
    "scalar_type": "float32",
    "orientation": "dataset-normalized"
  },
  "transfer_function": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "camera": {
    "position": [0.0, 0.0, 1.0],
    "focal_point": [0.0, 0.0, 0.0],
    "view_up": [0.0, 1.0, 0.0],
    "zoom": 1.0
  },
  "goal": {"target": "bone", "direction": "increase"},
  "command": {"attribute": "opacity", "target": "bone", "direction": "increase"},
  "timestamp": "2026-09-12T00:00:00Z"
}
```

Scene transitions include parent links, enabling branch extraction when a user
returns to an earlier scene and continues from it. A branch is recorded as
explicit metadata, not inferred from similar transfer-function vectors.

Optional fields:

- `features_before` and `features_after` for compatibility with current reward
  training and objective experiments
- `before_image` and `after_image` for debugging/audit only
- `verdict`, `accepted`, and `ended` for preference extraction
- `client_metadata` for web or `vrui` details that do not affect shared state

Training data should prefer scene state and renderer-derived 3D features as they
become available. PNG features remain legacy compatibility fields, not the
long-term visual representation.

## Preference Workflow

The web client records each command result as a scene transition. Human
judgments reference scene IDs and parent IDs, not image filenames. Pair export
resolves both scene states and preserves dataset, camera, transfer function,
goal, client, and optional audit paths.

Web and `vrui` clients write the same canonical JSONL contract. The reward data
pipeline can therefore combine preferences from both clients without changing
the model or extractor interface.

## Validation

Tests must cover:

- Dataset metadata matches Python loader dimensions, spacing, orientation, and
  scalar type
- Chunk reconstruction equals source typed array
- Invalid or incomplete chunks fail without producing a scene
- Transfer-function adapter preserves all 24 control values
- Camera round-trip preserves position, focal point, view-up, and zoom
- Browser scene transitions contain parent links and goal context
- Branch records remain explicit and extractable
- Existing PNG fallback still works
- Web and `vrui` records parse to the same canonical scene schema

Performance acceptance target: after initial volume download and GPU upload,
camera changes render locally without a server request. Dataset switching may
download new chunks and reset scene state.
