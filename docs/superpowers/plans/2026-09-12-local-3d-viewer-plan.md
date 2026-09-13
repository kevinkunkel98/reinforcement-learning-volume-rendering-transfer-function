# Local 3D Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local vtk.js volume viewer with typed NRRD-derived transport and renderer-neutral scene logging for web now and `vrui` later.

**Architecture:** Python remains source of truth for dataset loading, orientation, spacing, and transfer-function semantics. Server exposes validated metadata and binary float32 chunks; browser reconstructs the volume and renders locally with vtk.js. Browser scene transitions use a shared JSON contract so future `vrui` can submit and consume the same records.

**Tech Stack:** Python, FastAPI, VTK/NRRD, NumPy, JavaScript, vtk.js, WebGL, pytest.

---

## File Map

- Modify: `datasets.py` to expose normalized dataset metadata and typed volume arrays/chunk descriptors.
- Modify: `server.py` to add dataset metadata/chunk endpoints and scene-transition persistence endpoints without changing existing PNG routes.
- Modify: `requirements.txt` only if a server-side transport dependency is needed; use existing NumPy/VTK first.
- Create: `static/viewer.js` for vtk.js initialization, volume upload, camera, transfer-function, and render state.
- Modify: `static/index.html` to add local-viewer container and fallback toggle.
- Modify: `static/app.js` to load metadata/chunks, coordinate dataset changes, and submit scene transitions.
- Modify: `static/style.css` for viewer layout, loading/error/fallback states, and mobile sizing.
- Create: `scene_schema.py` for shared scene validation/normalization used by server and tests.
- Create: `tests/test_scene_schema.py` for scene records and camera/transfer-function validation.
- Modify: `tests/test_datasets.py` for metadata/chunk reconstruction.
- Modify: `tests/test_server.py` for metadata/chunk/scene endpoints.
- Create: `tests/test_static_viewer_contract.py` for browser asset contract checks that do not require a browser.
- Modify: `README.md` with local viewer and client-neutral scene workflow.

## Task 1: Define Scene Contract

**Files:** `scene_schema.py`, `tests/test_scene_schema.py`

- [ ] **Step 1: Write failing schema tests**

```python
def test_normalize_scene_preserves_goal_camera_and_transfer_function():
    scene = normalize_scene({
        "scene_id": "s:1", "parent_scene_id": "s:0", "session_id": "s",
        "client": "web", "dataset": "ct_cardio", "dataset_version": "sha256:x",
        "volume": {"dimensions": [4, 5, 6], "spacing": [0.7, 0.7, 1.0],
                   "scalar_type": "float32", "orientation": "dataset-normalized"},
        "transfer_function": [0.0] * 24,
        "camera": {"position": [0, 0, 1], "focal_point": [0, 0, 0],
                   "view_up": [0, 1, 0], "zoom": 1.0},
        "goal": {"target": "bone", "direction": "increase"},
        "command": {"attribute": "opacity", "target": "bone", "direction": "increase"},
    })
    assert scene["client"] == "web"
    assert scene["volume"]["dimensions"] == [4, 5, 6]
    assert len(scene["transfer_function"]) == 24

def test_normalize_scene_rejects_bad_volume_and_camera():
    with pytest.raises(ValueError):
        normalize_scene({"volume": {"dimensions": [0, 1, 1]}})
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_scene_schema.py -q`

Expected: FAIL because `scene_schema.py` does not exist.

- [ ] **Step 3: Implement strict normalization**

Add `normalize_scene(record)` validating IDs, client (`web`/`vrui`), dataset version, three positive dimensions, three positive spacing values, `float32` scalar type, 24 finite transfer-function values, camera arrays, finite positive zoom, goal target/direction, and opacity command. Return JSON-serializable normalized values without adding client-specific fields to shared state.

- [ ] **Step 4: Add branch and transition helpers**

Add `scene_transition(before, after, verdict=None, accepted=None, ended=None)` requiring `after.parent_scene_id == before.scene_id` and preserving optional preference fields. Add tests for valid parent links and rejection of mismatched parents.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_scene_schema.py -q`

Expected: PASS.

## Task 2: Dataset Metadata and Binary Chunks

**Files:** `datasets.py`, `tests/test_datasets.py`

- [ ] **Step 1: Write failing metadata/chunk tests**

```python
def test_dataset_metadata_matches_loaded_volume(monkeypatch):
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    monkeypatch.setattr(datasets, "load_dataset", lambda name: (volume, (0.5, 0.6, 0.7)))
    metadata = datasets.dataset_metadata("synthetic")
    assert metadata["dimensions"] == [2, 3, 4]
    assert metadata["spacing"] == [0.5, 0.6, 0.7]
    assert metadata["scalar_type"] == "float32"

def test_chunk_round_trip_reconstructs_float32_volume():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    chunks = list(datasets.iter_volume_chunks(volume, chunk_bytes=16))
    restored = np.frombuffer(b"".join(chunks), dtype=np.float32).reshape(volume.shape)
    np.testing.assert_array_equal(restored, volume)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_datasets.py -q`

Expected: FAIL because metadata/chunk helpers do not exist.

- [ ] **Step 3: Implement normalized metadata**

Add `dataset_metadata(name)` using `load_dataset(name)`, returning dataset name, checksum/version, dimensions, spacing, scalar type, orientation, intensity range, and chunk byte size. Use existing MRI flip/rescale behavior through `load_dataset`; do not duplicate orientation logic.

- [ ] **Step 4: Implement deterministic binary chunking**

Add `iter_volume_chunks(volume, chunk_bytes=8*1024*1024)` over contiguous C-order float32 bytes. Validate volume dtype and chunk size. Add `get_volume_chunk(name, index)` with bounds checks and metadata total count. Never silently cast a non-finite volume.

- [ ] **Step 5: Run dataset tests**

Run: `pytest tests/test_datasets.py -q`

Expected: PASS with existing dataset tests.

## Task 3: Server Dataset Endpoints

**Files:** `server.py`, `tests/test_server.py`

- [ ] **Step 1: Write failing endpoint tests**

Test `GET /api/datasets/{name}/metadata` returns dimensions, spacing, scalar type, version, and chunk descriptors; test `GET /api/datasets/{name}/chunks/{index}` returns `application/octet-stream`; test invalid dataset/chunk returns HTTP 404/400; test scene transition endpoint rejects malformed parent links.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_server.py -q -k 'metadata or chunk or scene'`

Expected: FAIL because endpoints do not exist.

- [ ] **Step 3: Add metadata/chunk routes**

Use FastAPI response types for JSON and `Response(content=chunk, media_type="application/octet-stream")`. Validate dataset names through existing `load_dataset`/registry. Keep current `/api/datasets`, `/api/state`, `/api/command`, PNG image routes, and server session behavior unchanged.

- [ ] **Step 4: Add scene transition route and persistence**

Add `POST /api/scenes/transition` accepting a normalized scene record, validate through `scene_schema`, append JSONL to `out/scene_transitions.jsonl`, and return the normalized record. Do not store binary volume chunks in scene logs. Add client/session fields from request, not server guesses.

- [ ] **Step 5: Run server tests**

Run: `pytest tests/test_server.py -q`

Expected: existing server tests and new endpoint tests pass.

## Task 4: vtk.js Browser Viewer

**Files:** `static/viewer.js`, `static/index.html`, `static/style.css`, `static/app.js`, `tests/test_static_viewer_contract.py`

- [ ] **Step 1: Write browser contract tests**

```python
def test_viewer_assets_define_local_volume_and_scene_hooks():
    viewer = Path("static/viewer.js").read_text()
    app = Path("static/app.js").read_text()
    assert "vtkVolume" in viewer
    assert "fetch" in viewer
    assert "parent_scene_id" in app
    assert "transfer_function" in app
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_static_viewer_contract.py -q`

Expected: FAIL because `static/viewer.js` does not exist and index has no viewer container.

- [ ] **Step 3: Add vtk.js dependency and viewer container**

Pin a vtk.js version in the project’s browser dependency mechanism, add `<div id="local-viewer">`, loading/error state, fallback toggle, and responsive CSS. Do not remove current image elements or controls.

- [ ] **Step 4: Implement volume download/reconstruction**

Implement `LocalVolumeViewer.loadDataset(metadata)` fetching all chunk URLs, validating response byte lengths and total bytes, constructing `Float32Array`, and creating vtk.js image data with dimensions and spacing. Reject incomplete data and expose fallback state.

- [ ] **Step 5: Implement transfer-function and camera adapters**

Implement `setTransferFunction(vector)` for exactly 24 values and `getCameraState`/`setCameraState` using position, focal point, view-up, and zoom. Camera interaction calls local render only; no server frame fetch occurs.

- [ ] **Step 6: Integrate dataset selector and fallback**

Load metadata/chunks after dataset selection, reset scene state, keep current PNG state visible when local viewer is disabled or fails, and expose a clear error rather than showing stale data.

- [ ] **Step 7: Run browser contract tests**

Run: `pytest tests/test_static_viewer_contract.py -q`

Expected: PASS.

## Task 5: Scene Logging and Preference Branches

**Files:** `static/app.js`, `static/viewer.js`, `server.py`, `scene_schema.py`, `tests/test_server.py`, `tests/test_static_viewer_contract.py`

- [ ] **Step 1: Write failing transition logging tests**

Assert each command result contains a unique `scene_id`, previous `parent_scene_id`, dataset/version, 24-value transfer function, camera, goal, command, and `client="web"`. Assert returning to an earlier scene then issuing a command records the correct parent.

- [ ] **Step 2: Implement scene snapshots**

Add browser `currentScene()` and `commitScene(command, goal)` helpers. Snapshot before command, update local transfer function/camera, snapshot after command, and POST one normalized transition. Camera-only changes use the same schema with command metadata indicating camera action or no opacity goal.

- [ ] **Step 3: Implement explicit human verdicts**

Attach `verdict`, `accepted`, and `ended` to transition records when the current UI judge/feedback controls are used. Preserve `client="web"`; future `vrui` can submit the same fields.

- [ ] **Step 4: Test extraction compatibility**

Feed generated scene records into `rl.extract_pairs` fixtures and assert branch/trajectory pairs preserve dataset, camera, goal, and parent metadata. Keep PNG legacy rows supported.

- [ ] **Step 5: Run focused integration tests**

Run: `pytest tests/test_scene_schema.py tests/test_datasets.py tests/test_server.py tests/test_static_viewer_contract.py tests/test_rl_extract_pairs.py -q`

Expected: PASS.

## Task 6: Documentation and Verification

**Files:** `README.md`, tests and implementation files from previous tasks

- [ ] **Step 1: Document local viewer workflow**

Add vtk.js local-rendering behavior, dataset chunk transport, fallback mode, scene JSON contract, `web`/`vrui` client compatibility, and preference collection instructions. State clearly that PNG features are legacy/audit fields, not target representation.

- [ ] **Step 2: Run targeted suite**

Run: `pytest tests/test_scene_schema.py tests/test_datasets.py tests/test_server.py tests/test_static_viewer_contract.py tests/test_rl_extract_pairs.py -q`

Expected: all targeted tests pass.

- [ ] **Step 3: Run full suite**

Run: `pytest -q`

Expected: full suite passes with no regressions in parser, rendering, UI, or RL tests.

- [ ] **Step 4: Verify static and repository quality**

Run: `python3 -m compileall -q .`, `git diff --check`, and `git status --short`. Confirm no volume data, generated chunks, secrets, or runtime logs are committed.
