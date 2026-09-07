# Manual Camera Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mouse-driven camera control (drag to rotate, scroll to zoom) to the chat UI, as a third input modality alongside text and voice, fully independent of the command parser.

**Architecture:** A new pure function (`camera.apply_camera_delta`) applies continuous azimuth/elevation/zoom deltas with the same wrap/clamp rules as the existing discrete grammar. A new stateless `Session.camera_delta` method and `/api/camera_delta` route render a preview (no history mutation) or commit exactly one new history step, mirroring how a typed camera command already works. The frontend sends pointer-drag and wheel deltas to this endpoint, throttled/debounced, swapping the displayed image directly during the gesture and only updating the full app state on commit.

**Tech Stack:** Python (FastAPI, numpy), vanilla JS (Pointer Events, Wheel Events), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-06-manual-camera-rotation-design.md`

---

### Task 1: `camera.apply_camera_delta`

**Files:**
- Modify: `camera.py` (append after `apply_camera_command`, currently ends at line 44)
- Test: `tests/test_camera.py` (append after the existing tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_camera.py`:

```python
from camera import apply_camera_delta


def test_apply_camera_delta_increases_azimuth():
    camera = dict(DEFAULT_CAMERA)
    after = apply_camera_delta(camera, d_azimuth=15.0, d_elevation=0.0, d_zoom_factor=1.0)
    assert after["azimuth"] == camera["azimuth"] + 15.0


def test_apply_camera_delta_wraps_azimuth():
    camera = {"azimuth": 350.0, "elevation": 0.0, "zoom": 1.0}
    after = apply_camera_delta(camera, d_azimuth=20.0, d_elevation=0.0, d_zoom_factor=1.0)
    assert after["azimuth"] == 10.0  # 350 + 20 = 370 -> wraps to 10


def test_apply_camera_delta_clamps_elevation():
    camera = dict(DEFAULT_CAMERA)
    after = apply_camera_delta(camera, d_azimuth=0.0, d_elevation=500.0, d_zoom_factor=1.0)
    assert after["elevation"] == ELEVATION_RANGE[1]


def test_apply_camera_delta_clamps_zoom():
    camera = dict(DEFAULT_CAMERA)
    after = apply_camera_delta(camera, d_azimuth=0.0, d_elevation=0.0, d_zoom_factor=1000.0)
    assert after["zoom"] == ZOOM_RANGE[1]


def test_apply_camera_delta_does_not_mutate_input():
    camera = dict(DEFAULT_CAMERA)
    original = dict(camera)
    apply_camera_delta(camera, d_azimuth=15.0, d_elevation=5.0, d_zoom_factor=1.1)
    assert camera == original
```

The top of `tests/test_camera.py` already has:
```python
from camera import DEFAULT_CAMERA, ELEVATION_RANGE, ZOOM_RANGE, apply_camera_command
```
Add `apply_camera_delta` to that same import line instead of a separate `from camera import apply_camera_delta` line — i.e. change it to:
```python
from camera import DEFAULT_CAMERA, ELEVATION_RANGE, ZOOM_RANGE, apply_camera_command, apply_camera_delta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_camera.py -v`
Expected: 5 new tests FAIL with `ImportError: cannot import name 'apply_camera_delta'`, the 7 existing tests in the file still pass (they'll also error at collection since the import line itself fails — that's expected and resolves once Step 3 is done).

- [ ] **Step 3: Implement `apply_camera_delta`**

Append to `camera.py` (after `apply_camera_command`, which currently ends at line 44 with `return camera`):

```python


def apply_camera_delta(camera: dict, d_azimuth: float, d_elevation: float, d_zoom_factor: float) -> dict:
    """Continuous analogue of apply_camera_command -- same wrap/clamp rules,
    raw deltas instead of discrete strength buckets. d_zoom_factor is
    multiplicative (1.0 = no change), matching ZOOM_STEP_FACTOR's convention.
    Does not mutate the input `camera` dict."""
    camera = dict(camera)
    camera["azimuth"] = (camera["azimuth"] + d_azimuth) % 360.0
    camera["elevation"] = float(np.clip(camera["elevation"] + d_elevation, *ELEVATION_RANGE))
    camera["zoom"] = float(np.clip(camera["zoom"] * d_zoom_factor, *ZOOM_RANGE))
    return camera
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_camera.py -v`
Expected: 12 passed (7 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add camera.py tests/test_camera.py
git commit -m "Add apply_camera_delta: continuous camera math for manual mouse control"
```

---

### Task 2: `Session.camera_delta` + `/api/camera_delta` route

**Files:**
- Modify: `server.py` (import line 28; new `Session` method after `judge`, currently ending at line 339; new route after the existing `/api/judge` route, currently ending at line 431)
- Test: `tests/test_server.py` (append after the existing camera tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
def test_camera_delta_preview_does_not_change_history():
    session = _fresh_session()
    before_total = session.state()["total"]
    before_cursor = session.cursor
    result = session.camera_delta(30.0, 10.0, 1.2, commit=False)
    assert session.state()["total"] == before_total
    assert session.cursor == before_cursor
    assert "image_b64" in result
    assert result["camera"]["azimuth"] == pytest.approx(60.0)  # DEFAULT_CAMERA azimuth (30.0) + 30.0


def test_camera_delta_preview_does_not_write_image_file():
    session = _fresh_session()
    session_dir = os.path.join("out", "ui_images", session.session_id)
    before_files = set(os.listdir(session_dir)) if os.path.exists(session_dir) else set()
    session.camera_delta(30.0, 10.0, 1.2, commit=False)
    after_files = set(os.listdir(session_dir)) if os.path.exists(session_dir) else set()
    assert after_files == before_files


def test_camera_delta_commit_appends_one_history_step():
    session = _fresh_session()
    before_total = session.state()["total"]
    state = session.camera_delta(30.0, 0.0, 1.0, commit=True)
    assert state["total"] == before_total + 1
    assert state["current"]["camera"]["azimuth"] == pytest.approx(60.0)
    assert state["current"]["cmd_text"] == "manual rotation"


def test_camera_delta_wraps_and_clamps_large_values():
    session = _fresh_session()
    from camera import ELEVATION_RANGE, ZOOM_RANGE
    state = session.camera_delta(730.0, 500.0, 100.0, commit=True)
    camera = state["current"]["camera"]
    assert 0.0 <= camera["azimuth"] < 360.0
    assert ELEVATION_RANGE[0] <= camera["elevation"] <= ELEVATION_RANGE[1]
    assert ZOOM_RANGE[0] <= camera["zoom"] <= ZOOM_RANGE[1]


def test_camera_delta_commit_works_while_judgment_pending():
    session = _fresh_session()
    session.command("increase opacity for bone strongly", parser="rule",
                     search=True, evaluator="human", steps=2)
    assert session.pending is not None
    state = session.camera_delta(30.0, 0.0, 1.0, commit=True)
    assert session.pending is not None  # pending judgment untouched
    assert state["current"]["cmd_text"] == "manual rotation"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k camera_delta`
Expected: FAIL with `AttributeError: 'Session' object has no attribute 'camera_delta'`.

- [ ] **Step 3: Add the import**

In `server.py`, change line 28 from:
```python
from camera import DEFAULT_CAMERA, apply_camera_command
```
to:
```python
from camera import DEFAULT_CAMERA, apply_camera_command, apply_camera_delta
```

- [ ] **Step 4: Implement `Session.camera_delta`**

In `server.py`, insert this new method into the `Session` class, immediately after the `judge` method (which currently ends at line 339 with `return self.state()`, followed by two blank lines and then `UI_SESSION_PATH = ...` at line 342):

```python
    def camera_delta(self, d_azimuth: float, d_elevation: float, d_zoom_factor: float, commit: bool) -> dict:
        current_step = self.history[self.cursor]
        candidate = apply_camera_delta(current_step["camera"], d_azimuth, d_elevation, d_zoom_factor)

        if not commit:
            params = np.array(current_step["params"], dtype=np.float64)
            image_b64, _, _ = _render_image_b64(params, candidate)
            return {"image_b64": image_b64, "camera": candidate}

        current_params = np.array(current_step["params"], dtype=np.float64)
        cmd_dict = {"camera": {"d_azimuth": d_azimuth, "d_elevation": d_elevation, "d_zoom_factor": d_zoom_factor}}
        step = _render_step(current_params, "manual rotation", cmd_dict, None, False,
                             self.history[-1]["id"] + 1, self.session_id, candidate)
        self.history = self.history[:self.cursor + 1] + [step]
        self.cursor = len(self.history) - 1
        self.save()
        return self.state()
```

Note the deliberate asymmetry: the `commit=False` branch never calls `self.save()`, never appends to `self.history`, and never writes a PNG to disk (it calls `_render_image_b64` directly, not `_render_step`, which is what triggers `_save_image_file`) — a fast drag can call this many times per second with no history growth and no disk writes. The `commit=True` branch mirrors the existing discrete camera-command path in `command()` exactly (same `_render_step` call shape, same history-append pattern, same disregard for `self.pending`).

- [ ] **Step 5: Add the route**

In `server.py`, insert this immediately after the `/api/judge` route (which currently ends at line 431 with `raise HTTPException(status_code=400, detail=str(exc))`), before the blank lines leading into `/api/transcribe`:

```python


class CameraDeltaRequest(BaseModel):
    d_azimuth: float = 0.0
    d_elevation: float = 0.0
    d_zoom_factor: float = 1.0
    commit: bool = False


@app.post("/api/camera_delta")
async def camera_delta(req: CameraDeltaRequest):
    return session.camera_delta(req.d_azimuth, req.d_elevation, req.d_zoom_factor, req.commit)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py -v -k camera_delta`
Expected: 5 passed.

- [ ] **Step 7: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 10 from 156 (5 from Task 1 + 5 from this task) — 166 passed, 2 deselected.

- [ ] **Step 8: Manual smoke test of the new endpoint**

Start the server (`.venv/bin/python server.py`) and in another terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/api/camera_delta -H "Content-Type: application/json" \
  -d '{"d_azimuth": 20, "d_elevation": 0, "d_zoom_factor": 1.0, "commit": false}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('camera:', d['camera']); print('has image:', bool(d.get('image_b64')))"

curl -s http://127.0.0.1:8000/api/state | python3 -c "import json,sys; print('total after preview:', json.load(sys.stdin)['total'])"
```

Expected: the preview call returns a camera dict with azimuth shifted by 20 and a non-empty image; `total` from `/api/state` is unchanged (the preview did not append to history). Stop the server afterward.

- [ ] **Step 9: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "Add Session.camera_delta and /api/camera_delta for stateless preview/commit camera control"
```

---

### Task 3: Frontend — drag to rotate, scroll to zoom

**Files:**
- Modify: `static/app.js` (append new section at the end of the file, after the existing command-reference modal code)
- Modify: `static/style.css` (add rules for the draggable image, in the `#current-image` block)

This task has no automated test (this project has no JS test framework — every existing frontend feature, e.g. the command-reference dialog, was verified manually in the browser). Steps below are implementation + a manual verification checklist.

- [ ] **Step 1: Add CSS for the draggable image**

In `static/style.css`, find the existing `#current-image` rule:

```css
#current-image {
  max-width: 100%;
  max-height: 78vh;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: #000;
  display: block;
}
```

Replace it with:

```css
#current-image {
  max-width: 100%;
  max-height: 78vh;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: #000;
  display: block;
  cursor: grab;
  touch-action: none;
}

#current-image.dragging {
  cursor: grabbing;
}
```

- [ ] **Step 2: Add the drag-to-rotate and scroll-to-zoom handlers**

Append this new section to the end of `static/app.js` (after the existing `el("commands-modal").addEventListener(...)` block, before the final `updateSendState(); loadDatasets(); loadState();` lines):

```javascript
// ---------- manual camera rotation (drag) + zoom (scroll) ----------

const DRAG_DEGREES_PER_PIXEL = 0.4;
const DRAG_THROTTLE_MS = 110;
const WHEEL_ZOOM_SENSITIVITY = 0.0015;
const WHEEL_COMMIT_DEBOUNCE_MS = 300;

const currentImage = el("current-image");

async function sendCameraDelta(dAzimuth, dElevation, dZoomFactor, commit) {
  const r = await fetch("/api/camera_delta", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ d_azimuth: dAzimuth, d_elevation: dElevation, d_zoom_factor: dZoomFactor, commit }),
  });
  return r.json();
}

let dragging = false;
let dragStartX = 0;
let dragStartY = 0;
let lastDragSendAt = 0;

currentImage.addEventListener("pointerdown", (e) => {
  if (state.pending) return;
  dragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  lastDragSendAt = 0;
  currentImage.setPointerCapture(e.pointerId);
  currentImage.classList.add("dragging");
});

currentImage.addEventListener("pointermove", async (e) => {
  if (!dragging) return;
  const now = performance.now();
  if (now - lastDragSendAt < DRAG_THROTTLE_MS) return;
  lastDragSendAt = now;
  const dAzimuth = (e.clientX - dragStartX) * DRAG_DEGREES_PER_PIXEL;
  const dElevation = -(e.clientY - dragStartY) * DRAG_DEGREES_PER_PIXEL;
  const data = await sendCameraDelta(dAzimuth, dElevation, 1.0, false);
  currentImage.src = `data:image/png;base64,${data.image_b64}`;
});

currentImage.addEventListener("pointerup", async (e) => {
  if (!dragging) return;
  dragging = false;
  currentImage.classList.remove("dragging");
  const dAzimuth = (e.clientX - dragStartX) * DRAG_DEGREES_PER_PIXEL;
  const dElevation = -(e.clientY - dragStartY) * DRAG_DEGREES_PER_PIXEL;
  const data = await sendCameraDelta(dAzimuth, dElevation, 1.0, true);
  await refresh(data);
  appendMessage(data.current);
});

let wheelZoomAccum = 1.0;
let wheelCommitTimer = null;

currentImage.addEventListener("wheel", async (e) => {
  if (state.pending) return;
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * WHEEL_ZOOM_SENSITIVITY);
  wheelZoomAccum *= factor;
  const data = await sendCameraDelta(0.0, 0.0, wheelZoomAccum, false);
  currentImage.src = `data:image/png;base64,${data.image_b64}`;

  if (wheelCommitTimer) clearTimeout(wheelCommitTimer);
  wheelCommitTimer = setTimeout(async () => {
    const finalZoom = wheelZoomAccum;
    wheelZoomAccum = 1.0;
    const finalData = await sendCameraDelta(0.0, 0.0, finalZoom, true);
    await refresh(finalData);
    appendMessage(finalData.current);
  }, WHEEL_COMMIT_DEBOUNCE_MS);
}, { passive: false });
```

Notes on why this is shaped this way (for the implementer's understanding, not to be added as code comments beyond what's shown above):
- `dAzimuth`/`dElevation` in both `pointermove` and `pointerup` are computed from `dragStartX`/`dragStartY` (the position when the drag began), **not** from the previous frame's position — this is the cumulative-since-gesture-start design from the spec, which keeps the backend stateless (every request recomputes from the same fixed starting camera).
- `wheelZoomAccum` is reset to `1.0` only once the debounced commit actually fires (captured into `finalZoom` first) — if the user keeps scrolling, each new wheel event keeps multiplying into the same accumulator rather than resetting, so the running preview always reflects the total zoom since the last commit.
- `{ passive: false }` on the wheel listener is required for `e.preventDefault()` to actually stop the page from scrolling.
- Preview responses (`commit: false`) only ever touch `currentImage.src` directly — they never call `refresh()`, since `refresh()` expects a full `state()`-shaped payload and a preview response has no `cursor`/`total`/`history` fields.

- [ ] **Step 3: Manual verification**

Start the server:
```bash
.venv/bin/python server.py
```

Open `http://127.0.0.1:8000` in a browser and verify:
1. Hovering the rendered image shows a grab-hand cursor.
2. Click-dragging left/right rotates the volume smoothly (image updates live during the drag); dragging up/down tilts it.
3. Releasing the mouse leaves the rotated view in place, and a new "manual rotation" message appears in the chat history on the right.
4. Scrolling over the image zooms in/out live; after you stop scrolling for a moment, a "manual rotation" message appears in the chat history.
5. Click the back button (`<`) after a manual rotation — the view reverts to the previous camera; forward (`>`) restores the rotated view.
6. Start a hill-climb search with the human evaluator (toggle "search" on, evaluator "human", send an opacity command) so the Better/Worse judge view is showing, then confirm dragging/scrolling on the image does nothing while the judge view is active (the `#current-image` element is hidden in that view, so pointer/wheel handlers simply don't fire — nothing to click on).
7. Confirm typed commands (`rotate right`, `zoom in`) still work exactly as before, appearing in history the same way they always have.

- [ ] **Step 4: Commit**

```bash
git add static/app.js static/style.css
git commit -m "Add drag-to-rotate and scroll-to-zoom manual camera control to the chat UI"
```

---

## Notes for the implementer

- Do not modify `commands.py`, `evaluate.py`, `mvp.py`, or the RL modules — this plan only touches `camera.py`, `server.py`, and the frontend.
- The discrete text/voice camera grammar (`apply_camera_command`, `_validate_camera_cmd`, the rule/LLM parser's camera branches) is untouched and must continue to work exactly as before — manual rotation is a fully separate, additional code path.
- There is no automated frontend test suite in this project; Task 3's verification is manual, following the existing precedent set by every other UI feature in this codebase.
