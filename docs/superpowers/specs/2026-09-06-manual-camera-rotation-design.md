# Manual Camera Rotation — Design

## Goal

Add mouse-driven camera control (drag to rotate, scroll to zoom) to the chat
UI, as a third input modality alongside text and voice — fully independent
of the command parser.

## Context

Camera state (`azimuth`, `elevation`, `zoom`) already lives server-side, one
entry per history step (`Session.history[i]["camera"]`), and is currently
only reachable through the discrete rule/LLM-parsed grammar
(`rotate left|right`, `tilt up|down`, `zoom in|out`, each with a
`slightly|moderately|strongly` step size — see `camera.py::apply_camera_command`).
Rendering is server-side, offscreen VTK (`render.render()` + `render.grab()`)
returning a PNG per request; there is no client-side 3D view.

## Architecture

Mouse drag/scroll sends **raw continuous deltas** directly to a new
endpoint, bypassing the parser and its strength-word grammar entirely (there
is no "slightly" for a mouse gesture). The backend stays stateless across an
in-progress gesture: every request during a drag recomputes from the same
fixed starting camera (the cursor's current committed camera) plus the
**cumulative** delta since the gesture began — never an incremental delta
since the last message. This means no server-side "pending drag" state is
needed, and dropped/reordered frames during a fast drag are harmless (each
request is self-contained).

## Backend changes

### `camera.py`: `apply_camera_delta`

```python
def apply_camera_delta(camera: dict, d_azimuth: float, d_elevation: float, d_zoom_factor: float) -> dict:
    """Continuous analogue of apply_camera_command -- same wrap/clamp rules,
    raw deltas instead of strength buckets. Does not mutate the input dict."""
    camera = dict(camera)
    camera["azimuth"] = (camera["azimuth"] + d_azimuth) % 360.0
    camera["elevation"] = float(np.clip(camera["elevation"] + d_elevation, *ELEVATION_RANGE))
    camera["zoom"] = float(np.clip(camera["zoom"] * d_zoom_factor, *ZOOM_RANGE))
    return camera
```

`d_zoom_factor` is multiplicative (1.0 = no change), matching how
`ZOOM_STEP_FACTOR` already works for the discrete grammar — a scroll gesture
accumulates factors by multiplying them together, not adding.

### `server.py`: `Session.camera_delta`

```python
def camera_delta(self, d_azimuth: float, d_elevation: float, d_zoom_factor: float, commit: bool) -> dict:
    base_camera = self.history[self.cursor]["camera"]
    candidate = apply_camera_delta(base_camera, d_azimuth, d_elevation, d_zoom_factor)

    if not commit:
        image_b64, _, _ = _render_image_b64(self.history[self.cursor]["params"], candidate)
        return {"image_b64": image_b64, "camera": candidate}

    current_params = np.array(self.history[self.cursor]["params"], dtype=np.float64)
    step = _render_step(current_params, "manual rotation",
                         {"camera": {"d_azimuth": d_azimuth, "d_elevation": d_elevation,
                                     "d_zoom_factor": d_zoom_factor}},
                         None, False, self.history[-1]["id"] + 1, self.session_id, candidate)
    self.history = self.history[:self.cursor + 1] + [step]
    self.cursor = len(self.history) - 1
    self.save()
    return self.state()
```

Key properties, both deliberate:

- **`commit=False` never touches `self.history`, `self.cursor`, or disk** —
  no `_save_image_file` call, no `self.save()`. A fast drag can call this
  many times per second without growing history or writing PNGs to
  `out/ui_images/`.
- **`commit=True` behaves exactly like the existing discrete camera-command
  path** in `Session.command()` — same `_render_step` call, same history
  append, same disregard for `self.pending` (a camera change is allowed at
  any time, including while a human judgment is pending, matching the
  already-tested behavior for typed camera commands — see
  `test_camera_change_during_pending_judgment_does_not_corrupt_final_camera`
  in `tests/test_server.py`).

### `server.py`: new route

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

## Frontend changes (`static/app.js`, `index.html`, `style.css`)

- Attach `pointerdown`/`pointermove`/`pointerup` listeners to the rendered
  `<img>`. On `pointerdown`, record the start position and
  `setPointerCapture` so the drag keeps tracking even if the cursor leaves
  the image.
- On `pointermove` while dragging: compute the **cumulative** `(dx, dy)`
  since `pointerdown`, convert to degrees via a fixed sensitivity constant
  (`DRAG_DEGREES_PER_PIXEL`), and throttle outgoing requests to roughly
  8-10/second (track last-sent timestamp; skip intermediate moves faster
  than that). Each response's `image_b64` replaces the displayed image
  directly — this does **not** go through the app's normal state-update path
  (no history list update, no thumbnail feed entry) since nothing has been
  committed yet.
- On `pointerup`: send the final cumulative delta with `commit: true`. The
  response is a full `state()` payload, handled exactly like any other
  `/api/command` response (updates history, thumbnails, etc.).
- On `wheel` over the image: `preventDefault()` (stop page scroll),
  accumulate a multiplicative zoom factor across events, send a
  `commit: false` preview immediately for feedback, and debounce a final
  `commit: true` request ~300ms after the last wheel event (a scroll
  gesture has no natural "release" event, unlike a drag).
- CSS: `cursor: grab` on the image, `cursor: grabbing` while dragging,
  `touch-action: none` so pointer events aren't hijacked by the browser's
  own scroll/zoom gestures on the image element.
- A manual-rotation history step displays with a fixed label (e.g. "manual
  rotation") in the history/thumbnail UI, since its `cmd_dict` shape
  (`{"camera": {"d_azimuth": ..., "d_elevation": ..., "d_zoom_factor": ...}}`)
  differs from the discrete grammar's `{"camera": {"action", "direction", "strength"}}`
  and has no natural short phrase to display.

## What does not change

- The text/voice `rotate|tilt|zoom` grammar, `apply_camera_command`, and
  `_validate_camera_cmd` are untouched — manual rotation is a fully separate,
  additional code path, not a replacement.
- History-step shape is unchanged (`camera` dict, `params`, etc.) — only the
  `cmd_dict`'s internal shape differs for a manual-rotation step, since it
  records continuous deltas instead of a discrete action/direction/strength.

## Testing

Mirrors the existing camera tests in `tests/test_server.py`:

- `camera_delta(..., commit=False)` does not change `history` length or
  `cursor`, and does not write a PNG file.
- `camera_delta(..., commit=True)` appends exactly one history step with the
  candidate camera (azimuth wrapped mod 360, elevation and zoom clamped to
  their existing ranges).
- A large single delta (e.g. `d_azimuth=730`) still wraps/clamps correctly,
  exactly like the discrete grammar's existing behavior.
- `camera_delta(..., commit=True)` still works and does not corrupt
  `self.pending` when a human judgment is currently pending (same property
  already tested for the discrete camera-command path).
- `apply_camera_delta` unit tests (in a new or existing `tests/test_camera.py`
  if one exists, else alongside `apply_camera_command`'s tests): wrap,
  clamp-elevation, clamp-zoom, does-not-mutate-input.

## Explicitly out of scope

- Replacing server-side rendering with a client-side 3D view (vtk.js/WebXR)
  — a much larger, separate effort noted in `docs/architecture.typ` §9's
  roadmap.
- Any change to the RL sub-projects, the parser, or the discrete camera
  command grammar.
- Rate-limiting or magnitude-clamping deltas server-side beyond the existing
  physical wrap/clamp — a runaway client delta is already made safe by
  `apply_camera_delta`'s own bounds, same as the discrete grammar today.
