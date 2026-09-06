# Camera as Controllable State + Voice Commands — Design

## Purpose

This is sub-project #1 of a larger VR roadmap (VR client itself is deferred
until lab hardware is known; see conversation context). It makes the camera
a piece of mutable, versioned session state — like the TF `params` vector
already is — and adds relative voice/text commands to move it
(`rotate left/right`, `tilt up/down`, `zoom in/out`). This is a prerequisite
for both a future native VR client (which needs "camera" to already be a
backend concept, not something baked into one renderer's hardcoded call)
and a future RL viewpoint-optimization agent (sub-project #2, not this one).

## Grounding in existing code

- `render.py::render()` currently hardcodes the camera on every call:
  ```python
  renderer.ResetCamera()
  cam = renderer.GetActiveCamera()
  cam.Azimuth(30)
  cam.Elevation(20)
  renderer.ResetCameraClippingRange()
  ```
  There is no persisted camera state anywhere — every render starts from
  VTK's auto-fit view plus this fixed 30°/20° offset.
- `server.py::Session` stores one dict per history step (via `_render_step`),
  containing `params`, `image_b64`, `masses`, `features`, etc. — camera
  becomes one more field on that same dict, so back/forward navigation
  (`Session.back`/`forward`, already cursor-based) restores the camera view
  for free, the same way it already restores `params`.
- `mvp.py` has an analogous but simpler pattern: `load_state()`/`save_state()`
  persist `params` to `out/state.json` between CLI invocations (no full
  history, just current state) — camera gets a parallel
  `load_camera()`/`save_camera()` pair against a separate JSON file.
- `commands.py::apply_command` is TF-vector-only and stays that way — camera
  commands are dispatched *before* `apply_command` is ever called, in both
  `server.py::Session.command()` and `mvp.py::main()`, via an
  `if "camera" in cmd:` early branch. `apply_command` never sees a
  camera-shaped command.
- The existing search-loop gate (`cmd.get("attribute") == "opacity"`, added
  when fixing the width/brightness/center search-loop bug) already excludes
  camera commands for free — a camera command has no `"attribute"` key at
  all, so `cmd.get("attribute") == "opacity"` is `False` and it never enters
  hill-climbing. No new gating code needed.

## New module: `camera.py`

```python
"""Camera state (azimuth/elevation/zoom) and relative camera commands."""
import numpy as np

DEFAULT_CAMERA = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}

ELEVATION_RANGE = (-85.0, 85.0)
ZOOM_RANGE = (0.3, 4.0)

ROTATE_STEP_DEGREES = {"slightly": 15.0, "moderately": 30.0, "strongly": 60.0}
ZOOM_STEP_FACTOR = {"slightly": 1.15, "moderately": 1.35, "strongly": 1.7}


def apply_camera_command(cam_cmd: dict, camera: dict) -> dict:
    camera = dict(camera)
    action = cam_cmd["action"]
    direction = cam_cmd["direction"]
    strength = cam_cmd.get("strength") or "moderately"

    if action == "rotate":
        delta = ROTATE_STEP_DEGREES[strength]
        sign = 1.0 if direction == "right" else -1.0
        camera["azimuth"] = (camera["azimuth"] + sign * delta) % 360.0
    elif action == "tilt":
        delta = ROTATE_STEP_DEGREES[strength]
        sign = 1.0 if direction == "up" else -1.0
        camera["elevation"] = float(np.clip(camera["elevation"] + sign * delta, *ELEVATION_RANGE))
    elif action == "zoom":
        factor = ZOOM_STEP_FACTOR[strength]
        if direction == "in":
            camera["zoom"] = float(np.clip(camera["zoom"] * factor, *ZOOM_RANGE))
        else:
            camera["zoom"] = float(np.clip(camera["zoom"] / factor, *ZOOM_RANGE))
    return camera
```

Azimuth wraps (mod 360, no natural bound to clamp against). Elevation is
hard-clamped to ±85° to avoid the camera flipping through the poles.
`ROTATE_STEP_DEGREES`/`ZOOM_STEP_FACTOR` use the same three strength *words*
as `STRENGTH_WORDS` everywhere else in the app (for a consistent voice
vocabulary), but their own magnitude tables — `STRENGTH_WORDS`' fractions
are shaped for the asymptotic bounded-`[-1,1]`-value formula used by TF
attributes, which doesn't apply to degrees or a multiplicative zoom factor.

## `render.py` change

```python
def render(volume: np.ndarray, params: np.ndarray, spacing=(1.0, 1.0, 1.0), camera: dict | None = None) -> vtk.vtkRenderWindow:
    ...
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam_state = camera or {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    cam.Azimuth(cam_state["azimuth"])
    cam.Elevation(cam_state["elevation"])
    cam.Zoom(cam_state["zoom"])
    renderer.ResetCameraClippingRange()
    ...
```

`camera=None` defaults to today's exact fixed values, so any existing
caller that doesn't pass `camera` renders identically to before —
regression-safe by construction, the same reasoning already used for the
`apply_command` opacity-path refactor. `cam.Zoom()` is applied after
`Azimuth`/`Elevation` (which rotate around the focal point) and before the
final `ResetCameraClippingRange()` (so clipping planes are computed for the
final, zoomed view).

## `commands.py` grammar

Three new regex checks in `parse_command_rule`, alongside the existing ones:

```python
m = re.search(r"\b(rotate|turn)\s+(left|right)\b", t)
if m:
    strength = "moderately"
    for word in STRENGTH_WORDS:
        if word in t:
            strength = word
    return {"camera": {"action": "rotate", "direction": m.group(2), "strength": strength}}

m = re.search(r"\btilt\s+(up|down)\b", t)
if m:
    strength = "moderately"
    for word in STRENGTH_WORDS:
        if word in t:
            strength = word
    return {"camera": {"action": "tilt", "direction": m.group(1), "strength": strength}}

m = re.search(r"\bzoom\s+(in|out)\b", t)
if m:
    strength = "moderately"
    for word in STRENGTH_WORDS:
        if word in t:
            strength = word
    return {"camera": {"action": "zoom", "direction": m.group(1), "strength": strength}}
```

Camera commands have no tissue target, so unlike every other command shape
in this file, there's nothing to run `_find_tissue` against — the strength
word (if any) is scanned for anywhere in the full text rather than a
captured target substring.

## LLM validator + prompt

```python
VALID_CAMERA_ACTIONS = {
    "rotate": {"left", "right"},
    "tilt": {"up", "down"},
    "zoom": {"in", "out"},
}

def _validate_camera_cmd(obj) -> bool:
    if not isinstance(obj, dict) or set(obj.keys()) != {"camera"}:
        return False
    cam = obj["camera"]
    if not isinstance(cam, dict) or set(cam.keys()) != {"action", "direction", "strength"}:
        return False
    if cam["action"] not in VALID_CAMERA_ACTIONS:
        return False
    if cam["direction"] not in VALID_CAMERA_ACTIONS[cam["action"]]:
        return False
    if cam["strength"] not in VALID_STRENGTHS:
        return False
    return True
```

`_validate_cmd` dispatches to this the same way it already dispatches to
`_validate_set_cmd` for compound sub-commands: a new
`if isinstance(obj, dict) and set(obj.keys()) == {"camera"}: return _validate_camera_cmd(obj)`
branch. `_SYSTEM_PROMPT` gets one new paragraph documenting the `{"camera":
{"action", "direction", "strength"}}` shape as a fully separate top-level
command type (not a variant of the `{"target", "attribute", ...}` schema).

## `evaluate.objective()`

One more early-return, same treatment as width/brightness/center:
```python
if "camera" in cmd:
    return 1
```
No established metric for "is this a better camera angle" — that's
sub-project #2's job, not this one.

## `server.py` integration

- `_render_image_b64(params, camera)` → `render(volume, params, spacing=spacing, camera=camera)`.
- `_render_step(params, cmd_text, cmd_dict, verdict, search, step_id, session_id, camera)` —
  new required `camera` parameter, stored as `"camera": camera` in the
  returned step dict. Every existing call site updates to pass it:
  `_load_or_init`/`switch_dataset` pass `dict(DEFAULT_CAMERA)` (a fresh
  session/dataset starts at the default view); `command()`'s search and
  non-search branches pass the *current* camera (unchanged by a TF
  command); `judge()`'s final step passes the current camera too.
- `Session.command()` gets a new early branch, checked before the existing
  search/non-search logic:
  ```python
  current_camera = dict(self.history[self.cursor].get("camera", DEFAULT_CAMERA))

  if "camera" in cmd:
      new_camera = apply_camera_command(cmd["camera"], current_camera)
      step = _render_step(current_params, text, cmd, None, False,
                           self.history[-1]["id"] + 1, self.session_id, new_camera)
      self.history = self.history[:self.cursor + 1] + [step]
      self.cursor = len(self.history) - 1
      self.save()
      return self.state()
  ```
  `.get("camera", DEFAULT_CAMERA)` makes this backward-compatible with a
  session JSON file saved before this feature existed (old history entries
  simply don't have the key; they fall back to the default view rather than
  crashing or requiring a migration).
- Camera commands do **not** get a `jsonl_append(LOG_PATH, ...)` entry —
  that log's schema is params-before/after diffs for the objective
  evaluator's training-data role, and a camera command never changes
  `params` (`params_before == params_after` always), which would just be
  noise in that log, not useful data.

## `mvp.py` integration

Mirrors the same shape as `load_state()`/`save_state()`:
```python
CAMERA_STATE_PATH = "out/camera_state.json"

def load_camera() -> dict:
    if os.path.exists(CAMERA_STATE_PATH):
        with open(CAMERA_STATE_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_CAMERA)


def save_camera(camera: dict) -> None:
    os.makedirs("out", exist_ok=True)
    with open(CAMERA_STATE_PATH, "w") as f:
        json.dump(camera, f)
```
`save_png(params, path, camera=None)` threads `camera` into its `render()`
call. `main()` loads `camera = load_camera()` alongside `params =
load_state()`, and gets one new early branch (before the existing
`args.learn` branch) checking `if "camera" in cmd:` — applies the command,
saves the new camera state, renders once to `out/camera.png`, prints the
new camera dict, and returns (no TF param diff to log, same reasoning as
`server.py`).

## `COMMAND_REFERENCE` addition

```python
{
    "category": "Camera",
    "examples": ["rotate left", "tilt down", "zoom in"],
    "description": "Adjust the viewing angle or zoom level. Doesn't change the transfer function.",
},
```
`COMMANDS.md` regenerated afterward (existing drift-check test catches
anyone forgetting to).

## Testing

- `tests/test_camera.py`: `apply_camera_command` — rotate left/right moves
  azimuth in the correct direction and wraps past 360°/below 0°; tilt
  up/down moves elevation and clamps at ±85° under repeated application;
  zoom in/out scales correctly and clamps at the configured range under
  repeated application.
- `tests/test_commands.py`: new cases for `parse_command_rule` recognizing
  "rotate left", "turn right", "tilt up", "zoom in" (with and without an
  explicit strength word), producing the correct `{"camera": {...}}` shape.
- `tests/test_commands.py`: `_validate_cmd` accepts a well-formed camera
  command and rejects a malformed one (bad action/direction combination,
  e.g. `{"action": "rotate", "direction": "up", ...}`).
- `tests/test_evaluate.py`: a camera command's `objective()` call always
  returns `1`.
- `tests/test_server.py`: a `Session.command()` call with a camera command
  (a) does not change `params`, (b) does change the resulting step's
  `camera` field, (c) is retrievable via back/forward navigation (i.e. the
  camera value is genuinely per-step, not a single mutable session-wide
  field).
- `render.py`'s existing tests must keep passing unchanged with no
  `camera` argument (regression check for the `camera=None` default).

## Explicitly out of scope

- Targeted framing commands ("show bone from the side") — would need
  locating a named tissue's actual position in the volume (HU-band voxel
  segmentation + centroid), not just reading a TF peak's center. Deferred;
  sub-project #2's RL agent may end up discovering good tissue-relative
  viewpoints without this being hand-coded at all.
- Any camera metric / hill-climbing search / RL training for viewpoint —
  that is sub-project #2, entirely separate.
- Any UI change — the rendered image already reflects the new camera
  angle; no new display element is needed for this piece.
- The actual VR/Vrui client — blocked on hardware information from an
  upcoming lab visit; this sub-project only makes the backend's camera
  concept exist so that client (whenever scoped) has something to drive.
