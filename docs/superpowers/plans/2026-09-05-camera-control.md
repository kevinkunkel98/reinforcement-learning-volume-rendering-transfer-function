# Camera Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the camera (azimuth/elevation/zoom) versioned session state, controllable via relative voice/text commands (rotate/tilt/zoom), as a prerequisite for both a future VR client and a future RL viewpoint-optimization agent.

**Architecture:** A new standalone `camera.py` module (state dict + pure command-application function) is threaded through `render.py` (as an optional parameter, backward-compatible default), `commands.py` (new grammar + LLM schema), `evaluate.py` (always-accept, same as other non-opacity attributes), `server.py` (per-history-step camera state), and `mvp.py` (a parallel simple JSON-file persistence, mirroring how `params` already works there).

**Tech Stack:** Existing stack only — no new dependencies.

Spec: `docs/superpowers/specs/2026-09-05-camera-control-design.md`

---

## File Structure

- Create: `camera.py` — camera state dict, `apply_camera_command`.
- Modify: `render.py` — accept an optional `camera` parameter.
- Modify: `commands.py` — rule-parser grammar, LLM validator/prompt, `COMMAND_REFERENCE`.
- Modify: `evaluate.py` — always-accept camera commands in `objective()`.
- Modify: `server.py` — thread camera through `Session`.
- Modify: `mvp.py` — parallel camera persistence + command handling.
- Modify: `COMMANDS.md` — regenerated from the updated `COMMAND_REFERENCE`.
- Create: `tests/test_camera.py`.
- Modify: `tests/test_render.py`, `tests/test_commands.py`, `tests/test_evaluate.py`, `tests/test_server.py`.

---

### Task 1: `camera.py` module

**Files:**
- Create: `camera.py`
- Create: `tests/test_camera.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_camera.py`:

```python
"""Tests for camera.py -- pure camera-state math, no rendering involved."""
from camera import DEFAULT_CAMERA, ELEVATION_RANGE, ZOOM_RANGE, apply_camera_command


def test_rotate_right_increases_azimuth():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "rotate", "direction": "right", "strength": "moderately"}
    after = apply_camera_command(cmd, camera)
    assert after["azimuth"] > camera["azimuth"]


def test_rotate_left_decreases_azimuth_and_wraps():
    camera = {"azimuth": 10.0, "elevation": 0.0, "zoom": 1.0}
    cmd = {"action": "rotate", "direction": "left", "strength": "strongly"}
    after = apply_camera_command(cmd, camera)
    # 10 - 60 = -50, wrapped into [0, 360)
    assert after["azimuth"] == 310.0


def test_tilt_up_increases_elevation_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "tilt", "direction": "up", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["elevation"] <= ELEVATION_RANGE[1]


def test_tilt_down_decreases_elevation_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "tilt", "direction": "down", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["elevation"] >= ELEVATION_RANGE[0]


def test_zoom_in_increases_zoom_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "zoom", "direction": "in", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["zoom"] <= ZOOM_RANGE[1]


def test_zoom_out_decreases_zoom_and_clamps():
    camera = dict(DEFAULT_CAMERA)
    cmd = {"action": "zoom", "direction": "out", "strength": "strongly"}
    for _ in range(10):
        camera = apply_camera_command(cmd, camera)
    assert camera["zoom"] >= ZOOM_RANGE[0]


def test_apply_camera_command_does_not_mutate_input():
    camera = dict(DEFAULT_CAMERA)
    original = dict(camera)
    apply_camera_command({"action": "rotate", "direction": "right", "strength": "moderately"}, camera)
    assert camera == original
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_camera.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'camera'`

- [ ] **Step 3: Implement `camera.py`**

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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_camera.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 8 from the current baseline.

- [ ] **Step 6: Commit**

```bash
git add camera.py tests/test_camera.py
git commit -m "Add camera state module with relative rotate/tilt/zoom commands"
```

---

### Task 2: `render.py` accepts an optional camera parameter

**Files:**
- Modify: `render.py`
- Modify: `tests/test_render.py`

Current `render()` (relevant excerpt):
```python
def render(volume: np.ndarray, params: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> vtk.vtkRenderWindow:
    ...
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(20)
    renderer.ResetCameraClippingRange()

    win.Render()
    return win
```

- [ ] **Step 1: Write the failing test**

`tests/test_render.py` already imports `from phantom import build_phantom`
and `from transfer import default_params` and builds volumes via
`build_phantom(size=48)` (see its existing tests for the exact pattern).
Append:

```python
def test_render_accepts_custom_camera_and_defaults_match_old_behavior():
    volume = build_phantom(size=48)
    params = default_params()

    win_default = render(volume, params)
    win_custom = render(volume, params, camera={"azimuth": 90.0, "elevation": 0.0, "zoom": 1.0})

    img_default = grab(win_default)
    img_custom = grab(win_custom)
    assert not np.array_equal(img_default, img_custom)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_render.py -k custom_camera -v`
Expected: FAIL with `TypeError: render() got an unexpected keyword argument 'camera'`

- [ ] **Step 3: Implement the change**

Edit `render.py`'s `render()` signature and camera setup:

```python
def render(volume: np.ndarray, params: np.ndarray, spacing=(1.0, 1.0, 1.0), camera: dict | None = None) -> vtk.vtkRenderWindow:
```

Replace:
```python
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(20)
    renderer.ResetCameraClippingRange()
```
with:
```python
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam_state = camera or {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    cam.Azimuth(cam_state["azimuth"])
    cam.Elevation(cam_state["elevation"])
    cam.Zoom(cam_state["zoom"])
    renderer.ResetCameraClippingRange()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_render.py -v`
Expected: all previous tests still pass (regression check — `camera=None` renders identically to the old hardcoded behavior), plus the new test passes.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 1 from Task 1's total.

- [ ] **Step 6: Commit**

```bash
git add render.py tests/test_render.py
git commit -m "Accept an optional camera parameter in render(), default matches prior fixed view"
```

---

### Task 3: Rule-parser grammar for camera commands

**Files:**
- Modify: `commands.py`
- Modify: `tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_commands.py`:

```python
def test_parse_rotate_right():
    cmd = parse_command_rule("rotate right")
    assert cmd == {"camera": {"action": "rotate", "direction": "right", "strength": "moderately"}}


def test_parse_turn_left_with_strength():
    cmd = parse_command_rule("turn left strongly")
    assert cmd == {"camera": {"action": "rotate", "direction": "left", "strength": "strongly"}}


def test_parse_tilt_up():
    cmd = parse_command_rule("tilt up")
    assert cmd == {"camera": {"action": "tilt", "direction": "up", "strength": "moderately"}}


def test_parse_tilt_down_slightly():
    cmd = parse_command_rule("tilt down slightly")
    assert cmd == {"camera": {"action": "tilt", "direction": "down", "strength": "slightly"}}


def test_parse_zoom_in():
    cmd = parse_command_rule("zoom in")
    assert cmd == {"camera": {"action": "zoom", "direction": "in", "strength": "moderately"}}


def test_parse_zoom_out():
    cmd = parse_command_rule("zoom out")
    assert cmd == {"camera": {"action": "zoom", "direction": "out", "strength": "moderately"}}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_commands.py -k "rotate or turn_left or tilt or zoom_in or zoom_out" -v`
Expected: FAIL — `parse_command_rule` raises `ValueError` for all of these.

- [ ] **Step 3: Implement the new patterns**

In `commands.py`'s `parse_command_rule`, add these three checks. Insertion point doesn't matter relative to the existing patterns (no keyword overlap with "sharpen"/"shift"/"opacity"/etc.) — add them right after the center-shift block and before the generalized absolute-level block:

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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_commands.py -v`
Expected: all previous tests plus all 6 new pass.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 6.

- [ ] **Step 6: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "Add rule-parser grammar for rotate/tilt/zoom camera commands"
```

---

### Task 4: LLM validator + prompt for camera commands

**Files:**
- Modify: `commands.py`
- Modify: `tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_commands.py`:

```python
def test_validate_accepts_well_formed_camera_command():
    from commands import _validate_cmd
    good = {"camera": {"action": "rotate", "direction": "left", "strength": "moderately"}}
    assert _validate_cmd(good) is True


def test_validate_rejects_camera_command_with_mismatched_direction():
    from commands import _validate_cmd
    bad = {"camera": {"action": "rotate", "direction": "up", "strength": "moderately"}}
    assert _validate_cmd(bad) is False


def test_validate_rejects_camera_command_with_unknown_action():
    from commands import _validate_cmd
    bad = {"camera": {"action": "pan", "direction": "left", "strength": "moderately"}}
    assert _validate_cmd(bad) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_commands.py -k validate_.*camera -v`
Expected: FAIL — `_validate_cmd` currently falls through to `_validate_single_cmd`, which requires keys `{"target", "attribute", "direction", "strength"}` and will reject (return `False`) a `{"camera": {...}}` shaped dict entirely, including the well-formed one that should be accepted.

- [ ] **Step 3: Implement the fix**

Add these two new module-level items in `commands.py`, right before `_validate_set_cmd` (near the other `VALID_*` constants):

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

Then edit `_validate_cmd`:
```python
def _validate_cmd(obj) -> bool:
    if isinstance(obj, dict) and set(obj.keys()) == {"compound"}:
        subs = obj["compound"]
        if not isinstance(subs, list) or not subs:
            return False
        return all(_validate_set_cmd(sub) for sub in subs)
    if isinstance(obj, dict) and set(obj.keys()) == {"camera"}:
        return _validate_camera_cmd(obj)
    return _validate_single_cmd(obj)
```

Then edit `_SYSTEM_PROMPT`, inserting one new paragraph right after the existing `"center" never takes an absolute "set"/"level" -- only increase/decrease.` line and before `Respond with JSON only, no prose."""`:

```python
"center" never takes an absolute "set"/"level" -- only increase/decrease.

Camera movement is a completely separate command shape, not a variant of
the schema above -- it has no tissue target at all:
{{"camera": {{"action": "rotate"|"tilt"|"zoom",
 "direction": "left"|"right" (rotate) | "up"|"down" (tilt) | "in"|"out" (zoom),
 "strength": "slightly"|"moderately"|"strongly"}}}}
Use this whenever the user wants to change the viewing angle or zoom level,
not the transfer function itself (e.g. "look from the other side", "zoom in").

Respond with JSON only, no prose."""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_commands.py -v`
Expected: all pass, including the 3 new ones.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 3.

- [ ] **Step 6: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "Add LLM validation and prompt documentation for camera commands"
```

---

### Task 5: `evaluate.objective()` always-accepts camera commands

**Files:**
- Modify: `evaluate.py`
- Modify: `tests/test_evaluate.py`

Current relevant part of `objective()`:
```python
def objective(params_before, params_after, cmd: dict) -> int:
    if "compound" in cmd or cmd["direction"] == "reset":
        return 1

    if cmd.get("attribute") in ("width", "brightness", "center"):
        return 1

    if cmd["direction"] == "show_only":
        ...
```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate.py`:

```python
def test_objective_always_accepts_camera_commands():
    from transfer import default_params
    params = default_params()
    cmd = {"camera": {"action": "rotate", "direction": "left", "strength": "moderately"}}
    assert objective(params, params, cmd) == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -k camera -v`
Expected: FAIL — a camera-shaped `cmd` has no `"direction"` key at the top level (it's nested under `cmd["camera"]`), so `cmd["direction"]` on the very first line raises `KeyError`, not a clean `-1`/`1`.

- [ ] **Step 3: Implement the fix**

The camera check must be the **very first line** in the function body,
before the existing `"compound" in cmd or cmd["direction"] == "reset"`
check. A camera-shaped `cmd` has no top-level `"direction"` key (it's
nested under `cmd["camera"]`), and `"compound" in cmd` is `False` for it,
so Python's left-to-right `or` evaluation would fall through to
`cmd["direction"]` and raise `KeyError` if the camera check were placed
after that line instead of before it:

```python
def objective(params_before, params_after, cmd: dict) -> int:
    if "camera" in cmd:
        return 1

    if "compound" in cmd or cmd["direction"] == "reset":
        return 1

    if cmd.get("attribute") in ("width", "brightness", "center"):
        return 1

    if cmd["direction"] == "show_only":
        ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 1.

- [ ] **Step 6: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "Treat camera commands as always-accepted in objective(), checked before direction access"
```

---

### Task 6: `server.py` integration

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

Current relevant parts of `server.py` (for reference — re-read the actual file
before editing, since line numbers may have shifted from earlier tasks in
this plan):

```python
def _render_image_b64(params):
    volume, spacing = get_volume()
    img = grab(render(volume, params, spacing=spacing))
    ...


def _render_step(params, cmd_text, cmd_dict, verdict, search, step_id, session_id):
    image_b64, img, png_bytes = _render_image_b64(params)
    image_path = _save_image_file(session_id, f"step_{step_id}", png_bytes)
    return {
        "id": step_id,
        ...
        "params": params.tolist(),
        ...
    }
```

```python
class Session:
    def _load_or_init(self):
        if os.path.exists(self.path):
            ...
        self.session_id = self._new_session_id()
        step = _render_step(default_params(), None, None, None, False, 0, self.session_id)
        self.history = [step]
        self.cursor = 0
        self.save()

    def switch_dataset(self, name: str):
        set_dataset(name)
        self.session_id = self._new_session_id()
        step = _render_step(default_params(), None, None, None, False, 0, self.session_id)
        self.history = [step]
        self.cursor = 0
        self.pending = None
        self.save()
        return self.state()

    def command(self, text, parser="rule", model="qwen2.5:7b", search=False, evaluator="objective", steps=10):
        cmd = parse_command(text, parser=parser, model=model)

        current_params = np.array(self.history[self.cursor]["params"], dtype=np.float64)

        if search and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
            if evaluator == "human":
                self.pending = self._start_human_search(text, cmd, current_params, steps)
                return self.state()
            new_params = self._run_objective_search(cmd, current_params, steps)
            step = _render_step(new_params, text, cmd, None, True,
                                 self.history[-1]["id"] + 1, self.session_id)
        else:
            new_params = apply_command(cmd, current_params)
            step = _render_step(new_params, text, cmd, None, False,
                                 self.history[-1]["id"] + 1, self.session_id)
            jsonl_append(LOG_PATH, {...})

        self.history = self.history[:self.cursor + 1] + [step]
        self.cursor = len(self.history) - 1
        self.save()
        return self.state()

    def judge(self, verdict: str):
        ...
        if iteration >= p["max_steps"] or step_size < 0.01:
            step = _render_step(current, p["cmd_text"], p["cmd"], human_verdict, True,
                                 self.history[-1]["id"] + 1, self.session_id)
            self.history = self.history[:self.cursor + 1] + [step]
            ...
```

- [ ] **Step 1: Write the failing tests**

Read `tests/test_server.py`'s existing style (it tests `Session` directly, per
its own documented constraint — see the file's top-of-file comment). Append:

```python
def test_camera_command_does_not_change_params():
    session = _fresh_session()
    before_params = session.history[session.cursor]["params"]
    session.command("rotate right")
    after_params = session.history[session.cursor]["params"]
    assert before_params == after_params


def test_camera_command_changes_camera_state():
    session = _fresh_session()
    before_camera = session.history[session.cursor]["camera"]
    session.command("rotate right")
    after_camera = session.history[session.cursor]["camera"]
    assert after_camera["azimuth"] != before_camera["azimuth"]


def test_camera_state_is_per_step_and_restored_by_back():
    session = _fresh_session()
    session.command("rotate right")
    rotated_camera = session.history[session.cursor]["camera"]
    session.back()
    original_camera = session.history[session.cursor]["camera"]
    assert original_camera != rotated_camera
    session.forward()
    assert session.history[session.cursor]["camera"] == rotated_camera


def test_new_session_starts_with_default_camera():
    from camera import DEFAULT_CAMERA
    session = _fresh_session()
    assert session.history[0]["camera"] == DEFAULT_CAMERA
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -k camera -v`
Expected: FAIL — `KeyError: 'camera'` (no history entry has a `"camera"` field yet).

- [ ] **Step 3: Implement the integration**

Add the import in `server.py`'s import block:
```python
from camera import DEFAULT_CAMERA, apply_camera_command
```

Edit `_render_image_b64` and `_render_step`:
```python
def _render_image_b64(params, camera):
    volume, spacing = get_volume()
    img = grab(render(volume, params, spacing=spacing, camera=camera))
    ...  # rest unchanged


def _render_step(params, cmd_text, cmd_dict, verdict, search, step_id, session_id, camera):
    image_b64, img, png_bytes = _render_image_b64(params, camera)
    image_path = _save_image_file(session_id, f"step_{step_id}", png_bytes)
    return {
        "id": step_id,
        ...  # unchanged existing fields
        "params": params.tolist(),
        "camera": camera,
        ...  # rest unchanged
    }
```

Update every existing call site of `_render_step` to pass a `camera` argument:
- `_load_or_init`: `_render_step(default_params(), None, None, None, False, 0, self.session_id, dict(DEFAULT_CAMERA))`
- `switch_dataset`: same as above — `dict(DEFAULT_CAMERA)` (a dataset switch resets to the default view, same as it already resets `params` to `default_params()`)
- `command()`'s search branch and non-search branch: pass `current_camera` (defined below)
- `judge()`'s final-step branch: pass `self.history[self.cursor].get("camera", dict(DEFAULT_CAMERA))` (cursor hasn't advanced yet at this point in `judge()`)

Edit `command()`:
```python
    def command(self, text, parser="rule", model="qwen2.5:7b", search=False, evaluator="objective", steps=10):
        cmd = parse_command(text, parser=parser, model=model)

        current_params = np.array(self.history[self.cursor]["params"], dtype=np.float64)
        current_camera = dict(self.history[self.cursor].get("camera", DEFAULT_CAMERA))

        if "camera" in cmd:
            new_camera = apply_camera_command(cmd["camera"], current_camera)
            step = _render_step(current_params, text, cmd, None, False,
                                 self.history[-1]["id"] + 1, self.session_id, new_camera)
            self.history = self.history[:self.cursor + 1] + [step]
            self.cursor = len(self.history) - 1
            self.save()
            return self.state()

        if search and cmd.get("attribute") == "opacity" and cmd.get("direction") in ("increase", "decrease"):
            if evaluator == "human":
                self.pending = self._start_human_search(text, cmd, current_params, steps)
                return self.state()
            new_params = self._run_objective_search(cmd, current_params, steps)
            step = _render_step(new_params, text, cmd, None, True,
                                 self.history[-1]["id"] + 1, self.session_id, current_camera)
        else:
            new_params = apply_command(cmd, current_params)
            step = _render_step(new_params, text, cmd, None, False,
                                 self.history[-1]["id"] + 1, self.session_id, current_camera)
            jsonl_append(LOG_PATH, {
                "timestamp": step["timestamp"], "command": cmd,
                "params_before": current_params.tolist(), "params_after": new_params.tolist(),
                "features_before": self.history[self.cursor]["features"], "features_after": step["features"],
                "verdict": None,
            })

        self.history = self.history[:self.cursor + 1] + [step]
        self.cursor = len(self.history) - 1
        self.save()
        return self.state()
```

(The `jsonl_append` block is unchanged from before — shown here only so the
surrounding structure is unambiguous.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py -v`
Expected: all pass, including the 4 new ones.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass, count up by 4.

- [ ] **Step 6: Manual smoke test**

Start a server from a fresh port and confirm a real camera command works
end to end (not just against the isolated `Session` tests):
```bash
nohup .venv/bin/python server.py --dataset synthetic --port 8005 > /tmp/server_camera_verify.log 2>&1 &
sleep 3
curl -s -X POST http://127.0.0.1:8005/api/command -H "Content-Type: application/json" \
  -d '{"text": "zoom in"}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['current']['camera'])"
```
Expected: prints a camera dict with `zoom` increased above `1.0`. This
matches `CommandRequest`'s actual schema (`server.py`: `text: str, parser:
str = "rule", model: str = "qwen2.5:7b", search: bool = False, evaluator:
str = "objective", steps: int = 10`) — only `text` is required, so the
payload above is already complete. Kill the test server afterward
(`lsof -ti:8005 | xargs kill`).

- [ ] **Step 7: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "Thread camera state through Session as per-step history"
```

---

### Task 7: `mvp.py` integration

**Files:**
- Modify: `mvp.py`

No automated test exists for `mvp.py` anywhere in this codebase (there is no
`tests/test_mvp.py`) — verified manually.

- [ ] **Step 1: Add camera persistence**

Add near the existing `STATE_PATH`/`LOG_PATH`/`PREF_PATH` constants:
```python
CAMERA_STATE_PATH = "out/camera_state.json"
```

Add the import: `from camera import DEFAULT_CAMERA, apply_camera_command`

Add these two functions near `load_state`/`save_state`:
```python
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

- [ ] **Step 2: Thread camera through `save_png`**

Edit:
```python
def save_png(params: np.ndarray, path: str, camera: dict | None = None) -> np.ndarray:
    volume, spacing = get_volume()
    img = grab(render(volume, params, spacing=spacing, camera=camera))
    Image.fromarray(img).save(path)
    return img
```

- [ ] **Step 3: Add camera command handling to `main()`**

In `main()`, right after `params = load_state()`, add:
```python
    camera = load_camera()
```

Right after the existing `print("parsed command:", cmd)` line and before
the existing `if args.learn and cmd.get("attribute") == "opacity" ...`
branch, add:
```python
    if "camera" in cmd:
        new_camera = apply_camera_command(cmd["camera"], camera)
        save_camera(new_camera)
        save_png(params, "out/camera.png", camera=new_camera)
        print("new camera state:", new_camera)
        return
```

- [ ] **Step 4: Manual verification**

```bash
.venv/bin/python mvp.py --cmd "zoom in"
```
Expected: prints `parsed command: {'camera': ...}` then `new camera state:
{...}` with `zoom` above `1.0`, and creates `out/camera.png`. Run it a
second time with `--cmd "rotate left"` and confirm `out/camera_state.json`
now reflects both accumulated changes (zoom still applied, azimuth also
changed) — i.e. camera state persists across separate CLI invocations, the
same way `out/state.json` already does for `params`.

- [ ] **Step 5: Run the full fast suite to confirm nothing broke**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (no new automated tests for this task, so count is
unchanged from Task 6).

- [ ] **Step 6: Commit**

```bash
git add mvp.py
git commit -m "Add camera persistence and command handling to the CLI"
```

---

### Task 8: `COMMAND_REFERENCE` addition

**Files:**
- Modify: `commands.py`
- Modify: `COMMANDS.md`

- [ ] **Step 1: Add the new category**

In `commands.py`'s `COMMAND_REFERENCE` list, add one more entry (position
doesn't matter — append at the end, right before `Reset`, or after it;
appending right before `Reset` keeps "Reset" as the natural last item):

```python
    {
        "category": "Camera",
        "examples": ["rotate left", "tilt down", "zoom in"],
        "description": "Adjust the viewing angle or zoom level. Doesn't change the transfer function.",
    },
```

- [ ] **Step 2: Regenerate `COMMANDS.md`**

Run: `.venv/bin/python -m tools.gen_commands_doc`
Expected: prints `Wrote COMMANDS.md`.

- [ ] **Step 3: Verify the drift-check test still passes**

Run: `.venv/bin/python -m pytest tests/test_commands_doc.py -v`
Expected: 1 passed.

- [ ] **Step 4: Sanity-check the new examples actually parse**

```bash
.venv/bin/python -c "
from commands import COMMAND_REFERENCE, parse_command_rule
for entry in COMMAND_REFERENCE:
    if entry['category'] == 'Camera':
        for example in entry['examples']:
            print(example, '->', parse_command_rule(example))
"
```
Expected: all three print a `{"camera": {...}}` dict with no exception.

- [ ] **Step 5: Run the full fast suite**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (no new automated tests beyond the existing drift check).

- [ ] **Step 6: Commit**

```bash
git add commands.py COMMANDS.md
git commit -m "Add Camera category to the command reference"
```

---

## Notes for the implementer

- Do not modify `search.py`, `rl/`, `plots/`, or `datasets.py` — untouched
  by this plan.
- `apply_command` (in `commands.py`) is never touched and never sees a
  camera-shaped command — camera dispatch happens earlier, in
  `Session.command()` (`server.py`) and `main()` (`mvp.py`), before
  `apply_command` would ever be called.
- The existing hill-climbing search-loop gate
  (`cmd.get("attribute") == "opacity"`) already excludes camera commands
  with zero new code, since a camera command has no `"attribute"` key —
  verify this holds rather than adding redundant gating logic.
- Task order matters: `camera.py` (Task 1) is a dependency of every other
  task; `render.py` (Task 2) is a dependency of `server.py`/`mvp.py`
  (Tasks 6-7); the rule-parser grammar (Task 3) is a dependency of the LLM
  validator work only in the sense that both describe the same command
  shape — Task 4 does not literally depend on Task 3's code, but should
  follow it for narrative consistency.
