# WebXR VR Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a WebXR headset (HTC Vive Focus 3, VIVE Browser) view the existing vtk.js volume in immersive VR, change the transfer function by voice via controller push-to-talk, step history with controller buttons, and see a read-only status HUD.

**Architecture:** vtk.js 25.11's built-in WebXR support (`openGLRenderWindow.enterXR(session)` / `stopXR()`) renders the same renderer/volume as the flat viewer, so the existing command pipeline updates VR for free. A DOM-free `static/xr_logic.js` holds all testable logic (sample distance, controller edge detection, HUD state, HUD geometry, reply summary); `static/xr.js` wires it to WebXR, the viewer and `app.js`, which emits `app:*` window events and exposes `window.appControls`.

**Tech Stack:** vtk.js 25.11.0 (UMD from unpkg), WebXR Device API, MediaRecorder, FastAPI (unchanged), `node --test` (Node 26) wrapped in pytest.

Spec: `docs/superpowers/specs/2026-09-29-webxr-vr-mode-design.md`.

**Repo conventions:** run from repo root; Python is `.venv/bin/python`; commit messages have NO `Co-Authored-By` / Claude trailers; `cat` is aliased to `bat` (use `command cat` in heredocs).

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `server.py` | modified (already, uncommitted) | `--host`, `--ssl-certfile`, `--ssl-keyfile` flags |
| `static/xr_logic.js` | create | pure logic: `xrSampleDistance`, `createControllerMapper`, `hudReduce`/`hudLines`, `hudPlacement`, `summarizeStep` |
| `tests/js/xr_logic.test.js` | create | `node:test` suite for `xr_logic.js` |
| `tests/test_xr_logic.py` | create | runs the node suite under pytest (skip without node) |
| `static/app.js` | modify | `emitAppEvent`, events at existing points, `window.appControls` |
| `static/viewer.js` | modify | `enterXR(session)`, `exitXR()`, XR getters; skip camera/render in XR |
| `static/xr.js` | create | Enter VR button, session lifecycle, input loop, haptics, HUD actor, tint |
| `static/index.html` | modify | Enter VR button, script tags |
| `static/style.css` | modify | nothing new needed (button reuses `.btn .btn-secondary .btn-sm`) |
| `docs/vr-integration.md` | modify | Focus 3 setup + manual checklist |

---

### Task 0: Commit the LAN/HTTPS server flags

**Files:** `server.py` (already edited), `.gitignore` (already has `certs/`)

- [ ] **Step 1: Verify the diff is only the flags and the ignore line**

Run: `git diff --stat`
Expected: `server.py` and `.gitignore` only.

- [ ] **Step 2: Commit**

```bash
git add server.py .gitignore
git commit -m "feat(server): add --host and TLS flags for LAN headset clients"
```

---

### Task 1: Test harness + `xrSampleDistance`

**Files:**
- Create: `static/xr_logic.js`
- Create: `tests/js/xr_logic.test.js`
- Create: `tests/test_xr_logic.py`

- [ ] **Step 1: Write the pytest wrapper**

`tests/test_xr_logic.py`:

```python
"""Runs the node:test suite for static/xr_logic.js so plain `pytest` covers it.

The frontend has no JS toolchain; Node's built-in test runner needs none.
"""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_xr_logic_node_suite():
    result = subprocess.run(
        ["node", "--test", "tests/js/xr_logic.test.js"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Write the failing JS test**

`tests/js/xr_logic.test.js`:

```js
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const xr = require("../../static/xr_logic.js");

test("xrSampleDistance uses the finest voxel spacing", () => {
  assert.equal(xr.xrSampleDistance([0.8, 0.8, 2.5]), 0.8);
  assert.equal(xr.xrSampleDistance([1.5, 1.2, 1.3]), 1.2);
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -v`
Expected: FAIL, output contains `Cannot find module '../../static/xr_logic.js'`.

- [ ] **Step 4: Minimal implementation**

`static/xr_logic.js`:

```js
// Pure, DOM-free logic for the WebXR mode (static/xr.js). Loaded as a plain
// script in the browser (window.xrLogic) and via require() in node:test.
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.xrLogic = api;
})(typeof window !== "undefined" ? window : globalThis, () => {
  "use strict";

  // The flat viewer ray-marches at half the finest spacing (see viewer.js);
  // in XR every frame is rendered twice on a mobile GPU, so march at full
  // spacing instead.
  function xrSampleDistance(spacing) {
    return Math.min(...spacing);
  }

  return { xrSampleDistance };
});
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -v`
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add static/xr_logic.js tests/js/xr_logic.test.js tests/test_xr_logic.py
git commit -m "feat(xr): add xr_logic module with node test harness"
```

---

### Task 2: Controller mapper (push-to-talk, back/forward)

Input model: each frame, `xr.js` passes one entry per controller:
`{ handedness: "left" | "right" | "none", pressed: boolean[] }` where
`pressed[i] = gamepad.buttons[i].pressed` (xr-standard mapping: 0 trigger,
4 A/X, 5 B/Y). Rules:
- trigger press (edge) while idle and not busy → `startRecording()`, remember hand;
- trigger release on the *recording* hand → `stopRecording()` (even if busy);
- A/X press (edge) while idle and not busy → `back()`; B/Y → `forward()`;
- no actions from the other hand while recording.

**Files:** Modify `static/xr_logic.js`, `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write failing tests** (append to `tests/js/xr_logic.test.js`)

```js
function recorder() {
  const calls = [];
  const actions = {
    startRecording: () => calls.push("start"),
    stopRecording: () => calls.push("stop"),
    back: () => calls.push("back"),
    forward: () => calls.push("forward"),
  };
  return { calls, mapper: xr.createControllerMapper(actions) };
}

const idle = [false, false, false, false, false, false];
function press(...indices) {
  const p = idle.slice();
  for (const i of indices) p[i] = true;
  return p;
}

test("holding the trigger records once and releasing stops", () => {
  const { calls, mapper } = recorder();
  mapper.update([{ handedness: "right", pressed: press(0) }], false);
  mapper.update([{ handedness: "right", pressed: press(0) }], false);
  mapper.update([{ handedness: "right", pressed: idle }], false);
  assert.deepEqual(calls, ["start", "stop"]);
});

test("release stops recording even while busy", () => {
  const { calls, mapper } = recorder();
  mapper.update([{ handedness: "left", pressed: press(0) }], false);
  mapper.update([{ handedness: "left", pressed: idle }], true);
  assert.deepEqual(calls, ["start", "stop"]);
});

test("busy blocks starting, back and forward", () => {
  const { calls, mapper } = recorder();
  mapper.update([{ handedness: "right", pressed: press(0, 4, 5) }], true);
  assert.deepEqual(calls, []);
});

test("A/X goes back, B/Y goes forward, on press edges only", () => {
  const { calls, mapper } = recorder();
  mapper.update([{ handedness: "right", pressed: press(4) }], false);
  mapper.update([{ handedness: "right", pressed: press(4) }], false);
  mapper.update([{ handedness: "left", pressed: press(5) }], false);
  assert.deepEqual(calls, ["back", "forward"]);
});

test("the other hand is ignored while recording", () => {
  const { calls, mapper } = recorder();
  mapper.update([
    { handedness: "right", pressed: press(0) },
    { handedness: "left", pressed: idle },
  ], false);
  mapper.update([
    { handedness: "right", pressed: press(0) },
    { handedness: "left", pressed: press(0, 4) },
  ], false);
  mapper.update([
    { handedness: "right", pressed: idle },
    { handedness: "left", pressed: press(0, 4) },
  ], false);
  assert.deepEqual(calls, ["start", "stop"]);
});

test("short button arrays are treated as unpressed", () => {
  const { calls, mapper } = recorder();
  mapper.update([{ handedness: "right", pressed: [true] }], false);
  mapper.update([{ handedness: "right", pressed: [false] }], false);
  assert.deepEqual(calls, ["start", "stop"]);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `xr.createControllerMapper is not a function`.

- [ ] **Step 3: Implement** (in `static/xr_logic.js`, above `return`, and export it)

```js
  // xr-standard gamepad mapping indices.
  const BUTTON = { trigger: 0, primary: 4, secondary: 5 };

  // Edge-detects controller buttons across frames and turns them into app
  // actions. `sources` is [{handedness, pressed: boolean[]}] per frame.
  function createControllerMapper(actions) {
    const previous = new Map();
    let recordingHand = null;
    return {
      update(sources, busy) {
        for (const { handedness, pressed } of sources) {
          const was = previous.get(handedness) || [];
          const down = (i) => Boolean(pressed[i]) && !was[i];
          const up = (i) => !pressed[i] && Boolean(was[i]);
          if (recordingHand === handedness && up(BUTTON.trigger)) {
            recordingHand = null;
            actions.stopRecording();
          } else if (recordingHand === null && !busy) {
            if (down(BUTTON.trigger)) {
              recordingHand = handedness;
              actions.startRecording();
            } else if (down(BUTTON.primary)) {
              actions.back();
            } else if (down(BUTTON.secondary)) {
              actions.forward();
            }
          }
          previous.set(handedness, pressed.slice());
        }
      },
    };
  }
```

Change the return to `return { xrSampleDistance, createControllerMapper };`.

- [ ] **Step 4: Run to verify pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all tests pass (`# fail 0`).

- [ ] **Step 5: Commit**

```bash
git add static/xr_logic.js tests/js/xr_logic.test.js
git commit -m "feat(xr): map controller buttons to push-to-talk and history"
```

---

### Task 3: HUD state, HUD lines, reply summary

Events (same names as the `app:*` window events from Task 5, without the prefix):
`recording {active}`, `transcribing`, `transcript {text}`, `busy {busy}`,
`applied {summary, dataset}`, `error {message}`, plus `voice-unavailable`
(from `xr.js`).

**Files:** Modify `static/xr_logic.js`, `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write failing tests**

```js
function run(events) {
  return events.reduce((s, e) => xr.hudReduce(s, e), xr.HUD_INITIAL);
}

test("hud walks through a voice command", () => {
  let s = run([{ type: "recording", active: true }]);
  assert.equal(xr.hudLines(s)[0], "● Recording…");
  s = run([
    { type: "recording", active: true },
    { type: "recording", active: false },
    { type: "transcribing" },
    { type: "busy", busy: true },
  ]);
  assert.equal(xr.hudLines(s)[0], "Transcribing…");
  s = xr.hudReduce(s, { type: "transcript", text: "show the lungs" });
  assert.equal(xr.hudLines(s)[0], "Applying…");
  assert.equal(xr.hudLines(s)[1], "“show the lungs”");
  s = xr.hudReduce(s, { type: "applied", summary: "the policy answered", dataset: "ct_chest" });
  s = xr.hudReduce(s, { type: "busy", busy: false });
  assert.deepEqual(xr.hudLines(s), [
    "Hold trigger to talk",
    "“show the lungs”",
    "the policy answered",
    "ct_chest · A/X back · B/Y forward",
  ]);
});

test("errors stick until the next recording", () => {
  let s = run([{ type: "error", message: "Could not parse that" }, { type: "busy", busy: false }]);
  assert.equal(xr.hudLines(s)[0], "⚠ Could not parse that");
  s = xr.hudReduce(s, { type: "recording", active: true });
  assert.equal(xr.hudLines(s)[0], "● Recording…");
});

test("busy from back/forward shows applying, then idle", () => {
  let s = run([{ type: "busy", busy: true }]);
  assert.equal(xr.hudLines(s)[0], "Applying…");
  s = xr.hudReduce(s, { type: "busy", busy: false });
  assert.equal(xr.hudLines(s)[0], "Hold trigger to talk");
});

test("voice unavailable replaces the idle prompt", () => {
  const s = run([{ type: "voice-unavailable" }]);
  assert.equal(xr.hudLines(s)[0], "Voice unavailable — A/X back · B/Y forward");
});

test("hudReduce ignores unknown events and never mutates", () => {
  const before = xr.HUD_INITIAL;
  const after = xr.hudReduce(before, { type: "nope" });
  assert.equal(after, before);
  xr.hudReduce(before, { type: "recording", active: true });
  assert.equal(before.status, "idle");
});

const CLASSES = ["lungs", "skeleton"];
const LABELS = { lungs: "Lungs", skeleton: "Skeleton" };
const SAID = { policy: "the policy answered", exact: "applied directly" };

test("summarizeStep lists classes that moved", () => {
  const text = xr.summarizeStep(
    { mode: "policy", class_visibility: { lungs: 0.5, skeleton: 0.2 } },
    { lungs: 0.1, skeleton: 0.2 }, CLASSES, LABELS, SAID);
  assert.equal(text, "the policy answered · Lungs 10.0→50.0%");
});

test("summarizeStep keeps the server message and falls back to exact", () => {
  const text = xr.summarizeStep(
    { mode: "weird", message: "no match", class_visibility: {} }, null, CLASSES, LABELS, SAID);
  assert.equal(text, "applied directly · no match");
});

test("summarizeStep says when nothing moved", () => {
  const text = xr.summarizeStep(
    { mode: "exact", class_visibility: { lungs: 0.1 } }, { lungs: 0.1 }, CLASSES, LABELS, SAID);
  assert.equal(text, "applied directly · nothing moved measurably");
});
```

- [ ] **Step 2: Run to verify failure**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `Cannot read properties of undefined` / `hudReduce is not a function`.

- [ ] **Step 3: Implement** (in `static/xr_logic.js`)

```js
  const HUD_INITIAL = Object.freeze({
    status: "idle", // idle | recording | transcribing | applying | error
    voice: true,
    transcript: "",
    summary: "",
    dataset: "",
    error: "",
  });

  // Folds app/xr events into what the VR status panel shows. Returns the same
  // object for events it ignores so callers can skip redraws cheaply.
  function hudReduce(state, event) {
    switch (event.type) {
      case "recording":
        return event.active
          ? { ...state, status: "recording", error: "" }
          : state;
      case "transcribing":
        return { ...state, status: "transcribing" };
      case "transcript":
        return { ...state, status: "applying", transcript: event.text };
      case "busy":
        if (event.busy) {
          return state.status === "idle" ? { ...state, status: "applying" } : state;
        }
        return state.status === "applying" || state.status === "transcribing"
          ? { ...state, status: "idle" }
          : state;
      case "applied":
        return { ...state, summary: event.summary, dataset: event.dataset || state.dataset };
      case "error":
        return { ...state, status: "error", error: event.message };
      case "voice-unavailable":
        return { ...state, voice: false };
      default:
        return state;
    }
  }

  function hudLines(state) {
    const idle = state.voice ? "Hold trigger to talk" : "Voice unavailable — A/X back · B/Y forward";
    const head = {
      idle,
      recording: "● Recording…",
      transcribing: "Transcribing…",
      applying: "Applying…",
      error: `⚠ ${state.error}`,
    }[state.status];
    const footer = [state.dataset, "A/X back · B/Y forward"].filter(Boolean).join(" · ");
    return [head, state.transcript ? `“${state.transcript}”` : "", state.summary, footer];
  }

  // One-line version of app.js's appendReply, for the VR panel.
  function summarizeStep(step, before, classOrder, classLabel, methodSaid) {
    const parts = [methodSaid[step.mode] || methodSaid.exact];
    if (step.message) parts.push(step.message);
    const now = step.class_visibility || {};
    const moved = classOrder.filter((c) => before && before[c] != null && now[c] != null
      && Math.abs(now[c] - before[c]) >= 0.0001);
    if (moved.length) {
      parts.push(moved.map((c) => `${classLabel[c]} ${(before[c] * 100).toFixed(1)}→${(now[c] * 100).toFixed(1)}%`).join(", "));
    } else if (!step.message) {
      parts.push("nothing moved measurably");
    }
    return parts.join(" · ");
  }
```

Note: `hudReduce` for `error` while `busy:false` arrives after — `busy` only resets `applying`/`transcribing`, so the error stays (tested above).

Return: `return { xrSampleDistance, createControllerMapper, HUD_INITIAL, hudReduce, hudLines, summarizeStep };`

- [ ] **Step 4: Run to verify pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add static/xr_logic.js tests/js/xr_logic.test.js
git commit -m "feat(xr): add HUD state reducer and reply summary"
```

---

### Task 4: HUD placement geometry

In XR, vtk.js derives the view from `camera.physicalViewUp` / `physicalViewNorth`.
`viewer.enterXR` (Task 6) sets those from the flat camera, so "up" and
"forward" (direction of projection) are known. The HUD is a plane above the
volume's bounding sphere, pulled half a radius toward the viewer, facing them.

**Files:** Modify `static/xr_logic.js`, `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write failing test**

```js
test("hudPlacement puts a viewer-facing panel above the volume", () => {
  // Unit cube-ish volume centred at origin, looking down -z with +y up.
  const p = xr.hudPlacement([-1, 1, -1, 1, -1, 1], [0, 1, 0], [0, 0, -1], 1024 / 384);
  const r = Math.sqrt(3);
  const width = 2 * r * 0.8;
  const height = width * 384 / 1024;
  const close = (a, b) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-9, `${a} vs ${b}`));
  // right = dop x up = (0,0,-1) x (0,1,0) = (1,0,0)
  close(p.origin, [-width / 2, r * 1.05, r * 0.5]);
  close(p.point1, [width / 2, r * 1.05, r * 0.5]);
  close(p.point2, [-width / 2, r * 1.05 + height, r * 0.5]);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `xr.hudPlacement is not a function`.

- [ ] **Step 3: Implement**

```js
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }

  function normalize(v) {
    const n = Math.hypot(...v);
    return v.map((x) => x / n);
  }

  // vtkPlaneSource corners (origin, point1 = +width, point2 = +height) for a
  // status panel floating above `bounds`, facing back along `dop`.
  function hudPlacement(bounds, up, dop, aspect) {
    const center = [0, 2, 4].map((i) => (bounds[i] + bounds[i + 1]) / 2);
    const radius = Math.hypot(...[0, 2, 4].map((i) => bounds[i + 1] - bounds[i])) / 2;
    const u = normalize(up);
    const d = normalize(dop);
    const right = normalize(cross(d, u));
    const width = 2 * radius * 0.8;
    const height = width / aspect;
    const add = (a, b, s) => a.map((x, i) => x + b[i] * s);
    let base = add(center, u, radius * 1.05);
    base = add(base, d, -radius * 0.5);
    const origin = add(base, right, -width / 2);
    return {
      origin,
      point1: add(origin, right, width),
      point2: add(origin, u, height),
    };
  }
```

Add `hudPlacement` to the returned object.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add static/xr_logic.js tests/js/xr_logic.test.js
git commit -m "feat(xr): compute HUD panel placement above the volume"
```

---

### Task 5: `app.js` events and `window.appControls`

No behavior change in the flat UI except: a failed transcription/command
during voice now shows a destructive toast instead of an unhandled rejection.

**Files:** Modify `static/app.js`, `static/index.html`

- [ ] **Step 1: Load `xr_logic.js` before `app.js`** (`static/index.html`, bottom)

```html
  <script src="https://unpkg.com/vtk.js@25.11.0"></script>
  <script src="/static/viewer.js"></script>
  <script src="/static/xr_logic.js"></script>
  <script src="/static/app.js"></script>
```

- [ ] **Step 2: Add the emitter next to `setLoading`** (`static/app.js` ~line 118) and emit busy transitions

Replace `setLoading` with:

```js
// Status events for listeners outside this file (static/xr.js's VR panel,
// which cannot see toasts or the chat thread while in an immersive session).
function emitAppEvent(type, detail = {}) {
  window.dispatchEvent(new CustomEvent(`app:${type}`, { detail }));
}

function setLoading(loading) {
  const wasActive = loadingCount > 0;
  loadingCount = Math.max(0, loadingCount + (loading ? 1 : -1));
  const active = loadingCount > 0;
  loadingOverlay.hidden = !active;
  loadingOverlay.setAttribute("aria-busy", String(active));
  if (active !== wasActive) emitAppEvent("busy", { busy: active });
}
```

- [ ] **Step 3: Emit `applied` from commands and history**

In `sendCommandImpl`, after `appendReply(data.current, before);` add:

```js
  emitAppEvent("applied", {
    summary: window.xrLogic.summarizeStep(data.current, before, CLASS_ORDER, CLASS_LABEL, METHOD_SAID),
    dataset: state.dataset,
  });
```

In `navigateImpl`, after the `postSceneTransition(...)` call add:

```js
  emitAppEvent("applied", {
    summary: path.endsWith("/back") ? "stepped back" : "stepped forward",
    dataset: state.dataset,
  });
```

- [ ] **Step 4: Emit errors from destructive toasts** (`showToast`, ~line 1059, first line of the body)

```js
  if (variant === "destructive") emitAppEvent("error", { message });
```

- [ ] **Step 5: Emit recording/transcription events** (in `startRecording`)

Replace the `mediaRecorder.onstop` handler and the lines after `mediaRecorder.start();` so the function reads:

```js
  mediaRecorder.onstop = async () => {
    micBtn.classList.remove("recording");
    micWaveform.hidden = true;
    micIcon.style.display = "";
    stopWaveform();
    stream.getTracks().forEach((t) => t.stop());
    emitAppEvent("recording", { active: false });

    const blob = new Blob(chunks, { type: "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "clip.webm");
    micBtn.disabled = true;
    emitAppEvent("transcribing");
    try {
      await withLoading(async () => {
        const r = await fetch("/api/transcribe", { method: "POST", body: form });
        const data = await r.json();
        if (data.text) {
          emitAppEvent("transcript", { text: data.text });
          await sendCommand(data.text);
        } else {
          emitAppEvent("error", { message: "No speech recognised" });
        }
      });
    } catch (err) {
      showToast(`Voice command failed: ${err.message}`, "destructive");
    } finally {
      micBtn.disabled = false;
    }
  };
  mediaRecorder.start();
  micBtn.classList.add("recording");
  micIcon.style.display = "none";
  micWaveform.hidden = false;
  startWaveform(stream);
  emitAppEvent("recording", { active: true });
}
```

(`await sendCommand` is new: it keeps the transcribe+command pair inside one `withLoading`, so `busy` stays true across both and errors reach the `catch`.)

- [ ] **Step 6: Expose controls** (right after the `el("reset-btn")` listener, ~line 636)

```js
// Driven by controller buttons in static/xr.js.
window.appControls = {
  startRecording,
  stopRecording,
  back: () => navigate("/api/back"),
  forward: () => navigate("/api/forward"),
};
```

`startRecording`/`stopRecording` are function declarations, hoisted, so referencing them here is fine.

- [ ] **Step 7: Manual flat-UI regression on the Mac**

Run: `.venv/bin/python server.py` and open `http://127.0.0.1:8000` in Chrome.
In DevTools console run:
```js
["busy","applied","error","recording","transcribing","transcript"].forEach(t =>
  addEventListener(`app:${t}`, e => console.log(t, e.detail)));
```
Then: type `show the lungs` + Enter → console shows `busy {busy:true}`, `applied {summary: "...", dataset: "ct_chest"}`, `busy {busy:false}`; click back → `applied {summary: "stepped back"}`; hold Space, say "show the bones", release → `recording`, `transcribing`, `transcript`, `applied`. Chat thread and toasts look exactly as before.

- [ ] **Step 8: Run the suite and commit**

Run: `.venv/bin/python -m pytest -q -m "not slow"`
Expected: same pass count as before this task plus the xr test; no new failures.

```bash
git add static/app.js static/index.html
git commit -m "feat(ui): emit app status events and expose voice/history controls"
```

---

### Task 6: `viewer.js` XR hooks

**Files:** Modify `static/viewer.js`

- [ ] **Step 1: Add XR state and hooks** (after `setCamera`, before `async function load`)

```js
  // Saved flat-viewer settings while an immersive session is running.
  let xrSaved = null;

  function inXR() {
    return xrSaved !== null;
  }

  // Enters VR with a session the caller requested (it needs the click's user
  // activation, and requesting it there makes rejection catchable -- vtk.js's
  // own startXR() swallows it inside a promise).
  async function enterXR(session) {
    if (!openGLRenderWindow || !mapper || !volumeMetadata) throw new Error("no local volume loaded");
    xrSaved = {
      style: interactor.getInteractorStyle(),
      sampleDistance: mapper.getSampleDistance(),
      camera: fromRendererCamera(),
      background: renderer.getBackground().slice(),
    };
    // vtk.js's trackball style binds the right trigger/trackpad to a camera
    // "fly" in XR, which would fight push-to-talk; the base style has no 3D
    // button handlers at all.
    interactor.setInteractorStyle(vtk.Rendering.Core.vtkInteractorStyle.newInstance());
    mapper.setSampleDistance(window.xrLogic.xrSampleDistance(volumeMetadata.spacing));
    // Keep the orientation the user was looking at in the flat viewer.
    camera.setPhysicalViewUp(...camera.getViewUp());
    camera.setPhysicalViewNorth(...camera.getDirectionOfProjection());
    try {
      await openGLRenderWindow.enterXR(session);
    } catch (error) {
      await exitXR();
      throw error;
    }
  }

  async function exitXR() {
    if (!xrSaved) return;
    const saved = xrSaved;
    await openGLRenderWindow.stopXR();
    xrSaved = null;
    interactor.setInteractorStyle(saved.style);
    mapper.setSampleDistance(saved.sampleDistance);
    renderer.setBackground(...saved.background);
    camera.setPosition(...saved.camera.position);
    camera.setFocalPoint(...saved.camera.focal_point);
    camera.setViewUp(...saved.camera.view_up);
    renderer.resetCameraClippingRange();
    renderWindow.render();
  }
```

- [ ] **Step 2: Skip camera changes while in XR**

At the top of `setCamera(value)`:

```js
  function setCamera(value) {
    // In XR the headset pose drives the view, and resetCamera() (used for
    // legacy camera values) would also reset vtk.js's physical scale.
    if (inXR()) return;
    if (!camera || !value) return;
```

- [ ] **Step 3: Extend the public API** (replace the `window.volumeViewer = ...` line)

```js
  window.volumeViewer = {
    load, setTransferFunction, setLabelAwareTransferFunction, getCamera, setCamera,
    // The XR frame loop renders on its own; a flat render() mid-session would
    // draw into the XR-sized canvas for nothing.
    render: () => { if (!inXR()) renderWindow?.render(); },
    enterXR, exitXR, inXR,
    getRenderer: () => renderer,
    getVolumeBounds: () => volume?.getBounds(),
    getCameraAxes: () => (camera ? { up: camera.getViewUp().slice(), dop: camera.getDirectionOfProjection().slice() } : null),
    get dataset() { return datasetName; },
    get transferFunction() { return transferFunction; },
  };
```

- [ ] **Step 4: Flat-viewer regression on the Mac**

Reload `http://127.0.0.1:8000`: volume loads, drag rotates, commands update the TF. In the console: `volumeViewer.inXR()` → `false`; `volumeViewer.getCameraAxes()` → two 3-vectors.

- [ ] **Step 5: Commit**

```bash
git add static/viewer.js
git commit -m "feat(viewer): add enter/exit XR hooks with XR sample distance"
```

---

### Task 7: `xr.js` — button, session, input, haptics, HUD, tint

**Files:** Create `static/xr.js`; Modify `static/index.html`

- [ ] **Step 1: Add the button** (`static/index.html`, inside `#nav`, after `reset-btn`)

```html
              <button id="enter-vr-btn" class="btn btn-secondary btn-sm" hidden data-tooltip="View in a VR headset (WebXR)">Enter VR</button>
```

And the script after `app.js`:

```html
  <script src="/static/xr.js"></script>
```

- [ ] **Step 2: Write `static/xr.js`**

```js
// Immersive VR mode: shows the local vtk.js volume in a WebXR headset, with
// controller push-to-talk, history buttons and a read-only status panel.
// Pure logic lives in xr_logic.js; this file only touches WebXR/vtk/DOM.
(() => {
  "use strict";

  const logic = window.xrLogic;
  const button = document.getElementById("enter-vr-btn");
  const HUD_W = 1024;
  const HUD_H = 384;
  const RECORDING_BG = [0.25, 0, 0];

  let session = null;
  let hudState = logic.HUD_INITIAL;
  let busy = false;
  let hud = null; // { canvas, ctx, texture, actor }
  let flatBackground = null;
  const mapper = logic.createControllerMapper({
    startRecording: () => {
      if (!hudState.voice) return pulseAll(0.6, 60, 2);
      window.appControls.startRecording();
    },
    stopRecording: () => window.appControls.stopRecording(),
    back: () => window.appControls.back(),
    forward: () => window.appControls.forward(),
  });

  // ---------- haptics ----------

  function pulseAll(intensity, ms, count = 1) {
    if (!session) return;
    for (const source of session.inputSources) {
      const actuator = source.gamepad?.hapticActuators?.[0];
      if (!actuator?.pulse) continue;
      for (let i = 0; i < count; i += 1) {
        setTimeout(() => actuator.pulse(intensity, ms), i * (ms + 80));
      }
    }
  }

  // ---------- HUD ----------

  function wrapLine(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let cut = text;
    while (cut.length > 1 && ctx.measureText(`${cut}…`).width > maxWidth) cut = cut.slice(0, -1);
    return `${cut}…`;
  }

  function drawHud() {
    if (!hud) return;
    const { ctx } = hud;
    const [head, transcript, summary, footer] = logic.hudLines(hudState);
    ctx.fillStyle = "rgba(12, 12, 16, 0.92)";
    ctx.fillRect(0, 0, HUD_W, HUD_H);
    ctx.strokeStyle = hudState.status === "recording" ? "#ef4444" : "#3f3f46";
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, HUD_W - 8, HUD_H - 8);
    const pad = 40;
    const max = HUD_W - 2 * pad;
    ctx.textBaseline = "top";
    ctx.fillStyle = hudState.status === "error" ? "#fca5a5" : "#fafafa";
    ctx.font = "600 64px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, head, max), pad, 36);
    ctx.fillStyle = "#e4e4e7";
    ctx.font = "italic 44px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, transcript, max), pad, 128);
    ctx.fillStyle = "#a1a1aa";
    ctx.font = "40px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, summary, max), pad, 200);
    ctx.fillStyle = "#71717a";
    ctx.font = "34px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, footer, max), pad, 300);
    hud.texture.modified();
  }

  function addHud() {
    const renderer = window.volumeViewer.getRenderer();
    const bounds = window.volumeViewer.getVolumeBounds();
    const axes = window.volumeViewer.getCameraAxes();
    const canvas = document.createElement("canvas");
    canvas.width = HUD_W;
    canvas.height = HUD_H;
    const place = logic.hudPlacement(bounds, axes.up, axes.dop, HUD_W / HUD_H);
    const plane = vtk.Filters.Sources.vtkPlaneSource.newInstance({
      origin: place.origin, point1: place.point1, point2: place.point2,
    });
    const planeMapper = vtk.Rendering.Core.vtkMapper.newInstance();
    planeMapper.setInputConnection(plane.getOutputPort());
    const texture = vtk.Rendering.Core.vtkTexture.newInstance();
    texture.setInterpolate(true);
    texture.setCanvas(canvas);
    const actor = vtk.Rendering.Core.vtkActor.newInstance();
    actor.setMapper(planeMapper);
    actor.addTexture(texture);
    // Unlit: the panel should read the same from any angle.
    actor.getProperty().setAmbient(1);
    actor.getProperty().setDiffuse(0);
    actor.getProperty().setSpecular(0);
    renderer.addActor(actor);
    hud = { canvas, ctx: canvas.getContext("2d"), texture, actor };
    drawHud();
  }

  function removeHud() {
    if (!hud) return;
    window.volumeViewer.getRenderer()?.removeActor(hud.actor);
    hud = null;
  }

  // ---------- app events ----------

  function applyEvent(event) {
    const next = logic.hudReduce(hudState, event);
    if (next === hudState) return;
    const prev = hudState;
    hudState = next;
    if (!session) return;
    drawHud();
    const renderer = window.volumeViewer.getRenderer();
    if (next.status === "recording" && prev.status !== "recording") {
      renderer.setBackground(...RECORDING_BG);
      pulseAll(0.5, 40);
    } else if (prev.status === "recording" && next.status !== "recording") {
      renderer.setBackground(...flatBackground);
      pulseAll(0.5, 40);
    }
    if (event.type === "applied") pulseAll(0.7, 90);
    if (event.type === "error") pulseAll(0.6, 60, 2);
  }

  for (const type of ["recording", "transcribing", "transcript", "busy", "applied", "error"]) {
    window.addEventListener(`app:${type}`, (e) => {
      if (type === "busy") busy = e.detail.busy;
      applyEvent({ type, ...e.detail });
    });
  }

  // ---------- session ----------

  function onXRFrame(time, frame) {
    if (!session) return;
    session.requestAnimationFrame(onXRFrame);
    const sources = [];
    for (const source of frame.session.inputSources) {
      if (!source.gamepad) continue;
      sources.push({ handedness: source.handedness, pressed: source.gamepad.buttons.map((b) => b.pressed) });
    }
    mapper.update(sources, busy);
  }

  async function onSessionEnd() {
    if (hudState.status === "recording") window.appControls.stopRecording();
    session = null;
    removeHud();
    await window.volumeViewer.exitXR();
    button.textContent = "Enter VR";
  }

  async function micAvailable() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      return true;
    } catch (err) {
      return false;
    }
  }

  async function enter() {
    // The mic prompt cannot be shown inside an immersive session: ask first.
    const voice = await micAvailable();
    let xrSession;
    try {
      xrSession = await navigator.xr.requestSession("immersive-vr");
    } catch (err) {
      // A slow permission prompt can use up the click's user activation.
      showMessage(`Could not start VR (${err.name}). Tap Enter VR again.`);
      return;
    }
    try {
      await window.volumeViewer.enterXR(xrSession);
    } catch (err) {
      xrSession.end().catch(() => {});
      showMessage(`Could not start VR: ${err.message}`);
      return;
    }
    session = xrSession;
    session.addEventListener("end", onSessionEnd, { once: true });
    flatBackground = window.volumeViewer.getRenderer().getBackground().slice();
    hudState = logic.HUD_INITIAL;
    if (!voice) hudState = logic.hudReduce(hudState, { type: "voice-unavailable" });
    addHud();
    button.textContent = "Exit VR";
    session.requestAnimationFrame(onXRFrame);
  }

  function showMessage(text) {
    const errorEl = document.getElementById("viewer-error");
    errorEl.textContent = text;
    errorEl.hidden = false;
    setTimeout(() => { errorEl.hidden = true; }, 6000);
  }

  button.addEventListener("click", () => {
    if (session) session.end();
    else enter();
  });

  if (navigator.xr?.isSessionSupported) {
    navigator.xr.isSessionSupported("immersive-vr")
      .then((supported) => { button.hidden = !supported; })
      .catch(() => {});
  }
})();
```

Notes for the implementer:
- `session.end()` fires `end`, which runs `onSessionEnd`; the system button on the headset fires the same event, so there's one exit path.
- `exitXR()` calls vtk's `stopXR()`, which calls `session.end()` again; vtk catches the resulting `DOMException`, so double-ending is safe.
- The HUD actor is added *after* `enterXR` so vtk.js's `resetXRScene()` (inside `enterXR`) frames only the volume.

- [ ] **Step 3: Mac check (no XR available)**

Reload `http://127.0.0.1:8000` in Chrome: no "Enter VR" button visible; no console errors from `xr.js`; flat UI unchanged.

- [ ] **Step 4: Emulated XR check (optional but recommended before the headset)**

Install the "Immersive Web Emulator" Chrome extension (Meta), reload, open DevTools → WebXR tab: the Enter VR button appears; clicking it enters a session showing the volume and HUD in the emulator. In the emulator, press/hold the right trigger → HUD shows `● Recording…` and background turns red; release → `Transcribing…` then the summary.

- [ ] **Step 5: Commit**

```bash
git add static/xr.js static/index.html
git commit -m "feat(xr): add immersive VR mode with push-to-talk and status HUD"
```

---

### Task 8: On-device verification + docs

**Files:** Modify `docs/vr-integration.md`

- [ ] **Step 1: Run on the Mac for the headset**

```bash
.venv/bin/python server.py --host 0.0.0.0 --port 8443 \
  --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem
```

Headset: VIVE Browser → `https://<mac-lan-ip>:8443` (accept the cert warning or install mkcert's root CA).

- [ ] **Step 2: Walk the checklist, note results**

1. Flat viewer loads `ct_chest` as a local volume (status says `Local ct_chest volume`, not the PNG fallback). If it falls back, stop here and record the error text — volume downsampling for the headset is a separate follow-up.
2. "Enter VR" visible; entering shows the volume in stereo, head tracking works, orientation matches the flat view. Note smoothness (smooth / jittery / slideshow).
3. HUD readable above the volume, not mirrored or upside down. If upside down, swap `origin`/`point2` roles in `hudPlacement` (origin at the top edge, point2 = origin − up·height) and update the Task 4 test to match.
4. Hold trigger → red background + HUD `● Recording…` + pulse; say "show the lungs"; release → `Transcribing…` → `Applying…` → summary; volume changes.
5. A/X steps back, B/Y forward; volume follows; HUD shows `stepped back` / `stepped forward`.
6. Headset system button exits; flat UI intact, camera restored, no HUD in the flat viewer.
7. Mac browser still shows no Enter VR button.

- [ ] **Step 3: Append to `docs/vr-integration.md`**

```markdown
## WebXR route (Vive Focus 3) — supersedes Vrui for headset viewing

Standalone WebXR headsets need no native toolkit: the existing browser
client *is* the VR client. See the design in
`docs/superpowers/specs/2026-09-29-webxr-vr-mode-design.md`.

Setup:

1. `mkcert -cert-file certs/lan.pem -key-file certs/lan-key.pem <mac-lan-ip> localhost`
   (`certs/` is gitignored). WebXR and the microphone need HTTPS off-localhost.
2. `.venv/bin/python server.py --host 0.0.0.0 --port 8443 --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem`
3. Headset (same Wi-Fi): VIVE Browser → `https://<mac-lan-ip>:8443`, accept
   the certificate warning (or install `$(mkcert -CAROOT)/rootCA.pem`).
4. "Enter VR" (next to Reset). Grant the microphone when asked.

In VR: hold either trigger to speak a command, release to send; A/X = back,
B/Y = forward; the headset system button exits. Camera commands have no
effect in VR (head pose drives the view).

On-device results (YYYY-MM-DD, fill in): <checklist outcomes from Task 8>
```

Replace the last line with the actual date and checklist outcomes from Step 2.

- [ ] **Step 4: Commit**

```bash
git add docs/vr-integration.md
git commit -m "docs(vr): document WebXR headset setup and on-device results"
```
