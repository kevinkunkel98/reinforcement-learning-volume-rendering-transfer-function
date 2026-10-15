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

  // Camera physicalScale/physicalTranslation that show the volume with a
  // bounding-sphere radius of `radiusM` metres, centred `distanceM` metres
  // along `north` (the viewer's forward). vtk.js maps world -> physical as
  // R((w + translation) / scale). Its own resetXRScene() default makes the
  // radius 1 m at 0.7 m -- the viewer starts inside the volume -- and pushes
  // along world z regardless of which way the view faces.
  function xrPlacement(bounds, north, radiusM, distanceM) {
    const center = [0, 2, 4].map((i) => (bounds[i] + bounds[i + 1]) / 2);
    const radius = Math.hypot(...[0, 2, 4].map((i) => bounds[i + 1] - bounds[i])) / 2;
    const scale = radius / radiusM;
    const n = normalize(north);
    return {
      scale,
      translation: center.map((c, i) => -c + n[i] * distanceM * scale),
    };
  }

  // vtk.js 25's enterXR() does `new XRWebGLLayer(session, get3DContext())`,
  // but get3DContext() returns the GL context wrapped in a state-caching
  // Proxy, which the browser's WebIDL check rejects. While `task` runs,
  // swap in a constructor that substitutes the raw context; only the layer
  // sees it, so vtk's cached GL state stays in sync. `initOverrides` are
  // merged into the layer's init dict (vtk passes none), e.g. a lower
  // framebufferScaleFactor.
  async function withXRLayerContext(root, context, task, initOverrides = {}) {
    const NativeLayer = root.XRWebGLLayer;
    root.XRWebGLLayer = function XRWebGLLayer(session, _proxied, init) {
      return new NativeLayer(session, context, { ...init, ...initOverrides });
    };
    try {
      return await task();
    } finally {
      root.XRWebGLLayer = NativeLayer;
    }
  }

  // Performance levers for headset testing, read from the page URL (see
  // docs/superpowers/specs/2026-10-15-vr-performance-levers-design.md).
  // The defaults reproduce the pre-flag behaviour; anything unparsable or out
  // of range falls back to its default instead of throwing.
  const PERF_DEFAULTS = Object.freeze({
    tex: "f32", crop: false, down: 1, fov: 0, xrscale: 0.5, xrsample: 1, adapt: false,
  });

  function parsePerfFlags(search) {
    const query = new URLSearchParams(search);
    const number = (key, min, max) => {
      const raw = query.get(key);
      const value = raw === null || raw.trim() === "" ? NaN : Number(raw);
      return Number.isFinite(value) && value >= min && value <= max ? value : PERF_DEFAULTS[key];
    };
    return {
      tex: query.get("tex") === "f16" ? "f16" : PERF_DEFAULTS.tex,
      crop: query.get("crop") === "1",
      down: query.get("down") === "2" ? 2 : PERF_DEFAULTS.down,
      fov: number("fov", 0, 1),
      xrscale: number("xrscale", 0.1, 2),
      xrsample: number("xrsample", 0.25, 8),
      adapt: query.get("adapt") === "1",
    };
  }

  // The HUD's two debug lines: measured performance, then the active config,
  // so a photo of the panel is a complete record of one lab measurement.
  function diagnosticLines(diag, fps, frameMs) {
    const flags = diag.flags || {};
    const on = (value) => (value ? 1 : 0);
    const dash = (value) => (value === null || value === undefined ? "-" : value);
    return [
      `${dash(fps)} fps ${frameMs == null ? "-" : frameMs.toFixed(1)}ms tex=${dash(diag.texture)} `
        + `flin=${on(diag.floatLinear)} dim=${diag.dimensions ? diag.dimensions.join("x") : "-"} `
        + `sd=${diag.sampleDistance == null ? "-" : diag.sampleDistance.toFixed(2)}`,
      `xrscale=${dash(diag.framebufferScale)} fov=${dash(diag.foveation)} adapt=${on(flags.adapt)} `
        + `crop=${on(flags.crop)} down=${dash(flags.down)} shade=${on(diag.shade)} auto=${on(diag.autoAdjust)}`,
    ];
  }

  // ---------- thumbstick rotation ----------
  // 3x3 matrices are row-major arrays of rows.

  function mat3Multiply(a, b) {
    return a.map((row) => [0, 1, 2].map((c) => row[0] * b[0][c] + row[1] * b[1][c] + row[2] * b[2][c]));
  }

  function mat3Transpose(m) {
    return [0, 1, 2].map((c) => [m[0][c], m[1][c], m[2][c]]);
  }

  const IDENTITY3 = Object.freeze([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);

  // vtk.js's world -> physical (room) rotation for a camera's physical view
  // north/up (vtkCamera.getWorldToPhysicalMatrix): rows right, up, -north.
  function physicalRotation(north, up) {
    return [cross(north, up), up.slice(), north.map((v) => -v)];
  }

  const STICK = Object.freeze({ degPerSec: 90, deadZone: 0.15, maxStepMs: 100 });

  // Thumbstick turning about the viewer's own axes: left/right yaws about
  // the room's vertical, forward/back tilts the top away/toward the viewer,
  // linear past a dead zone so a resting thumb doesn't drift. A thumbstick
  // click resets. `sources` per frame: [{handedness, stick: [x, y] (y < 0 is
  // forward, per the Gamepad API), resetPressed}]; `physical` is
  // physicalRotation() of the current XR frame. Returns {rotation:
  // world-space 3x3 or null if unchanged, event: "reset" | null}.
  function createStickRotator() {
    let rotation = IDENTITY3;
    const resetWas = new Map();
    const shape = (v) => {
      const magnitude = Math.abs(v);
      return magnitude <= STICK.deadZone ? 0 : Math.sign(v) * (magnitude - STICK.deadZone) / (1 - STICK.deadZone);
    };
    return {
      update(sources, physical, dtMs) {
        let reset = false;
        let x = 0;
        let y = 0;
        for (const source of sources) {
          if (source.resetPressed && !resetWas.get(source.handedness)) reset = true;
          resetWas.set(source.handedness, source.resetPressed);
          x += shape(source.stick[0]);
          y += shape(source.stick[1]);
        }
        if (reset) {
          rotation = IDENTITY3;
          return { rotation, event: "reset" };
        }
        if (x === 0 && y === 0) return { rotation: null, event: null };
        const step = (STICK.degPerSec * Math.PI / 180) * Math.min(dtMs, STICK.maxStepMs) / 1000;
        const yaw = Math.max(-1, Math.min(1, x)) * step;
        const pitch = Math.max(-1, Math.min(1, y)) * step;
        const [cy, sy, cp, sp] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch)];
        // Physical-space turn Q = Ry(yaw) * Rx(pitch), carried into world
        // space and applied on top: W = R^T Q R W.
        const turn = mat3Multiply([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]);
        rotation = mat3Multiply(mat3Multiply(mat3Multiply(mat3Transpose(physical), turn), physical), rotation);
        return { rotation, event: null };
      },
    };
  }

  // The camera physical frame ({north, up, translation}) that shows the scene
  // as if the volume were turned by world-space `rotation` about `center`.
  // vtk.js 25's volume ray caster mis-clips a volume whose own prop matrix is
  // rotated (straight-edged black cuts), so the room turns instead: with
  // world -> physical R (w + t) / s, R' = R W and t' = W^T (c + t) - c.
  function rotatedPhysicalFrame(base, rotation, center) {
    const inverse = mat3Transpose(rotation);
    const apply = (m, v) => m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
    const moved = apply(inverse, center.map((c, i) => c + base.translation[i]));
    return {
      north: apply(inverse, base.north),
      up: apply(inverse, base.up),
      translation: moved.map((v, i) => v - center[i]),
    };
  }

  // vtkProp3D user matrix (gl-matrix column-major, as Prop3D.computeMatrix
  // consumes it) that applies world-space `rotation` about `center`.
  function centeredUserMatrix(rotation, center) {
    const m = new Array(16).fill(0);
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) m[c * 4 + r] = rotation[r][c];
      m[12 + r] = center[r] - (rotation[r][0] * center[0] + rotation[r][1] * center[1] + rotation[r][2] * center[2]);
    }
    m[15] = 1;
    return m;
  }

  return {
    xrSampleDistance, createControllerMapper, HUD_INITIAL, hudReduce, hudLines,
    summarizeStep, hudPlacement, withXRLayerContext, xrPlacement,
    PERF_DEFAULTS, parsePerfFlags, diagnosticLines,
    physicalRotation, createStickRotator, centeredUserMatrix, rotatedPhysicalFrame,
  };
});
