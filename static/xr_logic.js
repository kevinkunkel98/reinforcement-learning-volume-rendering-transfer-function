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
    deltas: [],
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
        return { ...state, summary: event.summary, deltas: event.deltas || [], dataset: event.dataset || state.dataset };
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

  // The classes a step moved, as app.js's appendReply lists them in the chat
  // (same threshold, same two decimals), largest change first: the VR panel
  // draws one row per entry.
  function stepDeltas(now, before, classOrder, classLabel, classColor) {
    if (!before || !now) return [];
    return classOrder
      .filter((c) => before[c] != null && now[c] != null && Math.abs(now[c] - before[c]) >= 0.0001)
      .sort((a, b) => Math.abs(now[b] - before[b]) - Math.abs(now[a] - before[a]))
      .map((c) => ({
        name: c, label: classLabel[c] || c, color: classColor[c] || "#a1a1aa",
        before: before[c], after: now[c],
        text: `${(before[c] * 100).toFixed(2)} → ${(now[c] * 100).toFixed(2)}%`,
        up: now[c] > before[c],
      }));
  }

  // What the parser made of the sentence, when that is worth showing: the
  // LLM interpreted it (it may have guessed on a garbled transcript), or the
  // parser thinks a clause was dropped. Rule-parser answers are exact, so they
  // get null. `understood` / `parse_warning` come from the hybrid parser meta
  // the server spreads into each step.
  function parseNote(step) {
    const warning = step.parse_warning ? `⚠ ${step.parse_warning}` : "";
    if (step.parser_used !== "llm" && !warning) return null;
    const understood = step.understood ? `Understood: ${step.understood}` : "";
    return understood || warning ? { understood, warning } : null;
  }

  // The chat reply's header line for the VR panel: the parser's reading (see
  // parseNote), how the command was answered, the server's note, or that
  // nothing moved.
  function summarizeStep(step, before, classOrder, classLabel, methodSaid) {
    const note = parseNote(step);
    const parts = note ? [note.understood, note.warning].filter(Boolean) : [];
    parts.push(methodSaid[step.mode] || methodSaid.exact);
    if (step.message) parts.push(step.message);
    const moved = stepDeltas(step.class_visibility, before, classOrder, classLabel, {});
    if (!moved.length && !step.message) parts.push("nothing moved measurably");
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
  // docs/superpowers/specs/2026-10-15-vr-performance-levers-design.md, at tag pre-cleanup-2026-10-19).
  // The defaults reproduce the pre-flag behaviour; anything unparsable or out
  // of range falls back to its default instead of throwing.
  const PERF_DEFAULTS = Object.freeze({
    tex: "f32", crop: false, down: 1, fov: 0, xrscale: 0.5, xrsample: 1, adapt: false, renderer: "vtk",
    // Custom renderer only: ray step multiplier at the edge of each eye's
    // view (1 = uniform steps), a shader-side stand-in for the foveation
    // Wolvic doesn't offer.
    fovstep: 1,
  });

  // What a VR-capable (standalone headset) browser gets without flags,
  // from the 2026-10-15 Vive Focus 3 benchmark: frame cost tracks voxel
  // count, not pixel count, so halving the volume per axis (8x fewer
  // voxels) buys ~3x while a sharp 0.8 framebuffer stays cheap. VR sessions
  // run on the custom ray marcher (static/vr_renderer.js): 27-33 fps lit
  // against vtk.js's ~21-25, with linear HU and no wedge clipping;
  // ?renderer=vtk still picks vtk.js. fovstep=2 (2026-10-19, same TF):
  // ~20 % shorter frames than uniform steps on the flat benchmark and in VR,
  // no visible edge; 3 was no faster. Head-on at 0.9 m on a 523-slice body
  // scan: 21-22 fps. down=1 is out of reach: it restarted the VR runtime.
  // 2026-10-21: xrscale=1 for a sharp panel, with adapt on; the volume then
  // renders at 0.8-1.0 of that as load demands (splitLoad).
  const HEADSET_PERF_DEFAULTS = Object.freeze({
    ...PERF_DEFAULTS, tex: "f16", down: 2, xrscale: 1, adapt: true, renderer: "custom", fovstep: 2,
  });

  function parsePerfFlags(search, defaults = PERF_DEFAULTS) {
    const query = new URLSearchParams(search);
    const number = (key, min, max) => {
      const raw = query.get(key);
      const value = raw === null || raw.trim() === "" ? NaN : Number(raw);
      return Number.isFinite(value) && value >= min && value <= max ? value : defaults[key];
    };
    const choice = (key, options) => {
      const raw = query.get(key);
      return Object.prototype.hasOwnProperty.call(options, raw) ? options[raw] : defaults[key];
    };
    return {
      tex: choice("tex", { f32: "f32", f16: "f16" }),
      crop: choice("crop", { 0: false, 1: true }),
      down: choice("down", { 1: 1, 2: 2 }),
      fov: number("fov", 0, 1),
      xrscale: number("xrscale", 0.1, 2),
      xrsample: number("xrsample", 0.25, 8),
      adapt: choice("adapt", { 0: false, 1: true }),
      renderer: choice("renderer", { vtk: "vtk", custom: "custom" }),
      fovstep: number("fovstep", 1, 4),
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
        + `crop=${on(flags.crop)} down=${dash(flags.down)} fstep=${dash(flags.fovstep)} shade=${on(diag.shade)} auto=${on(diag.autoAdjust)}`,
    ];
  }

  // ---------- adaptive sample distance ----------

  // XR sample-distance controller tuning: hold 72 fps (the stage target),
  // coarsen fast after a few missed frames, refine slowly after ~1 s of
  // on-time ones, so quality doesn't visibly pump. A lone miss costs
  // `missPenalty` on-time frames instead of all of them: on a 90 Hz display
  // a few skipped refreshes a second are normal, and resetting on each one
  // made the step only ever coarsen (2026-10-21 headset: 3.0 -> 5.86 mm, never back).
  // Symmetrically an on-time frame shrinks the miss streak by a quarter instead of
  // clearing it, so 45 fps (every other frame late) still coarsens.
  const ADAPT_DEFAULTS = Object.freeze({
    minMul: 1, maxMul: 4, targetMs: 1000 / 72, missFactor: 1.15,
    coarsenAfter: 3, refineAfter: 90, coarsenStep: 1.25, refineStep: 1.1, missPenalty: 30,
  });
  const ADAPT_INITIAL = Object.freeze({ multiplier: 1, missStreak: 0, okStreak: 0 });

  // One controller step for one frame's duration. Pure; the caller applies
  // `multiplier` to the base sample distance when it changes.
  function adaptSampleDistance(state, frameMs, options = {}) {
    const o = { ...ADAPT_DEFAULTS, ...options };
    if (frameMs > o.targetMs * o.missFactor) {
      const missStreak = state.missStreak + 1;
      if (missStreak < o.coarsenAfter) {
        return { ...state, missStreak, okStreak: Math.max(0, state.okStreak - o.missPenalty) };
      }
      return { multiplier: Math.min(o.maxMul, state.multiplier * o.coarsenStep), missStreak: 0, okStreak: 0 };
    }
    const okStreak = state.okStreak + 1;
    const missStreak = state.missStreak * 0.75;
    if (okStreak < o.refineAfter) return { ...state, okStreak, missStreak };
    return { multiplier: Math.max(o.minMul, state.multiplier / o.refineStep), missStreak: 0, okStreak: 0 };
  }

  // How the custom VR renderer spends a load multiplier (cost reduction
  // asked of it, >= 1): volume resolution first, down to SPLIT_RES_MIN, then
  // a longer ray step, capped at SPLIT_STEP_MAX. Headset benchmark
  // 2026-10-21: 1.0 -> 0.8 resolution cut 28 %, 0.8 -> 0.6 nothing (texture
  // cache), and steps beyond 2x look like grain, so neither goes further.
  const SPLIT_RES_MIN = 0.8;
  const SPLIT_STEP_MAX = 2;
  const SPLIT_MAX_LOAD = SPLIT_STEP_MAX / (SPLIT_RES_MIN * SPLIT_RES_MIN);

  function splitLoad(load) {
    const res = Math.max(SPLIT_RES_MIN, Math.min(1, 1 / Math.sqrt(Math.max(1, load))));
    const stepMul = Math.max(1, Math.min(SPLIT_STEP_MAX, load * res * res));
    return { res, stepMul };
  }

  // ---------- volume reduction for the headset ----------

  // Inclusive voxel-index box of every value above `threshold` in a
  // Fortran-ordered (x fastest) volume, padded by `pad` voxels and clamped
  // to the volume. A volume with nothing above it keeps the full box.
  function bodyBoundingBox(values, dims, threshold, pad = 2) {
    const [nx, ny, nz] = dims;
    const min = [nx, ny, nz];
    const max = [-1, -1, -1];
    let index = 0;
    for (let k = 0; k < nz; k += 1) {
      for (let j = 0; j < ny; j += 1) {
        for (let i = 0; i < nx; i += 1, index += 1) {
          if (!(values[index] > threshold)) continue;
          if (i < min[0]) min[0] = i;
          if (i > max[0]) max[0] = i;
          if (j < min[1]) min[1] = j;
          if (j > max[1]) max[1] = j;
          if (k < min[2]) min[2] = k;
          if (k > max[2]) max[2] = k;
        }
      }
    }
    if (max[0] < 0) return { min: [0, 0, 0], max: [nx - 1, ny - 1, nz - 1] };
    return {
      min: min.map((value) => Math.max(0, value - pad)),
      max: max.map((value, axis) => Math.min(dims[axis] - 1, value + pad)),
    };
  }

  // Copies the inclusive `box` out of a Fortran-ordered volume.
  function cropVolume(values, dims, box) {
    const [nx, ny] = dims;
    const out = box.max.map((value, axis) => value - box.min[axis] + 1);
    const result = new values.constructor(out[0] * out[1] * out[2]);
    let dst = 0;
    for (let k = box.min[2]; k <= box.max[2]; k += 1) {
      for (let j = box.min[1]; j <= box.max[1]; j += 1) {
        const src = box.min[0] + nx * (j + ny * k);
        result.set(values.subarray(src, src + out[0]), dst);
        dst += out[0];
      }
    }
    return { values: result, dims: out };
  }

  // Nearest-neighbour: keeps every `factor`-th voxel per axis, so a label
  // band never blends into a neighbour's (averaging would invent values
  // between bands, i.e. a third class).
  function strideDownsample(values, dims, factor) {
    if (factor === 1) return { values, dims };
    const [nx, ny] = dims;
    const out = dims.map((size) => Math.ceil(size / factor));
    const result = new values.constructor(out[0] * out[1] * out[2]);
    let dst = 0;
    for (let k = 0; k < out[2]; k += 1) {
      for (let j = 0; j < out[1]; j += 1) {
        const row = nx * (j * factor + ny * k * factor);
        for (let i = 0; i < out[0]; i += 1) result[dst++] = values[row + i * factor];
      }
    }
    return { values: result, dims: out };
  }

  // The narrowest 16-bit integer array that holds the data, for vtk.js to
  // upload as a half-float (R16F) texture with preferSizeOverAccuracy. CT HU
  // and our label bands are integer-valued, so rounding loses nothing;
  // anything past the 16-bit range is clamped (typed arrays would wrap).
  function toSixteenBit(values) {
    let min = Infinity;
    for (let index = 0; index < values.length; index += 1) {
      if (values[index] < min) min = values[index];
    }
    const signed = min < 0;
    const [lo, hi] = signed ? [-32768, 32767] : [0, 65535];
    const out = signed ? new Int16Array(values.length) : new Uint16Array(values.length);
    for (let index = 0; index < values.length; index += 1) {
      out[index] = Math.min(hi, Math.max(lo, Math.round(values[index])));
    }
    return out;
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
    summarizeStep, parseNote, stepDeltas, hudPlacement, withXRLayerContext, xrPlacement,
    PERF_DEFAULTS, HEADSET_PERF_DEFAULTS, parsePerfFlags, diagnosticLines,
    physicalRotation, createStickRotator, centeredUserMatrix, rotatedPhysicalFrame,
    bodyBoundingBox, cropVolume, strideDownsample, toSixteenBit,
    ADAPT_DEFAULTS, ADAPT_INITIAL, adaptSampleDistance, SPLIT_MAX_LOAD, splitLoad,
  };
});
