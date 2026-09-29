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

  return {
    xrSampleDistance, createControllerMapper, HUD_INITIAL, hudReduce, hudLines,
    summarizeStep, hudPlacement, withXRLayerContext, xrPlacement,
  };
});
