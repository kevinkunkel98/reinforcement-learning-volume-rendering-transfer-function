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

  return { xrSampleDistance, createControllerMapper, HUD_INITIAL, hudReduce, hudLines, summarizeStep };
});
