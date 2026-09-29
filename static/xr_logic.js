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

  return { xrSampleDistance, createControllerMapper };
});
