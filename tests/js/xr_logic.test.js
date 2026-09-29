"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const xr = require("../../static/xr_logic.js");

test("xrSampleDistance uses the finest voxel spacing", () => {
  assert.equal(xr.xrSampleDistance([0.8, 0.8, 2.5]), 0.8);
  assert.equal(xr.xrSampleDistance([1.5, 1.2, 1.3]), 1.2);
});

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

// vtk.js 25 passes its Proxy-wrapped GL context to `new XRWebGLLayer(...)`,
// which browsers reject ("parameter 2 is not of type WebGLRenderingContext").
function fakeRoot() {
  const created = [];
  class NativeLayer {
    constructor(session, context, init) {
      created.push({ session, context, init });
      this.native = true;
    }
  }
  return { root: { XRWebGLLayer: NativeLayer }, NativeLayer, created };
}

test("withXRLayerContext hands XRWebGLLayer the raw context", async () => {
  const { root, NativeLayer, created } = fakeRoot();
  const raw = { raw: true };
  const proxied = { proxied: true };
  const layer = await xr.withXRLayerContext(root, raw, async () =>
    new root.XRWebGLLayer("session", proxied, { antialias: false }));
  assert.deepEqual(created, [{ session: "session", context: raw, init: { antialias: false } }]);
  assert.ok(layer instanceof NativeLayer);
  assert.equal(root.XRWebGLLayer, NativeLayer);
});

test("withXRLayerContext restores XRWebGLLayer when the task throws", async () => {
  const { root, NativeLayer } = fakeRoot();
  await assert.rejects(
    xr.withXRLayerContext(root, {}, async () => { throw new Error("boom"); }),
    /boom/);
  assert.equal(root.XRWebGLLayer, NativeLayer);
});

// vtk.js maps world -> physical (metres) as R((w + translation) / scale), with
// R taking the camera's physicalViewNorth to the viewer's forward direction.
test("xrPlacement puts the volume centre at the given distance along north", () => {
  const bounds = [0, 390, 0, 390, 0, 347.5];
  const north = [0, 1, 0];
  const { scale, translation } = xr.xrPlacement(bounds, north, 0.3, 0.9);
  const radius = Math.hypot(390, 390, 347.5) / 2;
  assert.ok(Math.abs(scale - radius / 0.3) < 1e-9);
  const center = [195, 195, 173.75];
  const physical = center.map((c, i) => (c + translation[i]) / scale);
  [0, 0.9, 0].forEach((v, i) => assert.ok(Math.abs(physical[i] - v) < 1e-9, `${physical}`));
});

test("xrPlacement normalizes north", () => {
  const a = xr.xrPlacement([-1, 1, -1, 1, -1, 1], [0, 0, -2], 0.5, 1);
  const b = xr.xrPlacement([-1, 1, -1, 1, -1, 1], [0, 0, -1], 0.5, 1);
  assert.deepEqual(a, b);
});
