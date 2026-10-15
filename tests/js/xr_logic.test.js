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

test("withXRLayerContext merges layer init overrides over vtk's init", async () => {
  const { root, created } = fakeRoot();
  await xr.withXRLayerContext(root, {}, async () =>
    new root.XRWebGLLayer("s", {}, { antialias: true, framebufferScaleFactor: 1 }),
  { framebufferScaleFactor: 0.5 });
  assert.deepEqual(created[0].init, { antialias: true, framebufferScaleFactor: 0.5 });
});

test("parsePerfFlags defaults reproduce the pre-flag behaviour", () => {
  assert.deepEqual(xr.parsePerfFlags(""), { ...xr.PERF_DEFAULTS });
  assert.deepEqual(xr.PERF_DEFAULTS, {
    tex: "f32", crop: false, down: 1, fov: 0, xrscale: 0.5, xrsample: 1, adapt: false,
  });
});

test("parsePerfFlags reads every lever", () => {
  assert.deepEqual(
    xr.parsePerfFlags("?tex=f16&crop=1&down=2&fov=0.5&xrscale=1&xrsample=2&adapt=1"),
    { tex: "f16", crop: true, down: 2, fov: 0.5, xrscale: 1, xrsample: 2, adapt: true },
  );
});

test("parsePerfFlags falls back to defaults on garbage, never throws", () => {
  assert.deepEqual(
    xr.parsePerfFlags("?tex=u8&crop=yes&down=3&fov=2&xrscale=abc&xrsample=0&adapt=true"),
    { ...xr.PERF_DEFAULTS },
  );
  assert.equal(xr.parsePerfFlags("?xrscale=").xrscale, 0.5);
});

test("diagnosticLines formats a full measurement record", () => {
  const lines = xr.diagnosticLines({
    texture: "R16F", floatLinear: false, dimensions: [256, 256, 140], sampleDistance: 1.6,
    framebufferScale: 1, foveation: "unsupported", shade: false, autoAdjust: true,
    flags: { adapt: true, crop: true, down: 2 },
  }, 72, 13.9);
  assert.deepEqual(lines, [
    "72 fps 13.9ms tex=R16F flin=0 dim=256x256x140 sd=1.60",
    "xrscale=1 fov=unsupported adapt=1 crop=1 down=2 shade=0 auto=1",
  ]);
});

test("diagnosticLines tolerates a viewer that has not loaded yet", () => {
  assert.deepEqual(xr.diagnosticLines({}, null, null), [
    "- fps -ms tex=- flin=0 dim=- sd=-",
    "xrscale=- fov=- adapt=0 crop=0 down=- shade=0 auto=0",
  ]);
});

// ---------- grab-to-rotate ----------

const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const rotZ = (deg) => { const r = (deg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
const rotY = (deg) => { const r = (deg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; };
function assertMat3Close(actual, expected) {
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      assert.ok(Math.abs(actual[r][c] - expected[r][c]) < 1e-9, `[${r}][${c}] ${actual[r][c]} != ${expected[r][c]}`);
    }
  }
}
// Physical frame identical to world: north -z, up +y.
const R_ID = () => xr.physicalRotation([0, 0, -1], [0, 1, 0]);
const rotX = (deg) => { const r = (deg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r); return [[1, 0, 0], [0, c, -s], [0, s, c]]; };
const stick = (handedness, x, y, resetPressed = false) => ({ handedness, stick: [x, y], resetPressed });

test("physicalRotation rows are right, up and -north", () => {
  assertMat3Close(R_ID(), I3);
  assertMat3Close(xr.physicalRotation([0, -1, 0], [0, 0, 1]), [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]);
});

test("stick rotator ignores a resting thumb inside the dead zone", () => {
  const sticks = xr.createStickRotator();
  const out = sticks.update([stick("left", 0.1, -0.1)], R_ID(), 16);
  assert.equal(out.rotation, null);
  assert.equal(out.event, null);
});

test("stick right turns the volume's front to the right at 90 deg/s", () => {
  const sticks = xr.createStickRotator();
  let out;
  for (let i = 0; i < 10; i += 1) out = sticks.update([stick("right", 1, 0)], R_ID(), 100);
  assertMat3Close(out.rotation, rotY(90));
});

test("stick forward tilts the top away from the viewer", () => {
  const sticks = xr.createStickRotator();
  let out;
  for (let i = 0; i < 10; i += 1) out = sticks.update([stick("left", 0, -1)], R_ID(), 100);
  assertMat3Close(out.rotation, rotX(-90));
});

test("a stall doesn't turn into a jump: each step is capped at 100 ms", () => {
  const sticks = xr.createStickRotator();
  assertMat3Close(sticks.update([stick("left", 1, 0)], R_ID(), 5000).rotation, rotY(9));
});

test("turning happens about the viewer's axes, not the volume's", () => {
  // World +z is physical up here, so a left/right push turns about world Z.
  const R = xr.physicalRotation([0, -1, 0], [0, 0, 1]);
  const sticks = xr.createStickRotator();
  let out;
  for (let i = 0; i < 4; i += 1) out = sticks.update([stick("left", 1, 0)], R, 100);
  assertMat3Close(out.rotation, rotZ(36));
});

test("thumbstick click resets the orientation, once per press", () => {
  const sticks = xr.createStickRotator();
  sticks.update([stick("left", 1, 0)], R_ID(), 100);
  const out = sticks.update([stick("left", 0, 0, true)], R_ID(), 16);
  assert.equal(out.event, "reset");
  assertMat3Close(out.rotation, I3);
  assert.equal(sticks.update([stick("left", 0, 0, true)], R_ID(), 16).event, null);
});

test("centeredUserMatrix rotates about the centre (column-major, gl-matrix)", () => {
  const m = xr.centeredUserMatrix(rotZ(90), [1, 0, 0]);
  const apply = (p) => [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
  const close = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
  assert.ok(close(apply([1, 0, 0]), [1, 0, 0]));
  assert.ok(close(apply([2, 0, 0]), [1, 1, 0]));
  assert.equal(m[15], 1);
});

test("rotating the physical frame looks the same as rotating the volume about its centre", () => {
  // vtk.js world -> physical: R (w + t) / s, R = physicalRotation(north, up).
  const toPhysical = (frame, s, w) => {
    const R = xr.physicalRotation(frame.north, frame.up);
    const v = w.map((x, i) => x + frame.translation[i]);
    return R.map((row) => (row[0] * v[0] + row[1] * v[1] + row[2] * v[2]) / s);
  };
  const base = { north: [0, -1, 0], up: [0, 0, 1], translation: [5, -3, 2] };
  const center = [1, 2, 3];
  const W = rotZ(40);
  const turned = xr.rotatedPhysicalFrame(base, W, center);
  for (const w of [[4, -1, 7], [1, 2, 3], [-2, 0.5, 9]]) {
    const rotatedPoint = [0, 1, 2].map((r) =>
      center[r] + W[r][0] * (w[0] - center[0]) + W[r][1] * (w[1] - center[1]) + W[r][2] * (w[2] - center[2]));
    const expected = toPhysical(base, 2, rotatedPoint);
    const actual = toPhysical(turned, 2, w);
    actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-9, `${actual} != ${expected}`));
  }
});

test("rotatedPhysicalFrame with no rotation is the base frame", () => {
  const base = { north: [0, -1, 0], up: [0, 0, 1], translation: [5, -3, 2] };
  const turned = xr.rotatedPhysicalFrame(base, I3, [1, 2, 3]);
  [["north", turned.north, base.north], ["up", turned.up, base.up], ["translation", turned.translation, base.translation]]
    .forEach(([name, a, b]) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-12, name)));
});

// ---------- crop / downsample / 16-bit ----------

const at = (dims, i, j, k) => i + dims[0] * (j + dims[1] * k);
const ramp = (n) => Float32Array.from({ length: n }, (_, i) => i);

test("bodyBoundingBox pads and clamps the non-air box", () => {
  const dims = [10, 8, 6];
  const values = new Float32Array(10 * 8 * 6).fill(-1000);
  values[at(dims, 3, 2, 1)] = 40;
  values[at(dims, 5, 6, 4)] = 900;
  assert.deepEqual(xr.bodyBoundingBox(values, dims, -500, 2), { min: [1, 0, 0], max: [7, 7, 5] });
});

test("bodyBoundingBox keeps the full volume when everything is air", () => {
  const values = new Float32Array(10 * 8 * 6).fill(-1000);
  assert.deepEqual(xr.bodyBoundingBox(values, [10, 8, 6], -500), { min: [0, 0, 0], max: [9, 7, 5] });
});

test("cropVolume copies the inclusive box and keeps the array type", () => {
  const { values, dims } = xr.cropVolume(ramp(24), [4, 3, 2], { min: [1, 1, 0], max: [2, 2, 1] });
  assert.ok(values instanceof Float32Array);
  assert.deepEqual(dims, [2, 2, 2]);
  assert.deepEqual(Array.from(values), [5, 6, 9, 10, 17, 18, 21, 22]);
});

test("strideDownsample keeps every factor-th voxel and rounds odd sizes up", () => {
  const { values, dims } = xr.strideDownsample(ramp(30), [5, 3, 2], 2);
  assert.deepEqual(dims, [3, 2, 1]);
  assert.deepEqual(Array.from(values), [0, 2, 4, 10, 12, 14]);
});

test("strideDownsample by 1 returns the input untouched", () => {
  const input = ramp(6);
  const result = xr.strideDownsample(input, [3, 2, 1], 1);
  assert.equal(result.values, input);
  assert.deepEqual(result.dims, [3, 2, 1]);
});

test("toSixteenBit picks Uint16 for non-negative data, rounding and clamping", () => {
  const out = xr.toSixteenBit(Float32Array.from([0, 35818.4, 70000]));
  assert.ok(out instanceof Uint16Array);
  assert.deepEqual(Array.from(out), [0, 35818, 65535]);
});

test("toSixteenBit picks Int16 when the data goes negative", () => {
  const out = xr.toSixteenBit(Float32Array.from([-1024.6, 3071, 40000]));
  assert.ok(out instanceof Int16Array);
  assert.deepEqual(Array.from(out), [-1025, 3071, 32767]);
});

test("parsePerfFlags takes per-device defaults, and explicit flags still win", () => {
  assert.deepEqual(xr.HEADSET_PERF_DEFAULTS, { ...xr.PERF_DEFAULTS, tex: "f16", down: 2, xrscale: 0.8 });
  assert.deepEqual(xr.parsePerfFlags("", xr.HEADSET_PERF_DEFAULTS), { ...xr.HEADSET_PERF_DEFAULTS });
  const explicit = xr.parsePerfFlags("?tex=f32&down=1&xrscale=1&crop=0&adapt=0", xr.HEADSET_PERF_DEFAULTS);
  assert.equal(explicit.tex, "f32");
  assert.equal(explicit.down, 1);
  assert.equal(explicit.xrscale, 1);
  assert.equal(explicit.crop, false);
  assert.equal(explicit.adapt, false);
});
