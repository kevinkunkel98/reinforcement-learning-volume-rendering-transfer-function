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

test("summarizeStep names the method; the moved classes are rows of their own", () => {
  const text = xr.summarizeStep(
    { mode: "policy", class_visibility: { lungs: 0.5, skeleton: 0.2 } },
    { lungs: 0.1, skeleton: 0.2 }, CLASSES, LABELS, SAID);
  assert.equal(text, "the policy answered");
});

test("stepDeltas lists what moved like the chat reply, largest change first", () => {
  const rows = xr.stepDeltas(
    { lungs: 0.5, skeleton: 0.19, heart: 0.05 }, { lungs: 0.1, skeleton: 0.2, heart: 0.05 },
    ["skeleton", "lungs", "heart"], { lungs: "Lungs", skeleton: "Skeleton", heart: "Heart" },
    { lungs: "#d95926", skeleton: "#3987e5" });
  assert.deepEqual(rows, [
    { name: "lungs", label: "Lungs", color: "#d95926", before: 0.1, after: 0.5, text: "10.00 → 50.00%", up: true },
    { name: "skeleton", label: "Skeleton", color: "#3987e5", before: 0.2, after: 0.19, text: "20.00 → 19.00%", up: false },
  ]);
});

test("stepDeltas has nothing without a before, and skips unmeasurable classes", () => {
  assert.deepEqual(xr.stepDeltas({ lungs: 0.5 }, null, CLASSES, LABELS, {}), []);
  assert.deepEqual(xr.stepDeltas({ lungs: null }, { lungs: 0.1 }, CLASSES, LABELS, {}), []);
});

test("the panel keeps the deltas of the last applied step", () => {
  const rows = [{ name: "lungs" }];
  let s = xr.hudReduce(xr.HUD_INITIAL, { type: "applied", summary: "x", deltas: rows });
  assert.deepEqual(s.deltas, rows);
  s = xr.hudReduce(s, { type: "applied", summary: "stepped back" });
  assert.deepEqual(s.deltas, []);
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
    tex: "f32", crop: false, down: 1, fov: 0, xrscale: 0.5, xrsample: 1, adapt: false, renderer: "vtk", fovstep: 1,
  });
});

test("parsePerfFlags reads every lever", () => {
  assert.deepEqual(
    xr.parsePerfFlags("?tex=f16&crop=1&down=2&fov=0.5&xrscale=1&xrsample=2&adapt=1&renderer=custom&fovstep=2.5"),
    { tex: "f16", crop: true, down: 2, fov: 0.5, xrscale: 1, xrsample: 2, adapt: true, renderer: "custom", fovstep: 2.5 },
  );
});

test("parsePerfFlags falls back to defaults on garbage, never throws", () => {
  assert.deepEqual(
    xr.parsePerfFlags("?tex=u8&crop=yes&down=3&fov=2&xrscale=abc&xrsample=0&adapt=true&fovstep=0.5"),
    { ...xr.PERF_DEFAULTS },
  );
  assert.equal(xr.parsePerfFlags("?xrscale=").xrscale, 0.5);
});

test("diagnosticLines formats a full measurement record", () => {
  const lines = xr.diagnosticLines({
    texture: "R16F", floatLinear: false, dimensions: [256, 256, 140], sampleDistance: 1.6,
    framebufferScale: 1, foveation: "unsupported", shade: false, autoAdjust: true,
    flags: { adapt: true, crop: true, down: 2, fovstep: 2 },
  }, 72, 13.9);
  assert.deepEqual(lines, [
    "72 fps 13.9ms tex=R16F flin=0 dim=256x256x140 sd=1.60",
    "xrscale=1 fov=unsupported adapt=1 crop=1 down=2 fstep=2 shade=0 auto=1",
  ]);
});

test("diagnosticLines tolerates a viewer that has not loaded yet", () => {
  assert.deepEqual(xr.diagnosticLines({}, null, null), [
    "- fps -ms tex=- flin=0 dim=- sd=-",
    "xrscale=- fov=- adapt=0 crop=0 down=- fstep=- shade=0 auto=0",
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
  assert.deepEqual(xr.HEADSET_PERF_DEFAULTS, {
    ...xr.PERF_DEFAULTS, tex: "f16", down: 2, xrscale: 1, adapt: true, renderer: "custom", fovstep: 2,
  });
  assert.deepEqual(xr.parsePerfFlags("", xr.HEADSET_PERF_DEFAULTS), { ...xr.HEADSET_PERF_DEFAULTS });
  const explicit = xr.parsePerfFlags("?tex=f32&down=1&xrscale=1&crop=0&adapt=0&renderer=vtk", xr.HEADSET_PERF_DEFAULTS);
  assert.equal(explicit.tex, "f32");
  assert.equal(explicit.down, 1);
  assert.equal(explicit.xrscale, 1);
  assert.equal(explicit.crop, false);
  assert.equal(explicit.adapt, false);
  assert.equal(explicit.renderer, "vtk");
});

// ---------- adaptive sample distance ----------

const MISS = 20; // ms, well over 1000/72 * 1.15
const OK = 11;   // ms, a 90 Hz frame
const runAdapt = (state, frames) => frames.reduce((s, ms) => xr.adaptSampleDistance(s, ms), state);

test("adaptSampleDistance coarsens after three consecutive missed frames", () => {
  assert.equal(runAdapt(xr.ADAPT_INITIAL, [MISS, MISS]).multiplier, 1);
  assert.equal(runAdapt(xr.ADAPT_INITIAL, [MISS, MISS, MISS]).multiplier, 1.25);
});

test("adaptSampleDistance forgives a lone miss among on-time frames", () => {
  assert.equal(runAdapt(xr.ADAPT_INITIAL, [MISS, MISS, OK, MISS]).multiplier, 1);
  assert.equal(runAdapt(xr.ADAPT_INITIAL, Array(10).fill([MISS, ...Array(29).fill(OK)]).flat()).multiplier, 1);
});

test("adaptSampleDistance coarsens when every other frame misses", () => {
  // 45 fps on a 90 Hz display: misses never come three in a row.
  assert.ok(runAdapt(xr.ADAPT_INITIAL, Array(10).fill([MISS, OK]).flat()).multiplier > 1);
});

test("adaptSampleDistance clamps coarsening at maxMul", () => {
  const state = { ...xr.ADAPT_INITIAL, multiplier: 3.9 };
  assert.equal(runAdapt(state, [MISS, MISS, MISS]).multiplier, 4);
});

test("adaptSampleDistance refines slowly after a sustained on-time run", () => {
  const state = { ...xr.ADAPT_INITIAL, multiplier: 2 };
  assert.equal(runAdapt(state, Array(89).fill(OK)).multiplier, 2);
  assert.equal(runAdapt(state, Array(90).fill(OK)).multiplier, 2 / 1.1);
});

test("adaptSampleDistance: isolated misses slow refinement instead of blocking it", () => {
  // 87 fps on a 90 Hz display: one skipped refresh every 30 frames.
  const second = (misses) => Array.from({ length: 90 }, (_, i) => (i % Math.floor(90 / misses) === 0 ? MISS : OK));
  const state = { ...xr.ADAPT_INITIAL, multiplier: 2 };
  // one miss a second still refines within ~2 s ...
  assert.ok(runAdapt(state, [...second(1), ...second(1)]).multiplier < 2);
  // ... three a second hold the step (no ratchet either way)
  assert.equal(runAdapt(state, [...second(3), ...second(3), ...second(3)]).multiplier, 2);
});

test("splitLoad lowers the volume resolution before coarsening the step", () => {
  assert.deepEqual(xr.splitLoad(1), { res: 1, stepMul: 1 });
  const light = xr.splitLoad(1.21);
  assert.equal(light.stepMul, 1);
  assert.ok(Math.abs(light.res - 1 / 1.1) < 1e-9);
  const heavy = xr.splitLoad(2.5);
  assert.equal(heavy.res, 0.8);
  assert.ok(Math.abs(heavy.stepMul - 2.5 * 0.64) < 1e-9);
});

test("splitLoad caps the step however high the load", () => {
  assert.deepEqual(xr.splitLoad(100), { res: 0.8, stepMul: 2 });
  assert.ok(Math.abs(xr.SPLIT_MAX_LOAD - 2 / 0.64) < 1e-9);
});

test("adaptSampleDistance never refines below minMul", () => {
  assert.equal(runAdapt(xr.ADAPT_INITIAL, Array(90).fill(OK)).multiplier, 1);
});

test("parsePerfFlags reads the renderer, defaulting to vtk", () => {
  assert.equal(xr.parsePerfFlags("").renderer, "vtk");
  assert.equal(xr.parsePerfFlags("?renderer=custom").renderer, "custom");
  assert.equal(xr.parsePerfFlags("?renderer=webgpu").renderer, "vtk");
});

test("parseNote shows what the LLM understood and what it may have missed", () => {
  assert.deepEqual(xr.parseNote({ parser_used: "llm", understood: "heart brightness high", parse_warning: null }), {
    understood: "Understood: heart brightness high", warning: "",
  });
  assert.deepEqual(xr.parseNote({
    parser_used: "llm", understood: "less soft tissue", parse_warning: "may have missed: 'moriorta'",
  }), { understood: "Understood: less soft tissue", warning: "⚠ may have missed: 'moriorta'" });
});

test("parseNote stays silent for exact rule-parser answers", () => {
  assert.equal(xr.parseNote({ parser_used: "rule", understood: "more kidneys", parse_warning: null }), null);
  assert.equal(xr.parseNote({ mode: "policy" }), null);
  assert.equal(xr.parseNote({ parser_used: "llm" }), null);
  // A warning is shown whichever parser answered.
  assert.deepEqual(xr.parseNote({ parser_used: "rule", understood: "more liver", parse_warning: "may have missed: 'x'" }),
    { understood: "Understood: more liver", warning: "⚠ may have missed: 'x'" });
});

test("summarizeStep leads with the LLM's reading, then the method", () => {
  const text = xr.summarizeStep(
    { mode: "policy", parser_used: "llm", understood: "less soft tissue", parse_warning: "may have missed: 'moriorta'",
      class_visibility: { lungs: 0.5 } },
    { lungs: 0.1 }, CLASSES, LABELS, SAID);
  assert.equal(text, "Understood: less soft tissue · ⚠ may have missed: 'moriorta' · the policy answered");
});

test("refinedStepApplies only when the refined step is still the one on screen", () => {
  const shown = { id: 7, params: [0] };
  const reply = { refined: true, dataset: "ts_s0477", current: { id: 7, params: [1] } };
  assert.equal(xr.refinedStepApplies(shown, "ts_s0477", reply, 7), true);
  assert.equal(xr.refinedStepApplies(shown, "ts_s0477", { ...reply, refined: false }, 7), false);
  assert.equal(xr.refinedStepApplies({ id: 8 }, "ts_s0477", reply, 7), false);
  assert.equal(xr.refinedStepApplies(shown, "ts_s0477", { ...reply, current: { id: 8 } }, 7), false);
  assert.equal(xr.refinedStepApplies(shown, "ct_chest", reply, 7), false);
  assert.equal(xr.refinedStepApplies(null, "ts_s0477", reply, 7), false);
  assert.equal(xr.refinedStepApplies(shown, "ts_s0477", null, 7), false);
});

test("voiceTimingLine reports the first and the final answer", () => {
  assert.equal(
    xr.voiceTimingLine({ asr: 812.4, first: 1650.2, final: 4890.7, parse: 12, path: "rule" }),
    "[voice-timing] asr=812ms first=1650ms final=4891ms parse=12ms path=rule");
  assert.equal(
    xr.voiceTimingLine({ asr: 800, first: 1500, final: null }),
    "[voice-timing] asr=800ms first=1500ms final=1500ms parse=-ms path=-");
});

test("summarizeStep says a policy answer is still refining, then that it was refined", () => {
  const step = { mode: "policy", class_visibility: { lungs: 0.5 } };
  assert.equal(xr.summarizeStep({ ...step, refining: true }, { lungs: 0.1 }, CLASSES, LABELS, SAID),
    "the policy answered · refining…");
  assert.equal(xr.summarizeStep({ ...step, refining: false, refined: true }, { lungs: 0.1 }, CLASSES, LABELS, SAID),
    "the policy answered · refined");
});
