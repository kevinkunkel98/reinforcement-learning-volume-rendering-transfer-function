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
