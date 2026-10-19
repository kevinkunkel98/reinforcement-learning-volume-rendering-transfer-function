"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const m = require("../../static/vr_math.js");

// Matrices are Float32Array, so compare at float32 precision.
const close = (a, b, eps = 1e-5) => a.every((v, i) => Math.abs(v - b[i]) < eps);
const rotZ = (deg) => { const r = (deg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };

test("invert(M) * M is the identity", () => {
  const M = m.multiply(m.translation([1, -2, 3]), m.multiply(m.fromRotation(rotZ(30)), m.scaling(2)));
  assert.ok(close(Array.from(m.multiply(m.invert(M), M)), Array.from(m.identity())));
});

test("transformPoint applies translation and the w divide", () => {
  assert.ok(close(m.transformPoint(m.translation([1, 2, 3]), [1, 1, 1]), [2, 3, 4]));
});

test("modelMatrix puts the data centre at the room position and scales mm to m", () => {
  const M = m.modelMatrix({
    center: [100, 200, 300], scale: 1000, room: [0, 0, -0.9],
    base: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  });
  assert.ok(close(m.transformPoint(M, [100, 200, 300]), [0, 0, -0.9]));
  assert.ok(close(m.transformPoint(M, [1100, 200, 300]), [1, 0, -0.9]));
});

test("modelMatrix applies the volume rotation before the base frame", () => {
  // base maps world +z to room up (+y); a 90 deg turn about world z then
  // swings world +x to world +y, which base maps to room -z.
  const base = [[1, 0, 0], [0, 0, 1], [0, -1, 0]];
  const M = m.modelMatrix({ center: [0, 0, 0], scale: 1, room: [0, 0, 0], base, rotation: rotZ(90) });
  assert.ok(close(m.transformPoint(M, [1, 0, 0]), [0, 0, -1]));
});

test("rayBoxExit is the nearest exit face along the ray", () => {
  assert.equal(m.rayBoxExit([0.5, 0.5, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1]), 0.5);
  assert.ok(Math.abs(m.rayBoxExit([0.5, 0.5, 0.5], [1, 1, 0].map((v) => v / Math.SQRT2), [0, 0, 0], [1, 1, 1]) - 0.5 * Math.SQRT2) < 1e-9);
  assert.equal(m.rayBoxExit([0.2, 0.5, 0.5], [-1, 0, 0], [0, 0, 0], [1, 1, 1]), 0.2);
});

test("perspective and lookAt project the target to the screen centre", () => {
  const P = m.perspective(Math.PI / 4, 2, 0.1, 100);
  const V = m.lookAt([0, -10, 0], [0, 0, 0], [0, 0, 1]);
  const ndc = m.transformPoint(m.multiply(P, V), [0, 0, 0]);
  assert.ok(Math.abs(ndc[0]) < 1e-5 && Math.abs(ndc[1]) < 1e-5);
});

const rotY = (deg) => { const r = (deg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; };
// A viewer pose (eye -> room): standing at `pos`, turned by `deg` about +y,
// pitched by `pitch` degrees about +x.
const pose = (pos, deg, pitch = 0) => {
  const p = (pitch * Math.PI) / 180, c = Math.cos(p), s = Math.sin(p);
  return m.multiply(m.translation(pos), m.multiply(m.fromRotation(rotY(deg)), m.fromRotation([[1, 0, 0], [0, c, -s], [0, s, c]])));
};

test("placeInFront puts the volume ahead of the viewer at head height, facing them", () => {
  const ahead = m.placeInFront(pose([0, 1.6, 0], 0), 0.9);
  assert.ok(close(ahead.room, [0, 1.6, -0.9]), `${ahead.room}`);
  assert.ok(Math.abs(ahead.yaw) < 1e-9);
  const left = m.placeInFront(pose([1, 1.6, 2], 90), 0.9);  // turned to face -x
  assert.ok(close(left.room, [0.1, 1.6, 2]), `${left.room}`);
  assert.ok(Math.abs(left.yaw - Math.PI / 2) < 1e-6, `${left.yaw}`);
});

test("placeInFront ignores head pitch, and looking straight down still faces -z", () => {
  const tilted = m.placeInFront(pose([0, 1.6, 0], 0, -40), 0.9);
  assert.ok(close(tilted.room, [0, 1.6, -0.9]), `${tilted.room}`);
  const down = m.placeInFront(pose([0, 1.6, 0], 0, -90), 0.9);
  assert.ok(close(down.room, [0, 1.6, -0.9]), `${down.room}`);
});

test("modelMatrix turns the placed volume by yaw about the room's vertical", () => {
  const M = m.modelMatrix({
    center: [0, 0, 0], scale: 1000, room: [0, 0, 0], base: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], yaw: Math.PI / 2,
  });
  // +x data (1000 mm) ends up along -z in the room after a 90° turn left
  assert.ok(close(m.transformPoint(M, [1000, 0, 0]), [0, 0, -1]), `${m.transformPoint(M, [1000, 0, 0])}`);
});
