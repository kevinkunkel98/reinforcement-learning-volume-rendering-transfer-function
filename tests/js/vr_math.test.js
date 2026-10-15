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
