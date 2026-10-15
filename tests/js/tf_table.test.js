"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const tf = require("../../static/tf_table.js");

// Param value v maps to range[0] + (v + 1) / 2 * span; this inverts it.
const toParam = (value, range) => 2 * (value - range[0]) / (range[1] - range[0]) - 1;
function paramsWith(peaks) {
  const params = new Array(48).fill(-1); // every height 0
  for (const [name, { center, width, height, rgb = [1, 1, 1] }] of Object.entries(peaks)) {
    const base = tf.PEAK_ORDER.indexOf(name) * 6;
    params[base] = toParam(center, tf.CENTER_RANGE);
    params[base + 1] = toParam(width, tf.WIDTH_RANGE);
    params[base + 2] = 2 * height - 1;
    rgb.forEach((v, c) => { params[base + 3 + c] = 2 * v - 1; });
  }
  return params;
}
const nearest = (row, hu) => row.reduce((best, s) => (Math.abs(s.hu - hu) < Math.abs(best.hu - hu) ? s : best));

test("labelled table: unlabelled row then one row per class, 256 samples over CENTER_RANGE", () => {
  const rows = tf.tfRows(paramsWith({}), {}, true);
  assert.equal(rows.length, 9);
  rows.forEach((row) => assert.equal(row.length, 256));
  assert.equal(rows[0][0].hu, -1050);
  assert.equal(rows[0][255].hu, 2000);
});

test("unlabelled table is one row blending every peak", () => {
  const rows = tf.tfRows(paramsWith({ liver: { center: 65, width: 45, height: 0.5 } }), {}, false);
  assert.equal(rows.length, 1);
  assert.ok(Math.abs(nearest(rows[0], 65).alpha - 0.5) < 0.01);
});

test("a class row is shaded by its own peak only", () => {
  const rows = tf.tfRows(paramsWith({ liver: { center: 65, width: 45, height: 0.6 } }), {}, true);
  const liverRow = rows[1 + tf.CLASS_ORDER.indexOf("liver")];
  const skeletonRow = rows[1 + tf.CLASS_ORDER.indexOf("skeleton")];
  assert.ok(Math.abs(nearest(liverRow, 65).alpha - 0.6) < 0.01);
  assert.equal(Math.max(...skeletonRow.map((s) => s.alpha)), 0);
});

test("row 0 (unlabelled tissue) uses skeleton, lungs and soft only", () => {
  const rows = tf.tfRows(paramsWith({ liver: { center: 65, width: 45, height: 0.6 } }), {}, true);
  assert.equal(Math.max(...rows[0].map((s) => s.alpha)), 0);
});

test("layer opacity scales its class row", () => {
  const params = paramsWith({ liver: { center: 65, width: 45, height: 0.6 } });
  const rows = tf.tfRows(params, { liver: { opacity: 0.5 } }, true);
  assert.ok(Math.abs(nearest(rows[1 + tf.CLASS_ORDER.indexOf("liver")], 65).alpha - 0.3) < 0.01);
});

test("tfTextureData packs rows as RGBA8, row-major by row then sample", () => {
  const rows = [[{ hu: 0, rgb: [1, 0.5, 0], alpha: 0.25 }, { hu: 1, rgb: [0, 0, 1], alpha: 1 }]];
  assert.deepEqual(Array.from(tf.tfTextureData(rows)), [255, 128, 0, 64, 0, 0, 255, 255]);
});
