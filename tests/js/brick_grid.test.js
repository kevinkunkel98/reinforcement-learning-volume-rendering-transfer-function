"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const bricks = require("../../static/brick_grid.js");

const at = (dims, i, j, k) => i + dims[0] * (j + dims[1] * k);
// A table row with alpha 1 for HU in [lo, hi], 0 elsewhere, over -1050..2000.
function rowVisibleBetween(lo, hi) {
  return Array.from({ length: 256 }, (_, i) => {
    const hu = -1050 + i / 255 * 3050;
    return { hu, rgb: [1, 1, 1], alpha: hu >= lo && hu <= hi ? 1 : 0 };
  });
}
const hidden = () => rowVisibleBetween(9999, 9999);

test("brickStats grid size rounds up", () => {
  const stats = bricks.brickStats(new Float32Array(10 * 9 * 8), null, [10, 9, 8], 4);
  assert.deepEqual(stats.grid, [3, 3, 2]);
});

test("brickStats records HU range and label bits, including a one-voxel apron", () => {
  const dims = [8, 4, 4];
  const hu = new Float32Array(8 * 4 * 4).fill(-1000);
  const labels = new Uint8Array(8 * 4 * 4);
  hu[at(dims, 4, 1, 1)] = 300;   // first voxel of brick x=1 (brick size 4)
  labels[at(dims, 4, 1, 1)] = 3;
  const stats = bricks.brickStats(hu, labels, dims, 4);
  // brick (1,0,0) holds it directly ...
  assert.equal(stats.huMax[1], 300);
  assert.equal(stats.labelMask[1] & (1 << 3), 1 << 3);
  // ... and brick (0,0,0) sees it through its apron (voxel x=4 is next to x=3).
  assert.equal(stats.huMax[0], 300);
  assert.equal(stats.labelMask[0] & (1 << 3), 1 << 3);
});

test("occupancy marks a brick visible only if a present label's row is visible in its HU range", () => {
  const dims = [8, 4, 4];
  const hu = new Float32Array(8 * 4 * 4).fill(40);
  const labels = new Uint8Array(8 * 4 * 4).fill(5); // liver everywhere
  const stats = bricks.brickStats(hu, labels, dims, 4);
  const rows = Array.from({ length: 9 }, hidden);
  assert.deepEqual(Array.from(bricks.occupancy(stats, rows)), [0, 0]);
  rows[5] = rowVisibleBetween(0, 100);
  assert.deepEqual(Array.from(bricks.occupancy(stats, rows)), [255, 255]);
  rows[5] = rowVisibleBetween(500, 900); // liver visible, but not at this HU
  assert.deepEqual(Array.from(bricks.occupancy(stats, rows)), [0, 0]);
});

test("occupancy is conservative: a brick containing one visible voxel is never empty", () => {
  const dims = [16, 16, 16];
  const hu = new Float32Array(16 ** 3).fill(-1000);
  const labels = new Uint8Array(16 ** 3);
  hu[at(dims, 9, 2, 13)] = 900;
  labels[at(dims, 9, 2, 13)] = 1;
  const stats = bricks.brickStats(hu, labels, dims, 8);
  const rows = Array.from({ length: 9 }, hidden);
  rows[1] = rowVisibleBetween(800, 1000);
  const occ = bricks.occupancy(stats, rows);
  const brickOf = (i, j, k) => Math.floor(i / 8) + 2 * (Math.floor(j / 8) + 2 * Math.floor(k / 8));
  assert.equal(occ[brickOf(9, 2, 13)], 255);
});

test("unlabelled volumes use row 0 for every brick", () => {
  const dims = [4, 4, 4];
  const hu = new Float32Array(64).fill(900);
  const stats = bricks.brickStats(hu, null, dims, 4);
  assert.deepEqual(Array.from(bricks.occupancy(stats, [rowVisibleBetween(800, 1000)])), [255]);
  assert.deepEqual(Array.from(bricks.occupancy(stats, [hidden()])), [0]);
});

test("occupiedFraction reports the share of visible bricks", () => {
  assert.equal(bricks.occupiedFraction(Uint8Array.from([0, 255, 255, 0])), 0.5);
});
