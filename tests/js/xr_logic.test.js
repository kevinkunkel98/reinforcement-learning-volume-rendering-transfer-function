"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const xr = require("../../static/xr_logic.js");

test("xrSampleDistance uses the finest voxel spacing", () => {
  assert.equal(xr.xrSampleDistance([0.8, 0.8, 2.5]), 0.8);
  assert.equal(xr.xrSampleDistance([1.5, 1.2, 1.3]), 1.2);
});
