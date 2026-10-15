# Custom VR Ray Marcher (Stage B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A VR-only WebGL2 volume ray marcher with empty-space skipping, separate HU/label textures and flat-view lighting, selectable with `?renderer=custom`, reaching ≥ 72 fps at `xrscale` ≥ 0.8 and `down=1` on the Vive Focus 3.

**Architecture:** Pure, node-tested modules (`tf_table.js`, `brick_grid.js`, `vr_math.js`) hold everything computable on the CPU: the shared label-scoped transfer-function table, brick statistics/occupancy, and matrix math. `vr_renderer.js` owns a separate WebGL2 context, uploads HU (`R16F`), labels (`R8`), TF table and occupancy, and runs its own WebXR frame loop drawing the volume and the HUD quad per eye. `xr.js` picks the vtk or custom path per session; the flat page stays on vtk.js and uses the shared TF table.

**Tech Stack:** WebGL2 (GLSL ES 3.00), WebXR (`XRWebGLLayer`, `local` reference space), vanilla JS, Node 26 `node:test`, headset benchmark over the Firefox remote debugging socket (scratchpad `rdp/` tools).

**Spec:** `docs/superpowers/specs/2026-10-15-vr-raymarcher-design.md`

**Conventions:** Branch `vr-perf-levers`. Commit messages `type(scope): summary`, **no** Co-Authored-By/Claude trailers. JS tests live in `tests/js/*.test.js`; pytest runs them through `tests/test_xr_logic.py`. Volume arrays are Fortran-ordered (x fastest): voxel `(i, j, k)` is at `i + nx * (j + ny * k)`. Matrices passed to WebGL are column-major `Float32Array(16)` (same as WebXR's `view.projectionMatrix`); 3×3 rotations from `xr_logic.js` are row-major arrays of rows.

**Data space:** millimetres, voxel `(i, j, k)` centred at `(i, j, k) * spacing` (vtk's convention, origin 0). Texture coordinate of a point `p`: `(p / spacing + 0.5) / dims`. The volume box is `[0, (dims - 1) * spacing]`.

---

### Task 1: Run every JS test file from pytest

**Files:**
- Modify: `tests/test_xr_logic.py`

- [ ] **Step 1: Run all `tests/js/*.test.js`**

Replace the `subprocess.run` call's argument list

```python
        ["node", "--test", "tests/js/xr_logic.test.js"],
```

with

```python
        ["node", "--test", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "js").glob("*.test.js"))],
```

and update the module docstring's first line to: `"""Runs the node:test suites in tests/js/ (static/*.js logic) so plain `pytest` covers them.`

- [ ] **Step 2: Verify**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_xr_logic.py
git commit -m "test: run every tests/js suite from pytest"
```

---

### Task 2: Shared transfer-function table (`tf_table.js`)

**Files:**
- Create: `static/tf_table.js`
- Create: `tests/js/tf_table.test.js`
- Modify: `static/index.html` (load it before `viewer.js`)
- Modify: `static/viewer.js` (`setTransferFunction` uses it; delete `internalPeak`/`blendPeaks`)

- [ ] **Step 1: Write the failing tests** — `tests/js/tf_table.test.js`:

```js
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/tf_table.test.js`
Expected: FAIL, `Cannot find module '../../static/tf_table.js'`.

- [ ] **Step 3: Implement** — `static/tf_table.js`:

```js
// The label-scoped transfer-function table shared by the flat vtk.js view
// (viewer.js) and the VR ray marcher (vr_renderer.js), so both shade every
// class the same way. Pure; loaded as window.tfTable and via require().
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.tfTable = api;
})(typeof window !== "undefined" ? window : globalThis, () => {
  "use strict";

  const CENTER_RANGE = [-1050, 2000];
  const WIDTH_RANGE = [10, 400];
  const PARAMS_PER_PEAK = 6;
  const SAMPLES = 256;
  // Mirrors transfer.CANONICAL_PEAK_ORDER and transfer.UNLABELLED_PEAKS.
  const PEAK_ORDER = ["lungs", "soft", "liver", "kidneys", "spleen", "heart", "vessels", "skeleton"];
  const UNLABELLED_PEAKS = ["skeleton", "lungs", "soft"];
  // Label id - 1 -> class (viewer.js CLASS_IDS).
  const CLASS_ORDER = ["skeleton", "lungs", "heart", "vessels", "liver", "kidneys", "spleen", "soft"];

  function internalPeak(params, index) {
    const base = index * PARAMS_PER_PEAK;
    const toRange = (value, range) => range[0] + (value + 1) / 2 * (range[1] - range[0]);
    const unit = (value) => (value + 1) / 2;
    return {
      center: toRange(params[base], CENTER_RANGE),
      width: toRange(params[base + 1], WIDTH_RANGE),
      height: unit(params[base + 2]),
      rgb: [unit(params[base + 3]), unit(params[base + 4]), unit(params[base + 5])],
    };
  }

  function blendPeaks(params, hu, peakNames, scale = 1) {
    let alpha = 0;
    let weight = 1e-6;
    const rgb = [0, 0, 0];
    for (const name of peakNames) {
      const peak = internalPeak(params, PEAK_ORDER.indexOf(name));
      const contribution = peak.height * Math.exp(-0.5 * ((hu - peak.center) / peak.width) ** 2);
      alpha += contribution;
      weight += contribution;
      peak.rgb.forEach((value, channel) => { rgb[channel] += contribution * value; });
    }
    return {
      alpha: Math.max(0, Math.min(1, alpha)) * scale,
      rgb: rgb.map((value) => Math.max(0, Math.min(1, value / weight))),
    };
  }

  // Label-scoped rows (transfer.label_scoped_tables): row 0 unlabelled,
  // row id for CLASS_ORDER[id - 1], shaded by that class's peak alone and
  // scaled by its layer opacity. Without labels: one shared HU row.
  // Each row: SAMPLES x {hu, rgb, alpha} evenly over CENTER_RANGE.
  function tfRows(params, layers, labelled) {
    const rows = labelled
      ? [{ peaks: UNLABELLED_PEAKS, scale: 1 }].concat(CLASS_ORDER.map((className) => (
        { peaks: [className], scale: layers?.[className]?.opacity ?? 1 })))
      : [{ peaks: PEAK_ORDER, scale: 1 }];
    return rows.map((row) => Array.from({ length: SAMPLES }, (_, i) => {
      const hu = CENTER_RANGE[0] + i / (SAMPLES - 1) * (CENTER_RANGE[1] - CENTER_RANGE[0]);
      return { hu, ...blendPeaks(params, hu, row.peaks, row.scale) };
    }));
  }

  // RGBA8 texture data, row after row (height = rows.length, width = SAMPLES).
  function tfTextureData(rows) {
    const out = new Uint8Array(rows.length * rows[0].length * 4);
    let o = 0;
    for (const row of rows) {
      for (const { rgb, alpha } of row) {
        out[o++] = Math.round(rgb[0] * 255);
        out[o++] = Math.round(rgb[1] * 255);
        out[o++] = Math.round(rgb[2] * 255);
        out[o++] = Math.round(alpha * 255);
      }
    }
    return out;
  }

  return {
    CENTER_RANGE, WIDTH_RANGE, SAMPLES, PEAK_ORDER, UNLABELLED_PEAKS, CLASS_ORDER,
    tfRows, tfTextureData,
  };
});
```

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/tf_table.test.js`
Expected: all PASS.

- [ ] **Step 5: Load it** — in `static/index.html` add `<script src="/static/tf_table.js"></script>` directly before `<script src="/static/xr_logic.js"></script>`.

- [ ] **Step 6: Use it in `viewer.js`** — delete the functions `internalPeak` and `blendPeaks`, and in `setTransferFunction` replace

```js
    // Label-scoped rows (transfer.label_scoped_tables): row 0 unlabelled,
    // row id for CLASS_ORDER[id - 1], shaded by that class's peak alone and
    // scaled by its layer opacity. Without labels: one shared HU row.
    const rows = hasLabels()
      ? [{ peaks: UNLABELLED_PEAKS, scale: 1 }].concat(CLASS_ORDER.map((className) => (
        { peaks: [className], scale: activeLayers[className]?.opacity ?? 1 })))
      : [{ peaks: PEAK_ORDER, scale: 1 }];
    rows.forEach((row, rowIndex) => {
      for (let i = 0; i < 256; i += 1) {
        const hu = CENTER_RANGE[0] + i / 255 * (CENTER_RANGE[1] - CENTER_RANGE[0]);
        const sample = blendPeaks(params, hu, row.peaks, row.scale);
        const x = hasLabels() ? rowIndex * LABEL_BAND + hu - CENTER_RANGE[0] : hu;
        color.addRGBPoint(x, ...sample.rgb);
        opacity.addPoint(x, sample.alpha);
      }
    });
```

with

```js
    // The same table the VR ray marcher uses (tf_table.js), laid out over
    // the banded scalar: row r covers [r * LABEL_BAND, r * LABEL_BAND + 3050].
    window.tfTable.tfRows(params, activeLayers, hasLabels()).forEach((row, rowIndex) => {
      for (const { hu, rgb, alpha } of row) {
        const x = hasLabels() ? rowIndex * LABEL_BAND + hu - CENTER_RANGE[0] : hu;
        color.addRGBPoint(x, ...rgb);
        opacity.addPoint(x, alpha);
      }
    });
```

Then remove the now-unused constants `WIDTH_RANGE`, `PEAK_ORDER` and `UNLABELLED_PEAKS` from the top of `viewer.js` (keep `CENTER_RANGE`, `TOTAL_PARAMS`, `LABEL_BAND`; `PARAMS_PER_PEAK`/`N_PEAKS` stay only if still referenced — check with `grep -n "PARAMS_PER_PEAK\|N_PEAKS" static/viewer.js` and delete them if the only hits are their declarations).

- [ ] **Step 7: Verify**

Run: `node --check static/viewer.js && .venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`.
Then on the headset (flat page, plain `/`), capture the benchmark-view image with the scratchpad `rdp/bench.js` + `rdp/grab.py` and compare with `bench_baseline.png` from the Stage C session: identical colours and opacity.

- [ ] **Step 8: Commit**

```bash
git add static/tf_table.js tests/js/tf_table.test.js static/index.html static/viewer.js
git commit -m "refactor(viewer): share the label-scoped TF table in tf_table.js"
```

---

### Task 3: Brick statistics and occupancy (`brick_grid.js`)

**Files:**
- Create: `static/brick_grid.js`
- Create: `tests/js/brick_grid.test.js`
- Modify: `static/index.html` (load it after `tf_table.js`)

- [ ] **Step 1: Write the failing tests** — `tests/js/brick_grid.test.js`:

```js
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/brick_grid.test.js`
Expected: FAIL, `Cannot find module '../../static/brick_grid.js'`.

- [ ] **Step 3: Implement** — `static/brick_grid.js`:

```js
// Empty-space skipping for the VR ray marcher: per-brick HU range and label
// set (once per volume), and per-brick visibility under the current
// transfer-function table (on every TF/layer change). Pure; loaded as
// window.brickGrid and via require().
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.brickGrid = api;
})(typeof window !== "undefined" ? window : globalThis, () => {
  "use strict";

  // Per brick of `size`^3 voxels: HU min/max and a bitmask of label ids,
  // over the brick plus a one-voxel apron (linear HU filtering at a brick
  // face mixes in the neighbouring voxels). `labels` may be null.
  function brickStats(hu, labels, dims, size = 8) {
    const [nx, ny, nz] = dims;
    const grid = dims.map((d) => Math.ceil(d / size));
    const count = grid[0] * grid[1] * grid[2];
    const huMin = new Float32Array(count).fill(Infinity);
    const huMax = new Float32Array(count).fill(-Infinity);
    const labelMask = new Uint16Array(count);
    const span = (v, n) => [Math.floor(Math.max(v - 1, 0) / size), Math.floor(Math.min(v + 1, n - 1) / size)];
    let index = 0;
    for (let k = 0; k < nz; k += 1) {
      const [k0, k1] = span(k, nz);
      for (let j = 0; j < ny; j += 1) {
        const [j0, j1] = span(j, ny);
        for (let i = 0; i < nx; i += 1, index += 1) {
          const [i0, i1] = span(i, nx);
          const value = hu[index];
          const bit = 1 << (labels ? labels[index] : 0);
          for (let bk = k0; bk <= k1; bk += 1) {
            for (let bj = j0; bj <= j1; bj += 1) {
              for (let bi = i0; bi <= i1; bi += 1) {
                const b = bi + grid[0] * (bj + grid[1] * bk);
                if (value < huMin[b]) huMin[b] = value;
                if (value > huMax[b]) huMax[b] = value;
                labelMask[b] |= bit;
              }
            }
          }
        }
      }
    }
    return { grid, size, huMin, huMax, labelMask };
  }

  // 255 where some label present in the brick has non-zero alpha anywhere
  // in [huMin, huMax] of its table row (HU clamped to the table's range, as
  // the shader clamps), else 0. Unlabelled tables (one row) use row 0.
  function occupancy(stats, rows, epsilon = 0.5 / 255) {
    const lo = rows[0][0].hu;
    const hi = rows[0][rows[0].length - 1].hu;
    const last = rows[0].length - 1;
    const toIndex = (h) => (Math.max(lo, Math.min(hi, h)) - lo) / (hi - lo) * last;
    const out = new Uint8Array(stats.huMin.length);
    for (let b = 0; b < out.length; b += 1) {
      const mask = stats.labelMask[b];
      if (!mask) continue;
      const i0 = Math.max(0, Math.floor(toIndex(stats.huMin[b])));
      const i1 = Math.min(last, Math.ceil(toIndex(stats.huMax[b])));
      for (let label = 0; label < 16 && !out[b]; label += 1) {
        if (!(mask & (1 << label))) continue;
        const row = rows[Math.min(label, rows.length - 1)];
        for (let i = i0; i <= i1; i += 1) {
          if (row[i].alpha > epsilon) { out[b] = 255; break; }
        }
      }
    }
    return out;
  }

  function occupiedFraction(occ) {
    let n = 0;
    for (const v of occ) if (v) n += 1;
    return occ.length ? n / occ.length : 0;
  }

  return { brickStats, occupancy, occupiedFraction };
});
```

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/brick_grid.test.js`
Expected: all PASS.

- [ ] **Step 5: Load it** — in `static/index.html` add `<script src="/static/brick_grid.js"></script>` directly after the `tf_table.js` tag.

- [ ] **Step 6: Commit**

```bash
git add static/brick_grid.js tests/js/brick_grid.test.js static/index.html
git commit -m "feat(vr): brick statistics and TF-dependent occupancy for empty-space skipping"
```

---

### Task 4: Matrix math (`vr_math.js`)

**Files:**
- Create: `static/vr_math.js`
- Create: `tests/js/vr_math.test.js`
- Modify: `static/index.html` (load it after `brick_grid.js`)

- [ ] **Step 1: Write the failing tests** — `tests/js/vr_math.test.js`:

```js
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/vr_math.test.js`
Expected: FAIL, `Cannot find module '../../static/vr_math.js'`.

- [ ] **Step 3: Implement** — `static/vr_math.js`:

```js
// 4x4 matrix helpers for the VR ray marcher. Column-major Float32Array(16),
// the layout WebGL and WebXR use. Pure; window.vrMath and require().
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.vrMath = api;
})(typeof window !== "undefined" ? window : globalThis, () => {
  "use strict";

  const identity = () => Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c += 1) {
      for (let r = 0; r < 4; r += 1) {
        out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  function invert(m) {
    const inv = new Float64Array(16);
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    return Float32Array.from(inv, (v) => v / det);
  }

  function translation([x, y, z]) {
    const out = identity();
    out[12] = x; out[13] = y; out[14] = z;
    return out;
  }

  function scaling(s) {
    const out = identity();
    out[0] = s; out[5] = s; out[10] = s;
    return out;
  }

  // Row-major 3x3 rotation -> column-major 4x4.
  function fromRotation(r) {
    const out = identity();
    for (let row = 0; row < 3; row += 1) for (let col = 0; col < 3; col += 1) out[col * 4 + row] = r[row][col];
    return out;
  }

  function transformPoint(m, [x, y, z]) {
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [0, 1, 2].map((r) => (m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]) / w);
  }

  // Data (mm) -> room (m): turn the volume about its centre (`rotation`,
  // world space), carry world into the room frame (`base`, world -> room
  // rotation: rows right, up, -north), shrink by `scale` mm per m, and put
  // the centre at `room`. Mirrors vtk's physical frame + Stage C rotation.
  function modelMatrix({ center, scale, room, base, rotation }) {
    let out = translation(center.map((v) => -v));
    out = multiply(fromRotation(rotation), out);
    out = multiply(fromRotation(base), out);
    out = multiply(scaling(1 / scale), out);
    return multiply(translation(room), out);
  }

  // Distance along a ray (unit `dir`) from `origin` (inside the box) to
  // where it leaves the axis-aligned box [lo, hi]. The shader's brick skip
  // mirrors this exactly.
  function rayBoxExit(origin, dir, lo, hi) {
    let exit = Infinity;
    for (let a = 0; a < 3; a += 1) {
      if (dir[a] > 0) exit = Math.min(exit, (hi[a] - origin[a]) / dir[a]);
      else if (dir[a] < 0) exit = Math.min(exit, (lo[a] - origin[a]) / dir[a]);
    }
    return exit;
  }

  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const out = new Float32Array(16);
    out[0] = f / aspect; out[5] = f; out[10] = (far + near) / (near - far); out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }

  function lookAt(eye, target, up) {
    const sub = (a, b) => a.map((v, i) => v - b[i]);
    const norm = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = norm(sub(eye, target));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    return Float32Array.from([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  }

  return {
    identity, multiply, invert, translation, scaling, fromRotation, transformPoint,
    modelMatrix, rayBoxExit, perspective, lookAt,
  };
});
```

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/vr_math.test.js`
Expected: all PASS.

- [ ] **Step 5: Load it** — in `static/index.html` add `<script src="/static/vr_math.js"></script>` after the `brick_grid.js` tag.

- [ ] **Step 6: Commit**

```bash
git add static/vr_math.js tests/js/vr_math.test.js static/index.html
git commit -m "feat(vr): matrix helpers for the custom ray marcher"
```

---

### Task 5: `?renderer=` flag and viewer data hooks

**Files:**
- Modify: `static/xr_logic.js` (`PERF_DEFAULTS.renderer`, parsing)
- Modify: `static/viewer.js` (`getVolumeData`, TF change listeners)
- Test: `tests/js/xr_logic.test.js`

- [ ] **Step 1: Update the tests** — in `tests/js/xr_logic.test.js`, the test "parsePerfFlags defaults reproduce the pre-flag behaviour" asserts the exact `PERF_DEFAULTS` object: add `renderer: "vtk"` to that expected object, and to the expected object of "parsePerfFlags reads every lever" (whose query gets `&renderer=custom` appended and expects `renderer: "custom"`). Append:

```js
test("parsePerfFlags reads the renderer, defaulting to vtk", () => {
  assert.equal(xr.parsePerfFlags("").renderer, "vtk");
  assert.equal(xr.parsePerfFlags("?renderer=custom").renderer, "custom");
  assert.equal(xr.parsePerfFlags("?renderer=webgpu").renderer, "vtk");
});
```

- [ ] **Step 2: Run to verify the new/changed tests fail**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL on the three renderer assertions.

- [ ] **Step 3: Implement** — in `static/xr_logic.js`: add `renderer: "vtk"` to `PERF_DEFAULTS`, and in `parsePerfFlags`'s returned object add `renderer: choice("renderer", { vtk: "vtk", custom: "custom" }),`.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all PASS.

- [ ] **Step 5: Viewer hooks** — in `static/viewer.js`, before `window.volumeViewer = {` add:

```js
  // For the custom VR ray marcher (vr_renderer.js): the raw volume, labels
  // and current TF state, plus the flat view's lighting and orientation so
  // VR looks and faces the same way.
  const tfListeners = new Set();

  function getVolumeData() {
    if (!rawVolumeValues || !volumeMetadata || !volume) return null;
    const property = volume.getProperty();
    camera.orthogonalizeViewUp();
    return {
      hu: rawVolumeValues,
      labels: hasLabels() ? labelValues : null,
      dims: volumeMetadata.dimensions.slice(),
      spacing: volumeMetadata.spacing.slice(),
      params: transferFunction.slice(),
      layers: { ...activeLayers },
      unitDistance: property.getScalarOpacityUnitDistance(0),
      light: {
        ambient: property.getAmbient(), diffuse: property.getDiffuse(),
        specular: property.getSpecular(), specularPower: property.getSpecularPower(),
      },
      north: camera.getDirectionOfProjection().slice(),
      up: camera.getViewUp().slice(),
    };
  }

  function onTransferFunctionChange(listener) {
    tfListeners.add(listener);
    return () => tfListeners.delete(listener);
  }
```

At the end of `setTransferFunction` (after `transferFunction = params.slice();` and the vtk property updates, before the `if (!inXR())` render line) add:

```js
    tfListeners.forEach((listener) => listener(transferFunction, activeLayers, hasLabels()));
```

In the `window.volumeViewer` object add `getVolumeData, onTransferFunctionChange,` after `setXRSampleMultiplier,`.

- [ ] **Step 6: Verify**

Run: `node --check static/viewer.js && .venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`. On the headset: `volumeViewer.getVolumeData()` (through `rdp/page.py`) returns dims, spacing, 48 params and a numeric `unitDistance`.

- [ ] **Step 7: Commit**

```bash
git add static/xr_logic.js static/viewer.js tests/js/xr_logic.test.js
git commit -m "feat(viewer): expose volume data and TF changes for a custom VR renderer"
```

---

### Task 6: The ray marcher — GL resources, shader, offscreen benchmark

**Files:**
- Create: `static/vr_renderer.js`
- Modify: `static/index.html` (load it after `vr_math.js`, before `app.js`)

No node test (WebGL); verified on the headset in Step 3.

- [ ] **Step 1: Write `static/vr_renderer.js`**

```js
// Custom WebGL2 volume ray marcher for VR (docs/superpowers/specs/
// 2026-10-15-vr-raymarcher-design.md). Separate HU (R16F, linear) and label
// (R8, nearest) textures, the shared TF table, empty-space skipping over
// 8^3-voxel bricks and early ray termination. The flat page stays on vtk.js.
(() => {
  "use strict";

  const BRICK = 8;
  const MAX_STEPS = 2048;

  const VOLUME_VS = `#version 300 es
in vec2 aPos;
out vec2 vNdc;
void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

  const VOLUME_FS = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;
uniform sampler3D uHu;
uniform sampler3D uLabel;
uniform sampler3D uOcc;
uniform sampler2D uTf;
uniform mat4 uInvMvp;      // NDC -> data mm
uniform vec3 uDims;
uniform vec3 uSpacing;
uniform vec3 uOccDims;
uniform float uStep;       // mm
uniform float uOpacityExp; // step / opacity unit distance
uniform float uRows;
uniform vec2 uHuRange;
uniform vec3 uEye;         // eye in data mm (headlight)
uniform vec4 uLight;       // ambient, diffuse, specular, specular power
uniform float uSkip;       // 1 = empty-space skipping on
in vec2 vNdc;
out vec4 outColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec2 boxHit(vec3 ro, vec3 inv, vec3 lo, vec3 hi) {
  vec3 t0 = (lo - ro) * inv;
  vec3 t1 = (hi - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
  vec4 nearP = uInvMvp * vec4(vNdc, -1.0, 1.0);
  vec4 farP = uInvMvp * vec4(vNdc, 1.0, 1.0);
  vec3 ro = nearP.xyz / nearP.w;
  vec3 rd = normalize(farP.xyz / farP.w - ro);
  vec3 inv = 1.0 / rd;
  vec3 hi = (uDims - 1.0) * uSpacing;
  vec2 hit = boxHit(ro, inv, vec3(0.0), hi);
  if (hit.y <= max(hit.x, 0.0)) { outColor = vec4(0.0); return; }
  float t = max(hit.x, 0.0) + hash(gl_FragCoord.xy) * uStep;
  vec4 acc = vec4(0.0);
  vec3 texel = 1.0 / uDims;
  for (int i = 0; i < ${MAX_STEPS}; i++) {
    if (t > hit.y || acc.a >= 0.98) break;
    vec3 p = ro + rd * t;
    vec3 vox = p / uSpacing;
    ivec3 iv = clamp(ivec3(floor(vox + 0.5)), ivec3(0), ivec3(uDims) - 1);
    if (uSkip > 0.5) {
      ivec3 b = iv / ${BRICK};
      if (texelFetch(uOcc, b, 0).r < 0.5) {
        vec3 blo = (vec3(b * ${BRICK}) - 0.5) * uSpacing;
        vec3 bhi = blo + float(${BRICK}) * uSpacing;
        vec2 bh = boxHit(p, inv, blo, bhi);
        t += max(bh.y, 0.0) + 0.01 * uStep;
        continue;
      }
    }
    vec3 tc = (vox + 0.5) * texel;
    float hu = texture(uHu, tc).r;
    float row = uRows > 1.5 ? texelFetch(uLabel, iv, 0).r * 255.0 : 0.0;
    float u = clamp((hu - uHuRange.x) / (uHuRange.y - uHuRange.x), 0.0, 1.0);
    vec4 c = texture(uTf, vec2(u * 255.0 / 256.0 + 0.5 / 256.0, (floor(row + 0.5) + 0.5) / uRows));
    if (c.a > 0.002) {
      float a = 1.0 - pow(1.0 - c.a, uOpacityExp);
      vec3 g = vec3(
        texture(uHu, tc + vec3(texel.x, 0.0, 0.0)).r - texture(uHu, tc - vec3(texel.x, 0.0, 0.0)).r,
        texture(uHu, tc + vec3(0.0, texel.y, 0.0)).r - texture(uHu, tc - vec3(0.0, texel.y, 0.0)).r,
        texture(uHu, tc + vec3(0.0, 0.0, texel.z)).r - texture(uHu, tc - vec3(0.0, 0.0, texel.z)).r) / uSpacing;
      float glen = length(g);
      vec3 rgb = c.rgb * (uLight.x + uLight.y);
      if (glen > 1e-3) {
        vec3 n = -g / glen;
        vec3 l = normalize(uEye - p);
        float ndl = abs(dot(n, l));
        rgb = c.rgb * (uLight.x + uLight.y * ndl) + uLight.z * pow(ndl, uLight.w);
      }
      acc.rgb += (1.0 - acc.a) * a * rgb;
      acc.a += (1.0 - acc.a) * a;
    }
    t += uStep;
  }
  outColor = acc;
}`;

  const HUD_VS = `#version 300 es
in vec3 aPos;
in vec2 aUv;
uniform mat4 uMvp;
out vec2 vUv;
void main() { vUv = aUv; gl_Position = uMvp * vec4(aPos, 1.0); }`;

  const HUD_FS = `#version 300 es
precision mediump float;
uniform sampler2D uTex;
in vec2 vUv;
out vec4 outColor;
void main() { outColor = texture(uTex, vUv); }`;

  function compile(gl, vs, fs) {
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(`shader compile failed: ${gl.getShaderInfoLog(shader)}`);
      }
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`shader link failed: ${gl.getProgramInfoLog(program)}`);
    }
    const uniforms = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i += 1) {
      const name = gl.getActiveUniform(program, i).name;
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return { program, uniforms };
  }

  function texture3D(gl, internalFormat, format, type, dims, data, filter) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, internalFormat, dims[0], dims[1], dims[2], 0, format, type, data);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, filter);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
      gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    }
    return tex;
  }

  // One renderer per GL context: textures, programs, TF state.
  function createScene(gl, data, { down = 1 } = {}) {
    const logic = window.xrLogic;
    const max3d = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
    let { hu, labels, dims } = data;
    ({ values: hu, dims } = logic.strideDownsample(hu, data.dims, down));
    if (labels) labels = logic.strideDownsample(labels, data.dims, down).values;
    if (dims.some((d) => d > max3d)) throw new Error(`volume ${dims.join("x")} exceeds MAX_3D_TEXTURE_SIZE ${max3d}`);
    const spacing = data.spacing.map((s) => s * down);
    const huClamped = Float32Array.from(hu, (v) => Math.max(-1050, Math.min(2000, v)));
    const stats = window.brickGrid.brickStats(huClamped, labels, dims, BRICK);

    const volumeProgram = compile(gl, VOLUME_VS, VOLUME_FS);
    const hudProgram = compile(gl, HUD_VS, HUD_FS);
    const quad = gl.createVertexArray();
    gl.bindVertexArray(quad);
    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(volumeProgram.program, "aPos");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    const huTex = texture3D(gl, gl.R16F, gl.RED, gl.FLOAT, dims, huClamped, gl.LINEAR);
    const labelTex = texture3D(gl, gl.R8, gl.RED, gl.UNSIGNED_BYTE, dims,
      labels || new Uint8Array(dims[0] * dims[1] * dims[2]), gl.NEAREST);
    let occTex = null;
    let tfTex = null;
    let rowCount = 1;
    let occupied = 1;

    function setTransferFunction(params, layers, labelled) {
      const rows = window.tfTable.tfRows(params, layers, labelled);
      rowCount = rows.length;
      if (tfTex) gl.deleteTexture(tfTex);
      tfTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tfTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, rows[0].length, rows.length, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        window.tfTable.tfTextureData(rows));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const occ = window.brickGrid.occupancy(stats, rows);
      occupied = window.brickGrid.occupiedFraction(occ);
      if (occTex) gl.deleteTexture(occTex);
      occTex = texture3D(gl, gl.R8, gl.RED, gl.UNSIGNED_BYTE, stats.grid, occ, gl.NEAREST);
    }
    setTransferFunction(data.params, data.layers, Boolean(labels));

    const baseStep = Math.min(...spacing);
    const state = { step: baseStep, skip: true };

    // Draw the volume for one view into the bound framebuffer/viewport.
    // `model`: data mm -> room m; `view`: room -> eye; `proj`: eye -> clip.
    function drawVolume(model, view, proj) {
      const mv = window.vrMath.multiply(view, model);
      const invMvp = window.vrMath.invert(window.vrMath.multiply(proj, mv));
      const eye = window.vrMath.transformPoint(window.vrMath.invert(mv), [0, 0, 0]);
      const { program, uniforms: u } = volumeProgram;
      gl.useProgram(program);
      [[huTex, gl.TEXTURE_3D, "uHu"], [labelTex, gl.TEXTURE_3D, "uLabel"],
        [occTex, gl.TEXTURE_3D, "uOcc"], [tfTex, gl.TEXTURE_2D, "uTf"]].forEach(([tex, target, name], unit) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(target, tex);
        gl.uniform1i(u[name], unit);
      });
      gl.uniformMatrix4fv(u.uInvMvp, false, invMvp);
      gl.uniform3fv(u.uDims, dims);
      gl.uniform3fv(u.uSpacing, spacing);
      gl.uniform1f(u.uStep, state.step);
      gl.uniform1f(u.uOpacityExp, state.step / data.unitDistance);
      gl.uniform1f(u.uRows, rowCount);
      gl.uniform2f(u.uHuRange, -1050, 2000);
      gl.uniform3fv(u.uEye, eye);
      gl.uniform4f(u.uLight, data.light.ambient, data.light.diffuse, data.light.specular, data.light.specularPower);
      gl.uniform1f(u.uSkip, state.skip ? 1 : 0);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(quad);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

    // HUD quad (corners in data mm, placed by `model` without rotation).
    let hud = null;
    function setHud(canvas, corners) {
      const [origin, p1, p2] = corners;
      const p3 = p1.map((v, i) => v + p2[i] - origin[i]);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from([
        ...origin, 0, 1, ...p1, 1, 1, ...p2, 0, 0,
        ...p2, 0, 0, ...p1, 1, 1, ...p3, 1, 0,
      ]), gl.STATIC_DRAW);
      const posLoc = gl.getAttribLocation(hudProgram.program, "aPos");
      const uvLoc = gl.getAttribLocation(hudProgram.program, "aUv");
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 20, 0);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 20, 12);
      gl.bindVertexArray(null);
      const tex = gl.createTexture();
      hud = { canvas, vao, buffer, tex };
      refreshHud();
    }
    function refreshHud() {
      if (!hud) return;
      gl.bindTexture(gl.TEXTURE_2D, hud.tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, hud.canvas);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    function drawHud(model, view, proj) {
      if (!hud) return;
      const { program, uniforms: u } = hudProgram;
      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, hud.tex);
      gl.uniform1i(u.uTex, 0);
      gl.uniformMatrix4fv(u.uMvp, false, window.vrMath.multiply(proj, window.vrMath.multiply(view, model)));
      gl.bindVertexArray(hud.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
    }

    function dispose() {
      [huTex, labelTex, occTex, tfTex, hud?.tex].forEach((t) => t && gl.deleteTexture(t));
      [quadBuffer, hud?.buffer].forEach((b) => b && gl.deleteBuffer(b));
      gl.deleteProgram(volumeProgram.program);
      gl.deleteProgram(hudProgram.program);
    }

    return {
      dims, spacing, baseStep, state, drawVolume, drawHud, setHud, refreshHud,
      setTransferFunction, dispose, get occupied() { return occupied; },
    };
  }

  // Headset benchmark without XR: offscreen canvas, fixed camera looking at
  // the volume centre from -y (as the Stage C rdp/bench.js did), volume
  // filling the view. Returns ms per frame (readPixels-synced) and,
  // optionally, a PNG data URL for screenshot parity.
  function benchmark({ w = 1152, h = 576, n = 8, skip = true, stepScale = 1, down = 1, image = false } = {}) {
    const data = window.volumeViewer.getVolumeData();
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: image });
    const scene = createScene(gl, data, { down });
    scene.state.skip = skip;
    scene.state.step = scene.baseStep * stepScale;
    const center = data.dims.map((d, i) => (d - 1) * data.spacing[i] / 2);
    const V = window.vrMath.lookAt([center[0], center[1] - 1500, center[2]], center, [0, 0, 1]);
    const P = window.vrMath.perspective(Math.PI / 6, w / h, 10, 10000);
    const M = window.vrMath.identity();
    const px = new Uint8Array(4);
    const frame = () => {
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      scene.drawVolume(M, V, P);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    };
    frame(); frame();
    const t0 = performance.now();
    for (let i = 0; i < n; i += 1) frame();
    const ms = +((performance.now() - t0) / n).toFixed(1);
    const result = { ms, occupied: +scene.occupied.toFixed(3), dims: scene.dims, step: scene.state.step };
    if (image) result.image = canvas.toDataURL("image/png");
    scene.dispose();
    return JSON.stringify(result);
  }

  window.vrRenderer = { createScene, benchmark };
})();
```

- [ ] **Step 2: Load it** — in `static/index.html` add `<script src="/static/vr_renderer.js"></script>` after the `vr_math.js` tag.

- [ ] **Step 3: Verify on the headset** (flat page, `ts_s0477`, `/?down=1`):
  - `vrRenderer.benchmark({skip: false})` and `vrRenderer.benchmark({skip: true})` through `rdp/page.py` → no exception; record `ms` and `occupied`. Expected: skip ≤ no-skip, both far below vtk's 27–30 ms baseline at the same size.
  - `vrRenderer.benchmark({image: true})` through `rdp/grab.py` (it returns JSON: write the `image` field) → the torso with ribs, organ colours as in `bench_baseline.png`, no blocks, no stripes.
  - `vrRenderer.benchmark({w: 2304, h: 1152})` → record.

- [ ] **Step 4: Commit**

```bash
git add static/vr_renderer.js static/index.html
git commit -m "feat(vr): WebGL2 ray marcher with empty-space skipping and a headset benchmark"
```

---

### Task 7: WebXR session loop in the custom renderer

**Files:**
- Modify: `static/vr_renderer.js` (add `start`/`stop`/diagnostics)

- [ ] **Step 1: Add the session API** — in `static/vr_renderer.js`, replace `window.vrRenderer = { createScene, benchmark };` with:

```js
  const XR_RADIUS_M = 0.3;
  const XR_DISTANCE_M = 0.9;

  // One immersive session on its own WebGL2 context. `options`: {xrscale,
  // down}. Rotation (thumbstick) and HUD come from xr.js through the same
  // methods the vtk path offers.
  let session = null;

  async function start(xrSession, options) {
    const data = window.volumeViewer.getVolumeData();
    if (!data) throw new Error("no local volume loaded");
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2", { xrCompatible: true, antialias: false });
    if (!gl) throw new Error("WebGL2 unavailable");
    const scene = createScene(gl, data, { down: options.down });
    const space = await xrSession.requestReferenceSpace("local");
    const layer = new XRWebGLLayer(xrSession, gl, { framebufferScaleFactor: options.xrscale, antialias: false });
    xrSession.updateRenderState({ baseLayer: layer, depthNear: 0.05, depthFar: 100 });

    const north = data.north;
    const up = data.up;
    const base = window.xrLogic.physicalRotation(north, up);
    const bounds = [0, 1, 2].flatMap((a) => [0, (data.dims[a] - 1) * data.spacing[a]]);
    const center = [0, 2, 4].map((i) => (bounds[i] + bounds[i + 1]) / 2);
    const radius = Math.hypot(...[0, 2, 4].map((i) => bounds[i + 1] - bounds[i])) / 2;
    const placement = { center, scale: radius / XR_RADIUS_M, room: [0, 0, -XR_DISTANCE_M], base };
    const identity3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let rotation = identity3;
    let background = [0, 0, 0];
    const unsubscribe = window.volumeViewer.onTransferFunctionChange((params, layers, labelled) =>
      scene.setTransferFunction(params, layers, labelled));

    function onFrame(time, frame) {
      if (!session || session.xrSession !== xrSession) return;
      xrSession.requestAnimationFrame(onFrame);
      const pose = frame.getViewerPose(space);
      if (!pose) return;
      gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
      gl.clearColor(...background, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const model = window.vrMath.modelMatrix({ ...placement, rotation });
      const hudModel = window.vrMath.modelMatrix({ ...placement, rotation: identity3 });
      for (const view of pose.views) {
        const vp = layer.getViewport(view);
        gl.viewport(vp.x, vp.y, vp.width, vp.height);
        scene.drawVolume(model, view.transform.inverse.matrix, view.projectionMatrix);
        scene.drawHud(hudModel, view.transform.inverse.matrix, view.projectionMatrix);
      }
    }

    session = {
      xrSession, gl, scene, unsubscribe, bounds, north, up, base,
      setRotation(r) { rotation = r; },
      setBackground(rgb) { background = rgb; },
    };
    xrSession.requestAnimationFrame(onFrame);
  }

  function stop() {
    if (!session) return;
    session.unsubscribe();
    session.scene.dispose();
    session = null;
  }

  window.vrRenderer = {
    createScene, benchmark, start, stop,
    active: () => session !== null,
    // Same surface xr.js uses on window.volumeViewer for the vtk path.
    getPhysicalRotation: () => session.base,
    setVolumeRotation: (rotation) => session?.setRotation(rotation),
    setXRSampleMultiplier: (m) => { if (session) session.scene.state.step = session.scene.baseStep * m; },
    setBackground: (rgb) => session?.setBackground(rgb),
    getHudFrame: () => ({ bounds: session.bounds, up: session.up, dop: session.north }),
    setHud: (canvas, corners) => session?.scene.setHud(canvas, corners),
    refreshHud: () => session?.scene.refreshHud(),
    getXRDiagnostics: () => (session ? {
      xrCapable: true, sampleDistance: session.scene.state.step, autoAdjust: false, shade: true,
      dimensions: session.scene.dims, texture: `R16F+R8 occ=${session.scene.occupied.toFixed(2)}`,
      floatLinear: true, foveation: null, framebufferScale: window.volumeViewer.perfFlags.xrscale,
      flags: window.volumeViewer.perfFlags,
    } : {}),
  };
```

- [ ] **Step 2: Verify**

Run: `node --check static/vr_renderer.js`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add static/vr_renderer.js
git commit -m "feat(vr): run the custom ray marcher in its own WebXR frame loop"
```

---

### Task 8: Route `xr.js` through the selected renderer

**Files:**
- Modify: `static/xr.js`

- [ ] **Step 1: A `target` per session** — near the top state in `static/xr.js` add:

```js
  // The renderer of the running session: window.vrRenderer (?renderer=custom)
  // or window.volumeViewer (vtk.js). Both offer getPhysicalRotation,
  // setVolumeRotation, setXRSampleMultiplier and getXRDiagnostics.
  let target = window.volumeViewer;
```

Replace every in-session use of `window.volumeViewer.getXRDiagnostics()`, `.getPhysicalRotation()`, `.setVolumeRotation(...)` and `.setXRSampleMultiplier(...)` in `xr.js` with `target.` (`grep -n "volumeViewer\.\(getXRDiagnostics\|getPhysicalRotation\|setVolumeRotation\|setXRSampleMultiplier\)" static/xr.js` lists them; leave `perfFlags` on `window.volumeViewer`).

- [ ] **Step 2: HUD for both paths** — in `addHud()`, keep the canvas creation, then branch:

```js
    if (target === window.vrRenderer) {
      const frame = target.getHudFrame();
      const place = logic.hudPlacement(frame.bounds, frame.up, frame.dop, HUD_W / HUD_H);
      hud = { canvas, ctx: canvas.getContext("2d"), refresh: () => target.refreshHud() };
      target.setHud(canvas, [place.origin, place.point1, place.point2]);
      drawHud();
      return;
    }
```

before the existing vtk actor code, and in the vtk branch set `hud = { canvas, ctx: canvas.getContext("2d"), texture, actor, refresh: () => texture.modified() };`. In `drawHud()` replace `hud.texture.modified();` with `hud.refresh();`. In `removeHud()` only remove the vtk actor when `hud.actor` exists:

```js
    if (hud.actor) window.volumeViewer.getRenderer()?.removeActor(hud.actor);
```

- [ ] **Step 3: Background colour while recording** — in `applyEvent`, replace `const renderer = window.volumeViewer.getRenderer();` and its two `renderer.setBackground(...)` calls with a helper:

```js
    const setBackground = (rgb) => (target === window.vrRenderer
      ? target.setBackground(rgb) : window.volumeViewer.getRenderer().setBackground(...rgb));
```

using `setBackground(RECORDING_BG)` and `setBackground(flatBackground)`. In `startSession`, set `flatBackground` with `target === window.vrRenderer ? [0, 0, 0] : window.volumeViewer.getRenderer().getBackground().slice()`.

- [ ] **Step 4: Start/stop** — in `startSession()`, replace

```js
      await window.volumeViewer.enterXR(xrSession);
```

with

```js
      const flags = window.volumeViewer.perfFlags;
      target = window.volumeViewer;
      if (flags.renderer === "custom") {
        try {
          await window.vrRenderer.start(xrSession, { xrscale: flags.xrscale, down: flags.down });
          target = window.vrRenderer;
        } catch (err) {
          console.error("[vr-renderer]", err);
          window.appControls.notify(`Custom VR renderer failed (${err.message}); using vtk.js.`);
        }
      }
      if (target === window.volumeViewer) await window.volumeViewer.enterXR(xrSession);
```

In the early-end branch and in `onSessionEnd`, replace `await window.volumeViewer.exitXR();` with:

```js
      if (target === window.vrRenderer) window.vrRenderer.stop();
      else await window.volumeViewer.exitXR();
```

- [ ] **Step 5: Verify**

Run: `node --check static/xr.js && .venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`.

- [ ] **Step 6: Headset test** — load `/?renderer=custom&down=1&xrscale=0.8` on the headset, the user enters VR:
  - both eyes show the volume and the HUD; placement in front, upright;
  - thumbstick turns the volume, click resets; no wedges; HUD stays put;
  - a voice command changes the render (TF listener path);
  - fps through `rdp/fps.sh` (the probe wraps `getPhysicalRotation`, which xr.js now calls on `target` — point the probe at `window.vrRenderer.getPhysicalRotation` for this path);
  - screenshot both eyes with `adb exec-out screencap`.

- [ ] **Step 7: Commit**

```bash
git add static/xr.js
git commit -m "feat(xr): run VR sessions on the custom ray marcher with ?renderer=custom"
```

---

### Task 9: Measure against the success criteria and pick defaults

**Files:**
- Modify: `static/xr_logic.js` (`HEADSET_PERF_DEFAULTS`) — only if the criteria hold
- Modify: `tests/js/xr_logic.test.js` (the headset-defaults assertion)
- Create: `docs/experiments/2026-10-15-vr-raymarcher-results.md`

- [ ] **Step 1: Record** in the results doc (same table format as the Stage C commit messages): benchmark ms for vtk (`rdp/bench.js`) vs custom (`vrRenderer.benchmark`) at 1152×576 and 2304×1152, `down=1` and `down=2`, skip on/off, plus `occupied`; in-VR fps at `xrscale` 0.8 and 1.0; screenshot parity notes.

- [ ] **Step 2: If criterion 1 holds (≥ 72 fps in VR at `xrscale` ≥ 0.8, `down=1`)**: change `HEADSET_PERF_DEFAULTS` to `{ ...PERF_DEFAULTS, renderer: "custom", down: 1, xrscale: <measured best ≥ 0.8> }` (drop `tex: "f16"`, which only the vtk path reads), update the headset-defaults test to the same object, run `node --test tests/js/xr_logic.test.js`. If it does not hold, keep the defaults and write in the results doc which term dominates (skip on/off delta, step scale, lighting), per the spec.

- [ ] **Step 3: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass except the two pre-existing `tests/test_pretrain.py` determinism failures.

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/2026-10-15-vr-raymarcher-results.md static/xr_logic.js tests/js/xr_logic.test.js
git commit -m "docs(experiments): custom VR ray marcher results; headset defaults"
```
