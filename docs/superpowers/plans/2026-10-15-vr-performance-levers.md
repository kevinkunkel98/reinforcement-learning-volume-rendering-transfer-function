# VR Performance Levers (Stage C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each headset performance lever (texture format, air crop, downsample, fixed foveation, framebuffer scale, adaptive sampling) switchable by URL flag and measurable on the in-VR HUD, and stop re-uploading the volume on every transfer-function change.

**Architecture:** Pure, DOM-free logic (flag parsing, crop/downsample, 16-bit conversion, foveation, the adaptive controller, HUD formatting) goes into `static/xr_logic.js` and is unit-tested with `node:test`. `static/viewer.js` applies the levers when building the vtk.js image data and entering XR; `static/xr.js` runs the controller in the XR frame loop and draws the diagnostics. Defaults reproduce today's behaviour exactly.

**Tech Stack:** vanilla JS, vtk.js 25.11 (WebGL2, WebXR), Node 26 `node:test` (run directly or through pytest's `tests/test_xr_logic.py`).

**Spec:** `docs/superpowers/specs/2026-10-15-vr-performance-levers-design.md`

**Conventions:** No JS toolchain; tests are `tests/js/xr_logic.test.js`. Run them with `node --test tests/js/xr_logic.test.js`. Commit messages follow the repo's `type(scope): summary` style and carry **no** Co-Authored-By / Claude trailers. Use `command cat` (not `cat`) in heredocs.

**Volume layout reminder:** volume arrays are Fortran-ordered (x fastest): voxel `(i, j, k)` lives at `i + nx * (j + ny * k)`, which is also vtkImageData's order.

---

### Task 1: URL performance flags

**Files:**
- Modify: `static/xr_logic.js` (add `PERF_DEFAULTS`, `parsePerfFlags`, export them)
- Modify: `static/index.html:206-210` (load `xr_logic.js` before `viewer.js`)
- Modify: `static/viewer.js:398-407` (read `xrscale`/`xrsample` from the parsed flags), `static/viewer.js:603-630` (expose `perfFlags`)
- Test: `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write the failing tests** (append to `tests/js/xr_logic.test.js`)

```js
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `xr.parsePerfFlags is not a function`.

- [ ] **Step 3: Implement** — in `static/xr_logic.js`, insert before the final `return {`:

```js
  // Performance levers for headset testing, read from the page URL (see
  // docs/superpowers/specs/2026-10-15-vr-performance-levers-design.md).
  // The defaults reproduce the pre-flag behaviour; anything unparsable or out
  // of range falls back to its default instead of throwing.
  const PERF_DEFAULTS = Object.freeze({
    tex: "f32", crop: false, down: 1, fov: 0, xrscale: 0.5, xrsample: 1, adapt: false,
  });

  function parsePerfFlags(search) {
    const query = new URLSearchParams(search);
    const number = (key, min, max) => {
      const raw = query.get(key);
      const value = raw === null || raw.trim() === "" ? NaN : Number(raw);
      return Number.isFinite(value) && value >= min && value <= max ? value : PERF_DEFAULTS[key];
    };
    return {
      tex: query.get("tex") === "f16" ? "f16" : PERF_DEFAULTS.tex,
      crop: query.get("crop") === "1",
      down: query.get("down") === "2" ? 2 : PERF_DEFAULTS.down,
      fov: number("fov", 0, 1),
      xrscale: number("xrscale", 0.1, 2),
      xrsample: number("xrsample", 0.25, 8),
      adapt: query.get("adapt") === "1",
    };
  }
```

and extend the export:

```js
  return {
    xrSampleDistance, createControllerMapper, HUD_INITIAL, hudReduce, hudLines,
    summarizeStep, hudPlacement, withXRLayerContext, xrPlacement,
    PERF_DEFAULTS, parsePerfFlags,
  };
```

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all tests PASS.

- [ ] **Step 5: Load `xr_logic.js` before `viewer.js`** — `viewer.js` now parses flags at load time. In `static/index.html` change

```html
  <script src="/static/viewer.js"></script>
  <script src="/static/xr_logic.js"></script>
```

to

```html
  <script src="/static/xr_logic.js"></script>
  <script src="/static/viewer.js"></script>
```

(`xr_logic.js` is DOM-free, so moving it earlier is safe.)

- [ ] **Step 6: Use the flags in `viewer.js`** — right after `const LABEL_BAND = 4096;` (line 14) add:

```js
  // Headset performance levers (?tex ?crop ?down ?fov ?xrscale ?xrsample
  // ?adapt), parsed once; defaults reproduce the pre-flag behaviour.
  const PERF = window.xrLogic.parsePerfFlags(window.location.search);
```

Replace the two constants at lines ~398-407:

```js
  const XR_FRAMEBUFFER_SCALE = Number(new URLSearchParams(window.location.search).get("xrscale")) || 0.5;
```
→
```js
  const XR_FRAMEBUFFER_SCALE = PERF.xrscale;
```
and
```js
  const XR_SAMPLE_DISTANCE_SCALE = Number(new URLSearchParams(window.location.search).get("xrsample")) || 1;
```
→
```js
  const XR_SAMPLE_DISTANCE_SCALE = PERF.xrsample;
```

(keep the comments above them). In the `window.volumeViewer = { ... }` object add, after `enterXR, exitXR, inXR,`:

```js
    perfFlags: PERF,
```

- [ ] **Step 7: Run the full JS suite through pytest**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`.

- [ ] **Step 8: Commit**

```bash
git add static/xr_logic.js static/index.html static/viewer.js tests/js/xr_logic.test.js
git commit -m "feat(xr): parse headset performance levers from URL flags"
```

---

### Task 2: Measurement HUD (frame time, texture format, config)

**Files:**
- Modify: `static/xr_logic.js` (add `diagnosticLines`, export)
- Modify: `static/viewer.js` (texture-format/float-linear helpers, `xrFoveation`, extend `getXRDiagnostics`)
- Modify: `static/xr.js` (`HUD_H`, frame time, two debug lines, console log)
- Test: `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write the failing tests**

```js
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `xr.diagnosticLines is not a function`.

- [ ] **Step 3: Implement `diagnosticLines`** in `static/xr_logic.js` (before `return {`):

```js
  // The HUD's two debug lines: measured performance, then the active config,
  // so a photo of the panel is a complete record of one lab measurement.
  function diagnosticLines(diag, fps, frameMs) {
    const flags = diag.flags || {};
    const on = (value) => (value ? 1 : 0);
    const dash = (value) => (value === null || value === undefined ? "-" : value);
    return [
      `${dash(fps)} fps ${frameMs == null ? "-" : frameMs.toFixed(1)}ms tex=${dash(diag.texture)} `
        + `flin=${on(diag.floatLinear)} dim=${diag.dimensions ? diag.dimensions.join("x") : "-"} `
        + `sd=${diag.sampleDistance == null ? "-" : diag.sampleDistance.toFixed(2)}`,
      `xrscale=${dash(diag.framebufferScale)} fov=${dash(diag.foveation)} adapt=${on(flags.adapt)} `
        + `crop=${on(flags.crop)} down=${dash(flags.down)} shade=${on(diag.shade)} auto=${on(diag.autoAdjust)}`,
    ];
  }
```

Add `diagnosticLines` to the exported object.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all PASS.

- [ ] **Step 5: Viewer diagnostics** — in `static/viewer.js`, after the `let isXRCapableDevice = false;` block add:

```js
  // Fixed-foveation level actually applied in the running XR session (or
  // "unsupported"); null outside XR.
  let xrFoveation = null;
  let floatLinear;
```

In `destroyViewer()` (next to the other resets) add `floatLinear = undefined;`.

Before `window.volumeViewer = {` add:

```js
  // The 3D texture vtk.js 25.11 builds from our scalars under WebGL2:
  // Float32 is uploaded as R32F; 16-bit integers become R16F (half float)
  // with preferSizeOverAccuracy on (Rendering/OpenGL/Texture.js,
  // create3DFilterableFromRaw). vtk.js doesn't expose the texture itself.
  function textureFormat() {
    const type = imageData?.getPointData().getScalars()?.getDataType();
    if (!type) return null;
    const sixteenBit = type === "Uint16Array" || type === "Int16Array";
    return sixteenBit && mapper.getPreferSizeOverAccuracy() ? "R16F" : "R32F";
  }

  // Whether this GPU can linearly filter float textures natively.
  function floatLinearSupported() {
    if (floatLinear === undefined && openGLRenderWindow) {
      const gl = openGLRenderWindow.getCanvas()
        .getContext(openGLRenderWindow.getWebgl2() ? "webgl2" : "webgl");
      floatLinear = Boolean(gl?.getExtension("OES_texture_float_linear"));
    }
    return floatLinear;
  }
```

Replace the whole `getXRDiagnostics: () => ({ ... }),` entry with:

```js
    getXRDiagnostics: () => ({
      xrCapable: isXRCapableDevice,
      sampleDistance: mapper?.getSampleDistance(),
      autoAdjust: mapper?.getAutoAdjustSampleDistances(),
      shade: volume?.getProperty().getShade(),
      dimensions: imageData?.getDimensions(),
      texture: textureFormat(),
      floatLinear: floatLinearSupported(),
      foveation: xrFoveation,
      framebufferScale: inXR() ? XR_FRAMEBUFFER_SCALE : null,
      flags: PERF,
    }),
```

- [ ] **Step 6: HUD in `static/xr.js`** — change `const HUD_H = 384;` to `const HUD_H = 420;`. Add `let frameMs = null;` next to `let fps = null;`.

In `drawHud()`, delete the block

```js
    if (fps !== null) {
      ctx.textAlign = "right";
      ctx.fillText(`${fps} fps`, HUD_W - pad, 300);
      ctx.textAlign = "left";
    }
```

and replace everything from the `// Debug line:` comment down to (but not including) `hud.texture.modified();` with:

```js
    // Debug lines: real numbers from this exact session plus the active
    // lever config, so a photo of the panel is a complete measurement.
    const [perf, config] = logic.diagnosticLines(window.volumeViewer.getXRDiagnostics(), fps, frameMs);
    ctx.fillStyle = "#52525b";
    ctx.font = "26px monospace";
    ctx.fillText(wrapLine(ctx, perf, max), pad, 340);
    ctx.fillText(wrapLine(ctx, config, max), pad, 376);
```

In `onXRFrame`, replace

```js
      fps = Math.round((frames * 1000) / (time - fpsSince));
      frames = 0;
      fpsSince = time;
      drawHud();
```

with

```js
      fps = Math.round((frames * 1000) / (time - fpsSince));
      frameMs = (time - fpsSince) / frames;
      frames = 0;
      fpsSince = time;
      drawHud();
      // Same record for remote DevTools / logcat captures.
      console.info("[xr-perf]", ...logic.diagnosticLines(window.volumeViewer.getXRDiagnostics(), fps, frameMs));
```

In `startSession()`, next to `fps = null;` add `frameMs = null;`.

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`.

- [ ] **Step 8: Manual check (desktop)**

Run: `.venv/bin/python server.py`, open `http://127.0.0.1:8000`, load a dataset. In DevTools console: `volumeViewer.getXRDiagnostics()` returns `texture: "R32F"`, `dimensions` equal to the dataset's, `flags` equal to the defaults, no errors on the page.

- [ ] **Step 9: Commit**

```bash
git add static/xr_logic.js static/viewer.js static/xr.js tests/js/xr_logic.test.js
git commit -m "feat(xr): show frame time, texture format and lever config on the VR HUD"
```

> **Lab checkpoint:** after this task the baseline row of the lab matrix (Task 10) can be measured on the headset.

---

### Task 3: Build the volume texture once per volume, not per TF change

**Files:**
- Modify: `static/viewer.js` (`buildVolumeImage`, `builtFrom`, `setLabelAwareTransferFunction`, load path, `destroyViewer`)

No unit test: this is vtk.js/DOM glue with no node harness. Verified manually in Step 4.

- [ ] **Step 1: Add the builder** — in `static/viewer.js`, directly after `function buildImageData(...) { ... }` add:

```js
  // Identity of the inputs the current imageData was built from: rebanding
  // and re-uploading the whole volume takes seconds on a headset, and a
  // transfer-function change alone needs neither.
  let builtFrom = { raw: null, labels: null };

  function buildVolumeImage() {
    const values = hasLabels()
      ? bandLabeledHuValues(rawVolumeValues, labelValues, labelMetadata, volumeMetadata)
      : rawVolumeValues;
    buildImageData(volumeMetadata, values);
    mapper.setInputData(imageData);
    builtFrom = { raw: rawVolumeValues, labels: hasLabels() ? labelValues : null };
  }
```

- [ ] **Step 2: Use it in `setLabelAwareTransferFunction`** — replace

```js
    if (rawVolumeValues && volumeMetadata && mapper) {
      const values = hasLabels()
        ? bandLabeledHuValues(rawVolumeValues, labelValues, labelMetadata, volumeMetadata)
        : rawVolumeValues;
      buildImageData(volumeMetadata, values);
      mapper.setInputData(imageData);
      applyLabelInterpolation();
    }
```

with

```js
    if (rawVolumeValues && volumeMetadata && mapper) {
      const labels = hasLabels() ? labelValues : null;
      if (builtFrom.raw !== rawVolumeValues || builtFrom.labels !== labels) buildVolumeImage();
      applyLabelInterpolation();
    }
```

- [ ] **Step 3: Use it in `load()`** — replace

```js
      const values = hasLabels()
        ? bandLabeledHuValues(rawVolumeValues, labelValues, labelMetadata, volumeMetadata)
        : rawVolumeValues;
      buildImageData(loaded.metadata, values);
      mapper = vtk.Rendering.Core.vtkVolumeMapper.newInstance();
      mapper.setInputData(imageData);
```

with

```js
      mapper = vtk.Rendering.Core.vtkVolumeMapper.newInstance();
      buildVolumeImage();
```

In `destroyViewer()`, next to `imageData = undefined;` add `builtFrom = { raw: null, labels: null };`.

- [ ] **Step 4: Manual check (desktop)**

Run the server, load a labelled TotalSegmentator dataset (`ts_…`). Apply a text command such as "more bone". The render changes as before, and in DevTools → Performance a recording of the command shows no multi-hundred-ms `bandLabeledHuValues` call. Switching datasets still renders the new dataset correctly.

- [ ] **Step 5: Commit**

```bash
git add static/viewer.js
git commit -m "perf(viewer): stop rebanding and re-uploading the volume on every transfer-function change"
```

---

### Task 4: Crop and downsample (pure functions)

**Files:**
- Modify: `static/xr_logic.js` (add `bodyBoundingBox`, `cropVolume`, `strideDownsample`, export)
- Test: `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write the failing tests**

```js
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `xr.bodyBoundingBox is not a function`.

- [ ] **Step 3: Implement** in `static/xr_logic.js` (before `return {`):

```js
  // Inclusive voxel-index box of every value above `threshold` in a
  // Fortran-ordered (x fastest) volume, padded by `pad` voxels and clamped
  // to the volume. A volume with nothing above it keeps the full box.
  function bodyBoundingBox(values, dims, threshold, pad = 2) {
    const [nx, ny, nz] = dims;
    const min = [nx, ny, nz];
    const max = [-1, -1, -1];
    let index = 0;
    for (let k = 0; k < nz; k += 1) {
      for (let j = 0; j < ny; j += 1) {
        for (let i = 0; i < nx; i += 1, index += 1) {
          if (!(values[index] > threshold)) continue;
          if (i < min[0]) min[0] = i;
          if (i > max[0]) max[0] = i;
          if (j < min[1]) min[1] = j;
          if (j > max[1]) max[1] = j;
          if (k < min[2]) min[2] = k;
          if (k > max[2]) max[2] = k;
        }
      }
    }
    if (max[0] < 0) return { min: [0, 0, 0], max: [nx - 1, ny - 1, nz - 1] };
    return {
      min: min.map((value) => Math.max(0, value - pad)),
      max: max.map((value, axis) => Math.min(dims[axis] - 1, value + pad)),
    };
  }

  // Copies the inclusive `box` out of a Fortran-ordered volume.
  function cropVolume(values, dims, box) {
    const [nx, ny] = dims;
    const out = box.max.map((value, axis) => value - box.min[axis] + 1);
    const result = new values.constructor(out[0] * out[1] * out[2]);
    let dst = 0;
    for (let k = box.min[2]; k <= box.max[2]; k += 1) {
      for (let j = box.min[1]; j <= box.max[1]; j += 1) {
        const src = box.min[0] + nx * (j + ny * k);
        result.set(values.subarray(src, src + out[0]), dst);
        dst += out[0];
      }
    }
    return { values: result, dims: out };
  }

  // Nearest-neighbour: keeps every `factor`-th voxel per axis, so a label
  // band never blends into a neighbour's (averaging would invent values
  // between bands, i.e. a third class).
  function strideDownsample(values, dims, factor) {
    if (factor === 1) return { values, dims };
    const [nx, ny] = dims;
    const out = dims.map((size) => Math.ceil(size / factor));
    const result = new values.constructor(out[0] * out[1] * out[2]);
    let dst = 0;
    for (let k = 0; k < out[2]; k += 1) {
      for (let j = 0; j < out[1]; j += 1) {
        const row = nx * (j * factor + ny * k * factor);
        for (let i = 0; i < out[0]; i += 1) result[dst++] = values[row + i * factor];
      }
    }
    return { values: result, dims: out };
  }
```

Add `bodyBoundingBox, cropVolume, strideDownsample` to the exported object.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add static/xr_logic.js tests/js/xr_logic.test.js
git commit -m "feat(xr): add air-crop and stride-downsample helpers for headset volumes"
```

---

### Task 5: 16-bit conversion (pure function)

**Files:**
- Modify: `static/xr_logic.js` (add `toSixteenBit`, export)
- Test: `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write the failing tests**

```js
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `xr.toSixteenBit is not a function`.

- [ ] **Step 3: Implement** in `static/xr_logic.js` (before `return {`):

```js
  // The narrowest 16-bit integer array that holds the data, for vtk.js to
  // upload as a half-float (R16F) texture with preferSizeOverAccuracy. CT HU
  // and our label bands are integer-valued, so rounding loses nothing;
  // anything past the 16-bit range is clamped (typed arrays would wrap).
  function toSixteenBit(values) {
    let min = Infinity;
    for (let index = 0; index < values.length; index += 1) {
      if (values[index] < min) min = values[index];
    }
    const signed = min < 0;
    const [lo, hi] = signed ? [-32768, 32767] : [0, 65535];
    const out = signed ? new Int16Array(values.length) : new Uint16Array(values.length);
    for (let index = 0; index < values.length; index += 1) {
      out[index] = Math.min(hi, Math.max(lo, Math.round(values[index])));
    }
    return out;
  }
```

Add `toSixteenBit` to the exported object.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add static/xr_logic.js tests/js/xr_logic.test.js
git commit -m "feat(xr): add 16-bit volume conversion for half-float textures"
```

---

### Task 6: Wire `crop`, `down` and `tex` into the viewer

**Files:**
- Modify: `static/viewer.js` (`buildImageData`, `buildVolumeImage`, flat sample distance in `load()`, XR sample distance in `enterXR`)

- [ ] **Step 1: Let `buildImageData` take an origin** — replace

```js
  function buildImageData(metadata, values) {
    imageData = vtk.Common.DataModel.vtkImageData.newInstance();
    imageData.setDimensions(...metadata.dimensions);
    imageData.setSpacing(...metadata.spacing);
```

with

```js
  function buildImageData(metadata, values, origin = [0, 0, 0]) {
    imageData = vtk.Common.DataModel.vtkImageData.newInstance();
    imageData.setDimensions(...metadata.dimensions);
    imageData.setSpacing(...metadata.spacing);
    imageData.setOrigin(...origin);
```

- [ ] **Step 2: Apply the levers in `buildVolumeImage`** — replace the Task 3 body

```js
  function buildVolumeImage() {
    const values = hasLabels()
      ? bandLabeledHuValues(rawVolumeValues, labelValues, labelMetadata, volumeMetadata)
      : rawVolumeValues;
    buildImageData(volumeMetadata, values);
    mapper.setInputData(imageData);
    builtFrom = { raw: rawVolumeValues, labels: hasLabels() ? labelValues : null };
  }
```

with

```js
  // ?crop: anything at or below this HU is air outside the body.
  const AIR_HU = -500;

  function buildVolumeImage() {
    const logic = window.xrLogic;
    let values = hasLabels()
      ? bandLabeledHuValues(rawVolumeValues, labelValues, labelMetadata, volumeMetadata)
      : rawVolumeValues;
    let dims = volumeMetadata.dimensions;
    let origin = [0, 0, 0];
    if (PERF.crop) {
      // On raw HU, so a label band's offset can't make air look like tissue.
      const box = logic.bodyBoundingBox(rawVolumeValues, dims, AIR_HU);
      ({ values, dims } = logic.cropVolume(values, dims, box));
      // Shift the origin so world coordinates (camera, XR placement) match
      // the uncropped volume's.
      origin = box.min.map((index, axis) => index * volumeMetadata.spacing[axis]);
    }
    ({ values, dims } = logic.strideDownsample(values, dims, PERF.down));
    const spacing = volumeMetadata.spacing.map((step) => step * PERF.down);
    if (PERF.tex === "f16") values = logic.toSixteenBit(values);
    buildImageData({ dimensions: dims, spacing }, values, origin);
    mapper.setPreferSizeOverAccuracy(PERF.tex === "f16");
    mapper.setInputData(imageData);
    builtFrom = { raw: rawVolumeValues, labels: hasLabels() ? labelValues : null };
  }

  // Base XR ray-march step for the current image (finest spacing after any
  // downsampling), times the fixed ?xrsample multiplier.
  function xrBaseSampleDistance() {
    return window.xrLogic.xrSampleDistance(imageData.getSpacing()) * XR_SAMPLE_DISTANCE_SCALE;
  }
```

`XR_SAMPLE_DISTANCE_SCALE` is declared further down in the file. That is fine: it is only read when `xrBaseSampleDistance()` runs, long after the module has finished evaluating.

- [ ] **Step 3: Sample distances follow the reduced image** — in `load()` replace

```js
      mapper.setSampleDistance(Math.min(...loaded.metadata.spacing) / 2.0);
```

with

```js
      mapper.setSampleDistance(Math.min(...imageData.getSpacing()) / 2.0);
```

In `enterXR` replace

```js
    mapper.setSampleDistance(window.xrLogic.xrSampleDistance(volumeMetadata.spacing) * XR_SAMPLE_DISTANCE_SCALE);
```

with

```js
    mapper.setSampleDistance(xrBaseSampleDistance());
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Manual check (desktop)** — run the server and load the same labelled dataset under each URL; compare against no flags:
  - `/` → identical to before; `getXRDiagnostics()` shows `texture: "R32F"` and the full dims.
  - `/?crop=1` → same image, same camera framing within a few percent; dims smaller in x/y.
  - `/?down=2` → slightly softer image, dims halved (rounded up).
  - `/?tex=f16` → `texture: "R16F"`; look for banding in soft tissue against `/`.
  - `/?crop=1&down=2&tex=f16` → all three combined, no console errors, commands still change the render.

- [ ] **Step 6: Commit**

```bash
git add static/viewer.js
git commit -m "feat(viewer): apply crop, downsample and half-float texture levers"
```

---

### Task 7: Fixed foveation

**Files:**
- Modify: `static/xr_logic.js` (add `applyFixedFoveation`; give `withXRLayerContext` an `onLayer` callback; export)
- Modify: `static/viewer.js` (`enterXR`, `exitXR`)
- Test: `tests/js/xr_logic.test.js`

Why `onLayer`: vtk.js installs its layer with `session.updateRenderState(...)`, and WebXR applies render-state updates on the *next* frame, so `session.renderState.baseLayer` can still be null right after `enterXR`. The layer object is captured where `withXRLayerContext` constructs it; setting `fixedFoveation` on it takes effect once the layer is active.

- [ ] **Step 1: Write the failing tests**

```js
test("applyFixedFoveation sets the level where the layer supports it", () => {
  const layer = { fixedFoveation: 0 };
  assert.equal(xr.applyFixedFoveation(layer, 0.5), 0.5);
  assert.equal(layer.fixedFoveation, 0.5);
});

test("applyFixedFoveation reports unsupported layers instead of throwing", () => {
  assert.equal(xr.applyFixedFoveation({}, 1), "unsupported");
  assert.equal(xr.applyFixedFoveation({ fixedFoveation: null }, 1), "unsupported");
  assert.equal(xr.applyFixedFoveation(undefined, 1), "unsupported");
});

test("withXRLayerContext reports the layer it constructs", async () => {
  const { root, NativeLayer } = fakeRoot();
  let seen = null;
  const layer = await xr.withXRLayerContext(root, {}, async () =>
    new root.XRWebGLLayer("session", {}, {}), {}, (created) => { seen = created; });
  assert.ok(seen instanceof NativeLayer);
  assert.equal(seen, layer);
});
```

(`fakeRoot()` already exists in the test file, above the existing `withXRLayerContext` tests; put the new test after those.)

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `xr.applyFixedFoveation is not a function`, and the `onLayer` test fails with `seen` still null.

- [ ] **Step 3: Implement** — in `static/xr_logic.js` replace

```js
  async function withXRLayerContext(root, context, task, initOverrides = {}) {
    const NativeLayer = root.XRWebGLLayer;
    root.XRWebGLLayer = function XRWebGLLayer(session, _proxied, init) {
      return new NativeLayer(session, context, { ...init, ...initOverrides });
    };
```

with

```js
  async function withXRLayerContext(root, context, task, initOverrides = {}, onLayer = () => {}) {
    const NativeLayer = root.XRWebGLLayer;
    root.XRWebGLLayer = function XRWebGLLayer(session, _proxied, init) {
      const layer = new NativeLayer(session, context, { ...init, ...initOverrides });
      onLayer(layer);
      return layer;
    };
```

and add to the comment above it: `` `onLayer` receives the constructed layer (e.g. to set fixedFoveation). ``

Then, before `return {`, add:

```js
  // WebXR fixed foveation (lower resolution toward the lens edges), where
  // the browser exposes it; the spec makes the attribute null when it isn't
  // supported. Returns what the HUD should report.
  function applyFixedFoveation(layer, level) {
    if (!layer || !("fixedFoveation" in layer) || layer.fixedFoveation === null) return "unsupported";
    layer.fixedFoveation = level;
    return layer.fixedFoveation;
  }
```

Add `applyFixedFoveation` to the exported object.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all PASS, including the existing `withXRLayerContext` tests.

- [ ] **Step 5: Apply it on entering XR** — in `static/viewer.js` `enterXR`, replace

```js
      await window.xrLogic.withXRLayerContext(window, rawContext,
        () => openGLRenderWindow.enterXR(session),
        { framebufferScaleFactor: XR_FRAMEBUFFER_SCALE });
```

with

```js
      await window.xrLogic.withXRLayerContext(window, rawContext,
        () => openGLRenderWindow.enterXR(session),
        { framebufferScaleFactor: XR_FRAMEBUFFER_SCALE },
        (layer) => { xrFoveation = window.xrLogic.applyFixedFoveation(layer, PERF.fov); });
```

In `exitXR`'s `finally` block, next to `xrSaved = null;` add `xrFoveation = null;`.

- [ ] **Step 6: Commit**

```bash
git add static/xr_logic.js static/viewer.js tests/js/xr_logic.test.js
git commit -m "feat(xr): apply WebXR fixed foveation from the fov flag"
```

---

### Task 8: Adaptive sample-distance controller (pure function)

**Files:**
- Modify: `static/xr_logic.js` (add `ADAPT_DEFAULTS`, `ADAPT_INITIAL`, `adaptSampleDistance`, export)
- Test: `tests/js/xr_logic.test.js`

- [ ] **Step 1: Write the failing tests**

```js
const MISS = 20; // ms, well over 1000/72 * 1.15
const OK = 11;   // ms, a 90 Hz frame
const runAdapt = (state, frames) => frames.reduce((s, ms) => xr.adaptSampleDistance(s, ms), state);

test("adaptSampleDistance coarsens after three consecutive missed frames", () => {
  assert.equal(runAdapt(xr.ADAPT_INITIAL, [MISS, MISS]).multiplier, 1);
  assert.equal(runAdapt(xr.ADAPT_INITIAL, [MISS, MISS, MISS]).multiplier, 1.25);
});

test("adaptSampleDistance only counts consecutive misses", () => {
  assert.equal(runAdapt(xr.ADAPT_INITIAL, [MISS, MISS, OK, MISS]).multiplier, 1);
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

test("adaptSampleDistance never refines below minMul", () => {
  assert.equal(runAdapt(xr.ADAPT_INITIAL, Array(90).fill(OK)).multiplier, 1);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/js/xr_logic.test.js`
Expected: FAIL, `Cannot read properties of undefined` (no `ADAPT_INITIAL`) or `adaptSampleDistance is not a function`.

- [ ] **Step 3: Implement** in `static/xr_logic.js` (before `return {`):

```js
  // XR sample-distance controller tuning: hold 72 fps (the stage target),
  // coarsen fast after a few missed frames, refine slowly after ~1 s of
  // on-time ones, so quality doesn't visibly pump.
  const ADAPT_DEFAULTS = Object.freeze({
    minMul: 1, maxMul: 4, targetMs: 1000 / 72, missFactor: 1.15,
    coarsenAfter: 3, refineAfter: 90, coarsenStep: 1.25, refineStep: 1.1,
  });
  const ADAPT_INITIAL = Object.freeze({ multiplier: 1, missStreak: 0, okStreak: 0 });

  // One controller step for one frame's duration. Pure; the caller applies
  // `multiplier` to the base sample distance when it changes.
  function adaptSampleDistance(state, frameMs, options = {}) {
    const o = { ...ADAPT_DEFAULTS, ...options };
    if (frameMs > o.targetMs * o.missFactor) {
      const missStreak = state.missStreak + 1;
      if (missStreak < o.coarsenAfter) return { ...state, missStreak, okStreak: 0 };
      return { multiplier: Math.min(o.maxMul, state.multiplier * o.coarsenStep), missStreak: 0, okStreak: 0 };
    }
    const okStreak = state.okStreak + 1;
    if (okStreak < o.refineAfter) return { ...state, okStreak, missStreak: 0 };
    return { multiplier: Math.max(o.minMul, state.multiplier / o.refineStep), missStreak: 0, okStreak: 0 };
  }
```

Add `ADAPT_DEFAULTS, ADAPT_INITIAL, adaptSampleDistance` to the exported object.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/js/xr_logic.test.js`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add static/xr_logic.js tests/js/xr_logic.test.js
git commit -m "feat(xr): add adaptive sample-distance controller step"
```

---

### Task 9: Run the controller in the XR frame loop

**Files:**
- Modify: `static/viewer.js` (`setXRSampleMultiplier`, export)
- Modify: `static/xr.js` (`adapt` state, `onXRFrame`, `startSession`, `onSessionEnd`)

- [ ] **Step 1: Viewer hook** — in `static/viewer.js`, after `function xrBaseSampleDistance() { ... }` (added in Task 6) add:

```js
  // Called by xr.js's adaptive controller (?adapt=1) when its multiplier moves.
  function setXRSampleMultiplier(multiplier) {
    if (inXR()) mapper.setSampleDistance(xrBaseSampleDistance() * multiplier);
  }
```

and in `window.volumeViewer = { ... }`, after `perfFlags: PERF,` add `setXRSampleMultiplier,`.

- [ ] **Step 2: Controller in `static/xr.js`** — next to `let frameMs = null;` add:

```js
  let adapt = null; // controller state when ?adapt=1, else null
  let lastFrameTime = null;
```

In `onXRFrame`, directly after `session.requestAnimationFrame(onXRFrame);` add:

```js
    if (adapt && lastFrameTime !== null) {
      const next = logic.adaptSampleDistance(adapt, time - lastFrameTime);
      if (next.multiplier !== adapt.multiplier) window.volumeViewer.setXRSampleMultiplier(next.multiplier);
      adapt = next;
    }
    lastFrameTime = time;
```

In `startSession()`, next to `frameMs = null;` add:

```js
    adapt = window.volumeViewer.perfFlags.adapt ? logic.ADAPT_INITIAL : null;
    lastFrameTime = null;
```

In `onSessionEnd()`, next to `mapper = null;` add `adapt = null;`.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_xr_logic.py -q`
Expected: `1 passed`.

- [ ] **Step 4: Commit**

```bash
git add static/viewer.js static/xr.js
git commit -m "feat(xr): drive the XR sample distance from the adaptive controller"
```

---

### Task 10: Lab measurement matrix

**Files:**
- Create: `docs/experiments/2026-10-15-vr-perf-matrix.md`

- [ ] **Step 1: Write the matrix**

```markdown
# VR performance matrix — Vive Focus 3, 2026-10-15

Spec: `docs/superpowers/specs/2026-10-15-vr-performance-levers-design.md`.
Target: ≥ 72 fps sustained at `xrscale=1.0`.

**Fixed for every row:** dataset `<fill in: one labelled ts_… volume>`, same
start transfer function per TF column, same viewing position (stand still,
look at the volume centre for 10 s, then turn your head slowly for 10 s).

- **Dense TF:** the app's default start (soft tissue visible).
- **Sparse TF:** after the command "show only the skeleton".

Record fps and frame ms from the HUD's first debug line, GPU % from logcat
(`adb logcat | grep VRPerfMgr`), and sharpness/banding by eye (1 = unusable,
5 = desktop-like). Photograph the HUD for each row.

| # | URL flags | xrscale | Dense fps / GPU % | Sparse fps / GPU % | Sharpness | Banding | Notes |
|---|---|---|---|---|---|---|---|
| 1 | (none, baseline) | 0.5 | | | | | |
| 2 | (none, baseline) | 1.0 | | | | | |
| 3 | `tex=f16` | 1.0 | | | | | |
| 4 | `crop=1` | 1.0 | | | | | |
| 5 | `down=2` | 1.0 | | | | | |
| 6 | `fov=1` | 1.0 | | | | | |
| 7 | `tex=f16&crop=1` | 1.0 | | | | | |
| 8 | `tex=f16&crop=1&fov=1` | 1.0 | | | | | |
| 9 | `tex=f16&crop=1&fov=1&down=2` | 1.0 | | | | | |
| 10 | best of 8/9 + `adapt=1` | 1.0 | | | | | |
| 11 | best of 8/9 + `adapt=1` | 0.75 | | | | | |

Full URL per row: `https://<lan-host>:8000/?xrscale=<xrscale>&<flags>`.

## Verdict

- Best configuration: …
- Meets ≥ 72 fps at 1.0 on the dense TF? yes / no
- If no → Stage B (custom ray marcher). The rows above are the evidence.
```

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/2026-10-15-vr-perf-matrix.md
git commit -m "docs(experiments): VR performance lab matrix for Stage C levers"
```

---

### Task 11: Full verification

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: everything passes except the two known, pre-existing failures in `tests/test_pretrain.py` (`test_collect_is_deterministic_and_independent_of_workers`, `test_collect_search_is_deterministic_and_independent_of_workers`), which fail on `main` before this work too.

- [ ] **Step 2: Desktop regression** — server running, `/` with no flags: load a labelled and an unlabelled dataset, apply two commands each, undo/redo, switch datasets. Rendering and camera look as before; no console errors.

- [ ] **Step 3: Headset smoke test** — on the Vive Focus 3 open `/?tex=f16&crop=1&fov=1&adapt=1&xrscale=1`, enter VR: the HUD shows both debug lines with `tex=R16F`, reduced dims, a numeric or `unsupported` fov, and `[xr-perf]` lines appear in remote DevTools once per second. Push-to-talk and A/B undo/redo still work.
