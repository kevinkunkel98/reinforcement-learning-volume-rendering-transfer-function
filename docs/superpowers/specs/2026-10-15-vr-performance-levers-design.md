# VR performance levers (Stage C) — design

**Date:** 2026-10-15 · **Hardware:** HTC Vive Focus 3 (Snapdragon XR2 / Adreno 650), in the Ohio lab

## Goal

Smooth, sharp volume rendering on the headset — both on the flat page in the
headset browser and in immersive VR. **Target: ≥ 72 fps sustained at
`xrscale=1.0`** (native per-eye resolution), 90 fps where reachable.

Today: 0.5× runs at 88–90 fps but looks pixelated; 0.75× drops to 6–7 fps;
1.0× with 2× coarse sampling runs at 64–69 fps but blurry. The flat page in
the headset browser already stutters before entering VR, and head motion in
VR lags.

## Diagnosis

`static/viewer.js` uploads the full-resolution CT (~36 M voxels) as a
`Float32Array`. In vtk.js 25.11 under WebGL2 that becomes an `R32F` 3D
texture: 4 bytes per sample, ~144 MB, and on mobile often without native
linear filtering. The sharp cliff between 0.5× and 0.75× fits a memory- or
bandwidth-bound workload, not a smooth compute limit. Nothing shrinks the
work per ray either: no air cropping, no downsampling, no foveation, and
vtk.js's `AutoAdjustSampleDistances` does not engage inside the XR frame loop.

A second, separate stutter source: every transfer-function change reaches
`setLabelAwareTransferFunction` through `load()`, which re-bands all voxels
and re-uploads the whole 3D texture (`viewer.js`, on a load it even happens
twice). A transfer-function change needs neither.

## Strategy

Staged (option C): try the cheap levers inside vtk.js first, measure each on
the headset, and move to a custom WebGL2 ray marcher (Stage B) only if they
can't reach the target.

## Design

### 1. Every lever is a URL flag; the HUD shows the active config

| Flag | Lever | Values | Default (this stage) |
|---|---|---|---|
| `?tex=` | scalar texture format | `f32` · `f16` | `f32` |
| `?crop=` | crop air to the body's bounding box | `0` · `1` | `0` |
| `?down=` | nearest-neighbour stride downsample | `1` · `2` | `1` |
| `?fov=` | WebXR fixed foveation level | `0`–`1` | `0` |
| `?xrscale=` | per-eye framebuffer scale (exists) | e.g. `0.5`–`1.0` | `0.5` |
| `?adapt=` | adaptive sample-distance controller | `0` · `1` | `0` |

Defaults reproduce today's behaviour exactly, so the baseline row of the lab
matrix needs no special build. After the lab session, the winning combination
becomes the default on VR-capable devices (a follow-up change, not part of
this design). Desktop behaviour stays unchanged regardless of flags, except
`tex`, `crop` and `down` also apply on desktop when explicitly set, for
side-by-side quality checks.

Flag parsing is a pure function `parsePerfFlags(search)` in
`static/xr_logic.js`: unknown or out-of-range values fall back to the default,
never throw.

The HUD's existing debug line (`static/xr.js`, `drawHud`) is extended with:
texture format actually in use, voxel dimensions after crop/downsample,
foveation level (or `unsupported`), framebuffer scale, sample distance, mean
frame time in ms, and fps. A photo of the HUD is then a complete measurement
record. The same diagnostics are logged to the console once per second, so
remote DevTools/logcat captures them too.

### 2. Texture format (`tex`)

- `f32`: today's path, unchanged.
- `f16`: the banded values (range 0 to 8·4096 + 3050 = 35 818) are stored as a
  `Uint16Array` instead of `Float32Array`, and the mapper gets
  `setPreferSizeOverAccuracy(true)`, which makes vtk.js 25.11 upload a
  `HALF_FLOAT` 3D texture (2 bytes per voxel). The cost is precision of
  roughly 18–32 HU per step at the top of the range. Whether that bands
  visibly against peaks 45–280 HU wide is judged by eye in the lab.
  Unlabelled volumes (no banding) use the same path with HU offset by
  `-CENTER_RANGE[0]` so values stay non-negative.

vtk.js 25.11 doesn't expose its scalar texture, so the HUD derives the format
from the same rule vtk.js applies under WebGL2 (`Float32` → `R32F`; 16-bit
integers with `preferSizeOverAccuracy` → `R16F`), and also reports whether the
device has `OES_texture_float_linear` (`flin=`).

### 2b. Build the volume texture once per volume, not per TF change

`viewer.js` remembers which raw-value and label arrays the current image data
was built from and skips rebanding and re-uploading when they haven't
changed. A transfer-function change then only updates the transfer function.

### 3. Crop (`crop`) and downsample (`down`)

Pure functions in `static/xr_logic.js`, unit-tested in
`tests/js/xr_logic.test.js` (run from pytest through `tests/test_xr_logic.py`):

- `bodyBoundingBox(values, dims, threshold)` → inclusive voxel index box of all
  voxels whose raw HU is above `threshold` (−500 HU, i.e. not air), padded by
  2 voxels and clamped to the volume. An all-air volume returns the full box.
- `cropVolume(values, dims, box)` → `{values, dims}` copied out of the box.
- `strideDownsample(values, dims, factor)` → nearest-neighbour stride
  sampling, so label bands never mix (averaging would invent values between
  bands).

`viewer.js` applies crop and then downsample to the *banded* values before
`buildImageData`. The crop is computed on raw HU so banding cannot affect it.
The image data's origin is shifted by `box.min × spacing` and its spacing is
multiplied by the downsample factor, so world-space bounds, camera and the
HUD/volume placement stay consistent. The reduced volume serves both the flat
page and XR on that device; no second texture is uploaded.

Because the shared `CENTER_RANGE`/`LABEL_BAND` banding is unchanged, the
transfer-function code (`setLabelAwareTransferFunction`) needs no changes.

### 4. Fixed foveation (`fov`)

After `openGLRenderWindow.enterXR(session)`, if
`session.renderState.baseLayer` has a `fixedFoveation` property, set it to the
flag value. Otherwise the HUD shows `fov=unsupported`. No polyfill.

### 5. Adaptive sample-distance controller (`adapt`)

A pure step function in `static/xr_logic.js`:

```
adaptSampleDistance(state, frameMs, { base, minMul: 1, maxMul: 4,
                                      targetMs, coarsenAfter, refineAfter })
  → { multiplier, missStreak, okStreak }
```

- `targetMs` = 1000 / 72 (the stage target).
- A frame counts as a miss if `frameMs > targetMs × 1.15`.
- After `coarsenAfter` = 3 consecutive misses: multiplier × 1.25 (clamped to
  `maxMul`), streaks reset.
- After `refineAfter` = 90 consecutive on-time frames (about 1 s): multiplier
  ÷ 1.1 (clamped to `minMul`), streaks reset.
- Coarsening is fast and refining is slow, which avoids visible oscillation.

`xr.js`'s `onXRFrame` feeds it the delta between successive XR frame
timestamps and, when the multiplier changes, calls a new
`volumeViewer.setXRSampleMultiplier(m)` that sets
`base × m` on the mapper. It only adjusts sample distance: changing the
framebuffer scale mid-session would need a new `XRWebGLLayer`, out of scope.
With `adapt=0` the existing fixed `?xrsample` multiplier applies as today.

### 6. Lab protocol

`docs/experiments/2026-10-15-vr-perf-matrix.md` holds a pre-filled table:
one row per configuration (baseline; each lever alone; cumulative
combinations), each at `xrscale` 0.5 / 0.75 / 1.0, with the same dataset and
transfer function every time (one dense TF: soft tissue visible; one sparse
TF: show only skeleton). Columns: fps (HUD), GPU % (logcat `VRPerfMgr`),
sharpness/banding by eye, notes.

### 7. Stop/go for Stage B

If the best combination cannot hold **≥ 72 fps at `xrscale=1.0`** with an
acceptable image on the dense TF, the matrix is the evidence for Stage B: a
custom WebGL2 ray marcher with empty-space skipping, early ray termination and
16-bit integer textures, built for this data.

## Out of scope

- Empty-space skipping (needs shader work; Stage B).
- `uint8` with per-class windowing (too few levels per band).
- Changing defaults on VR-capable devices (done after the lab measurements).
- Interaction: grab/rotate/scale, environment, comfort features.
- Server-side rendering/streaming.

## Testing

- Unit tests (node:test) for `parsePerfFlags`, `bodyBoundingBox`,
  `cropVolume`, `strideDownsample` and `adaptSampleDistance`, including edge
  cases: an all-air volume, odd dimensions with `down=2`, clamping at
  `minMul`/`maxMul`, and garbage flag values.
- Existing pytest + node suites stay green.
- Manual: the desktop flat viewer with no flags looks and behaves exactly as
  before; on the headset, the HUD shows the flags and the format actually in use.
