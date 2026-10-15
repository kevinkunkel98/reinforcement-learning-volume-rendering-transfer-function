# Custom VR ray marcher (Stage B) — design

**Date:** 2026-10-15 · **Hardware:** HTC Vive Focus 3 (Adreno 650), VIVE Browser = Wolvic on Gecko 105
**Predecessor:** `2026-10-15-vr-performance-levers-design.md` (Stage C)

## Goal

Replace vtk.js's volume renderer **in VR only** with a purpose-built WebGL2
ray marcher, to reach **≥ 72 fps at `xrscale` ≥ 0.8 and full data resolution
(`down=1`)** on `ts_s0477` with the default transfer function, while looking
the same as the flat web app. The flat page stays on vtk.js, unchanged.

## Why (evidence from the Stage C lab session)

Measured on the headset (flat renderer, VR-equivalent pixel counts, volume
filling the view, `gl.readPixels` sync):

| Observation | Number |
|---|---|
| Native res, vtk.js, in VR | ~2 fps (575 ms) |
| 4× the pixels (1152×576 → 2304×1152) | +10–40 % time |
| 8× fewer voxels (`down=2`) | 27 → 8 ms (3.4×) |
| Half the bytes (`tex=f16`) | ~5 % |
| Volume out of view | 2 ms |
| Best Stage C config (`f16`, `down=2`, 0.8×, lit) | 25–40 fps |

Frame cost tracks **how many distinct voxels rays pull through the texture
cache**, not pixels or bytes. vtk.js 25 has no way to skip voxels that are
invisible under the current transfer function, and three of today's bugs
(stereo small-viewport, rotated-prop clipping, auto-adjust in XR) came from
its ray caster not being built for WebXR. Also, label-scoped shading forces
NEAREST interpolation of one banded scalar, which makes every voxel a hard
block.

## Scope

In: a VR-only renderer (volume + HUD panel), empty-space skipping, early ray
termination, separate HU/label textures, lighting matching the flat view,
thumbstick rotation, a headset benchmark entry point, a `?renderer=` switch.

Out: the flat/desktop viewer, server-side rendering, new interactions,
multi-volume, changing the transfer-function model or the RL pipeline.

## Architecture

```
xr.js (session, controllers, HUD canvas)  ──►  vr_renderer.js (WebGL2 + WebXR frame loop)
        │                                         ├─ volume pass (ray march shader)
        │                                         └─ HUD pass (textured quad)
viewer.js (data + TF state, flat vtk.js view) ──►  getVolumeData(), onTransferFunctionChange()
tf_table.js (pure) ── shared by viewer.js and vr_renderer.js
brick_grid.js (pure) ── brick stats + occupancy
```

### Units

| File | Responsibility | Tested by |
|---|---|---|
| `static/tf_table.js` (new, pure) | The label-scoped transfer-function table: `tfTable(params, layers, labelled)` → 9 rows × 256 RGBA (or 1 row unlabelled), exactly the samples `setTransferFunction` feeds vtk today. `viewer.js` switches to it, so flat and VR share one source. | `node:test`: matches the current viewer's sampling for fixed params/layers |
| `static/brick_grid.js` (new, pure) | `brickStats(hu, labels, dims, brickSize)` → per brick: label bitmask (9 bits) + HU min/max, taken over the brick **plus a one-voxel apron** (LINEAR HU filtering at a brick face mixes in the neighbour's voxels). `occupancy(stats, table)` → per brick 0/1: visible if any present label's row has max alpha > ε over [HU min, HU max] (prefix-max per row). | `node:test`: synthetic volumes; conservative (never marks a brick with a visible voxel empty) |
| `static/vr_renderer.js` (new) | WebGL2 resources, shaders, the XR frame loop, stereo views, placement/rotation, HUD quad, `benchmark()` | headset benchmark + screenshots (manual) |
| `static/xr.js` (changed) | Starts the session through `vr_renderer.js` when `renderer=custom`, otherwise the existing vtk path; controller/HUD logic unchanged | existing node tests + headset |
| `static/viewer.js` (changed) | Exposes raw HU, labels, dims, spacing, current params/layers; notifies on TF change; uses `tf_table.js` | existing tests + flat regression check |

### Data on the GPU

- **HU texture:** `R16F` 3D, HU clamped to `CENTER_RANGE` (−1050…2000, exact
  in half float), **LINEAR** filtering (core WebGL2, no extension needed).
- **Label texture:** `R8` 3D, **NEAREST**. Sampling label and HU separately
  replaces the banded single scalar: HU interpolates smoothly inside an organ,
  the class stays crisp, and no interpolated value can land in a third class.
- **TF table:** `RGBA8` 2D, 256 × 9 (HU × label row), LINEAR along HU.
- **Occupancy:** `R8` 3D at brick resolution (8³ voxels per brick: 243×243×404
  → 31×31×51), NEAREST. Recomputed on the CPU on every TF/layer change
  (brick stats are computed once per volume).

### Ray march (fragment shader, per eye)

1. Ray from the eye through the pixel in volume texture space (model matrix
   = XR placement × thumbstick rotation; rotation is applied directly, so the
   rotated-prop clipping bug cannot recur).
2. Intersect with the volume box; start at entry + per-pixel jitter in
   [0, step) (removes the wood-grain stripes).
3. Loop: look up the occupancy brick. **Empty → jump to the brick's exit**
   (slab test against the brick's AABB, plus a small epsilon). Occupied →
   sample label (nearest) and HU (linear), look up RGBA from the TF table,
   apply opacity correction for the step length, light if alpha > 0.01,
   composite front to back.
4. **Early termination** at accumulated alpha ≥ 0.98.
5. Step = finest voxel spacing (as vtk in VR today); the Stage C adaptive
   controller (`?adapt=1`) can scale it.

### Look parity with the flat view

- Same TF table (shared `tf_table.js`).
- Opacity correction as vtk.js does it for the same unit distance.
- Lighting: headlight Phong, ambient 0.1, diffuse 0.7, specular 0.2,
  specular power as vtk's default; gradient by central differences on the HU
  texture, computed only for samples that contribute.
- Acceptance is by side-by-side screenshot: `vr_renderer.benchmark()` renders
  a fixed camera offscreen; the flat vtk view renders the same camera.

### Frame loop and HUD

`vr_renderer.js` owns `session.requestAnimationFrame`: per view it sets the
viewport from `baseLayer.getViewport(view)`, uses `view.projectionMatrix` and
`view.transform.inverse.matrix`, draws the volume, then the HUD quad (canvas
texture from `xr.js`, depth-tested against nothing, drawn last). Placement
reuses `xrLogic.xrPlacement`; the thumbstick rotation reuses
`xrLogic.createStickRotator`; the HUD stays fixed in the room.
`framebufferScaleFactor` comes from `PERF.xrscale` as today.

### Switch and rollout

`?renderer=custom|vtk`. During development the default stays `vtk`; once the
success criterion holds on the headset, headsets default to `custom`. The vtk
VR path stays available for A/B and as a fallback.

## Error handling

- WebGL2 or 3D texture creation fails, or the volume exceeds
  `MAX_3D_TEXTURE_SIZE` → notify on the page and fall back to the vtk VR path.
- Shader compile/link error → same fallback, with the info log in the console
  (visible through the remote debug socket).
- Session ends mid-frame → release GL resources on `end`, as `xr.js` already
  handles for the vtk path.
- No labels for a dataset → label texture all zeros, TF table with a single
  row (the unlabelled path, as the flat view does).

## Testing

- `node:test` (through `tests/test_xr_logic.py`, extended to run the new test
  files): `tf_table` parity, `brickStats`, `occupancy` (conservative, label
  bitmask respected, layer opacity 0 hides a class's bricks), slab-exit math in
  a JS reference function the shader mirrors.
- Headset benchmark (`vr_renderer.benchmark({w, h, step})`) against the Stage C
  numbers, same camera and pixel counts, plus in-VR fps through the existing
  frame probe.
- Screenshot parity flat vs custom; the regressions found today (both eyes
  render, no crop border, no wedges after rotation) checked explicitly.

## Success criteria

1. ≥ 72 fps in VR at `xrscale` ≥ 0.8, `down=1`, `ts_s0477`, default TF,
   volume filling most of the view.
2. Screenshot parity with the flat view (same organs, colours and lighting;
   no stripes or blocks beyond the data's own resolution).
3. Voice commands, undo/redo, thumbstick rotation and HUD work as today.

If criterion 1 fails, the benchmark shows which term dominates (occupied
bricks, lighting, step count), and the next lever is chosen from that, not
guessed.
