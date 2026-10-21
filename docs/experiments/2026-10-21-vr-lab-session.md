# VR lab session 3 — 2026-10-21

Vive Focus 3 over USB, custom ray marcher, `ct_chest` then `ts_s0358`.
Goal: no stutter when the head moves, and a sharper image. Measured from the
`[xr-perf]` logcat line, which now also carries the worst frame interval and
the number of late frames (> 1.5 refreshes at 90 Hz) of each second.

## What changed

| Problem | Cause | Fix |
|---|---|---|
| Stutter every second, even head-on (worst frame 44–112 ms in seconds with 1–2 late frames) | the fps tick redrew the 2048×1520 VR panel and re-uploaded it with mipmaps: 30–35 ms CPU | panel redraws only on content changes; live numbers stay in logcat (`xr.js`) |
| `adapt=1` only ever coarsened (3.0 → 5.86 mm, never back) | each isolated skipped refresh reset the on-time streak | a lone miss costs 30 on-time frames (`missPenalty`) |
| `adapt=1` did not react at 40–48 fps | every other frame on time, so never 3 misses in a row | an on-time frame shrinks the miss streak by a quarter instead of clearing it |
| White dots flickering when the body turns | specular highlights on gradients that are CT noise inside homogeneous tissue | lighting fades in over a gradient of 10–60 HU/mm (`state.surface`), also made frames clearly cheaper |
| Grain from per-pixel ray offsets | white-noise `sin` hash | interleaved gradient noise |
| Turn boost left on for the session after one stick turn | XR frame `time` and `performance.now()` on different bases in Wolvic | both sides use `performance.now()` |
| Turn boost (3× step) on top of the adaptive step reached 36 mm | two independent multipliers | one load, split by `xrLogic.splitLoad`: resolution 1.0 → 0.8 first, then step, capped at 2× |
| UI blurry | `xrscale=0.8` | headset default `xrscale=1`, `adapt` on; volume renders at 0.8–1.0 of that into an offscreen target and is upscaled, panel always full resolution |
| Ray marcher ran every pixel of both eyes | full-screen triangle | volume and upscale passes scissored to the box's projected rectangle (`vrMath.screenRect`) — **not yet measured** |

Headset defaults are now `renderer=custom down=2 xrscale=1 tex=f16 fovstep=2 adapt=1`.

## Measurements

Flat benchmark (`vrRenderer.benchmark`, 1152×576 scaled, `down=2 fovstep=2`, after the surface-weighted lighting):

| scale | step 1× | step 2× |
|---|---|---|
| 1.0 | 8.8 ms | 5.3–5.6 ms |
| 0.8 | 6.3–6.5 ms | 3.8–3.9 ms |
| 0.6 | 6.4–7.5 ms | 3.9–4.0 ms |

Below 0.8 the resolution buys nothing (texture cache), hence `SPLIT_RES_MIN = 0.8`.
Benchmarks drift ~1.5× within a sitting (13 vs 20 ms for the same config): interleave and repeat.

In VR, after the panel fix: quiet seconds at 90 fps, worst frame 12 ms; heavy
views (leaning in, stick turning) still 40–70 fps for 2–4 s. On `ts_s0358` at
`xrscale=1` the controller held `res=0.80` and a 1.3–1.7× step even head-on.

## Negative result: pre-classified volume

Baking colour (TF applied per voxel, RGBA8 3D texture, rebuilt per TF change
on the GPU) to replace the HU + label + TF fetches: **2× slower** (25 vs 13 ms,
interleaved). A baked RGBA8 normal volume made lighting 4.3 → 15.7 ms. The
Adreno 650 is bandwidth-bound on 3D textures; 4-byte voxels cost more than
the R16F + R8 pair plus a cached 2D table. Reverted.

## TODO

1. **Measure the scissor change** in VR: head-on and leaning in; does
   `res` reach 1.00 and `sd` ~3.0 head-on on `ts_s0358`?
2. **Staircase organ edges**: the label volume is nearest-sampled at
   `down=2`. Try labels at full resolution (R8, 256×256×524 ≈ 34 MB) with HU
   at `down=2`; measure the cost of the bigger label fetch.
3. **Glittery bone**: cortex thinner than a 4–5 mm step. Options: finer
   step only where bone opacity is high, or refine the step after a
   transparent→opaque transition (one step back, half step).
4. **Controller pumping** between `res` 0.8/1.0 under changing load: check
   whether the change is visible; if so, slow the refine or add hysteresis.
5. Keep the headset defaults and the SPLIT constants in sync with the
   README's VR paragraph and `tools/headset/README.md` once 1–4 settle.
