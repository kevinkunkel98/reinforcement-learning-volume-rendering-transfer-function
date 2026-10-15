# VR lab session — 2026-10-15 (handoff)

Ohio lab, HTC Vive Focus 3 (Snapdragon XR2 / Adreno 650) over USB-C.
Branch **`vr-perf-levers`** (not merged, not pushed), 24 commits on top of `main`.

## Result in one line

From "1/20 resolution or ~2 fps at full resolution, one eye missing" to a
sharp, lit, correct stereo view at **27–33 fps** (`?renderer=custom`,
`down=2`, `xrscale=0.8`) with a crisp HUD and thumbstick rotation. The user's
verdict: "holy smokes this all looks so good now".

## What was achieved

| Problem at the start | Cause | Fix (commit) |
|---|---|---|
| "Abysmal" resolution in VR | vtk.js auto-adjust (from `7786d80`) measures its own interactor loop, which doesn't run in XR (reads ~2 fps) → always renders at 1/20 per axis | auto-adjust off in VR (`3f6737b`) |
| Volume missing in one eye, black crop border | vtk's offscreen "small viewport" path sizes itself from the canvas, not the XR eye viewports | same fix |
| Black straight-edged wedges after turning | vtk.js 25 clips rays against the unrotated box when the volume prop is rotated | turn the room frame instead (`3f6737b`); custom renderer applies rotation directly |
| Stutter on every voice command | every TF change re-banded 36M voxels and re-uploaded the 3D texture | build once per volume (`4084499`): **1170 → 190 ms** per command |
| Volume tilted/off to the side | flat camera view-up nearly parallel to its view direction | orthogonalize before entering VR (`b59c00d`) |
| Flat, "colorful" look vs. the web app | shading switched off in VR (`7786d80`) | lighting back on (`c74f5e9`), ~20 % fps |
| No way to turn the volume | — | thumbstick yaw/pitch, click resets (`9081ff8`; grip-grab tried and dropped as messy) |
| Blocky organs, stripes | single banded scalar forces NEAREST; fixed-step slicing | custom renderer: separate HU (R16F, linear) + label (R8, nearest) textures, jitter |
| Blurry, shimmering HUD text | 1024×420 canvas, no mipmaps | 2× canvas, mipmaps + anisotropic filtering (`ce4c5a3`) |
| Plain URL on the headset got slow settings | — | headset defaults `f16`, `down=2`, `xrscale=0.8`, lit (`c74f5e9`) |

New code: `static/vr_renderer.js` (WebGL2 ray marcher + own WebXR loop),
`static/tf_table.js` (TF table shared by flat and VR), `static/brick_grid.js`
(empty-space-skipping bricks), `static/vr_math.js`, URL performance flags in
`static/xr_logic.js`, measurement HUD. 68 JS tests (`tests/js/`, run by
`pytest tests/test_xr_logic.py`). Tools: `tools/headset/` (see its README).

Specs/plans: `docs/superpowers/specs/2026-10-15-vr-performance-levers-design.md`,
`docs/superpowers/specs/2026-10-15-vr-raymarcher-design.md`, plans of the same names in
`docs/superpowers/plans/`.

## Measurements (headset, `ts_s0477`, 243×243×404 @ 1.5 mm)

Flat-renderer benchmark, volume filling the view, ms per frame (lower is better):

| Lever | Effect |
|---|---|
| 4× pixels (1152×576 → 2304×1152) | +10–40 % (vtk) |
| `down=2` (8× fewer voxels) | 27 → 8 ms (3.4×) |
| `tex=f16` | ~5 % |
| `crop=1` | 0 % (TotalSegmentator volumes are already cropped) |
| lighting / empty-space skipping (custom) | ~0 % with the default TF (72–82 % of bricks occupied) |
| volume out of view | 2 ms |
| foveation | not supported by Wolvic |

**Key insight:** frame cost tracks **how many distinct voxels rays read**, not
pixels, bytes or shader math. Resolution is cheap; voxel count is expensive.

In VR: vtk.js at full resolution ~2 fps; vtk `f16+down=2+0.8×` 25–40 fps
(25 lit); custom renderer `down=2+0.8×` lit **27–33 fps**, visibly cleaner.
Numbers drift with headset temperature (same config 36 → 50 ms after a run).

## How to resume

```bash
git switch vr-perf-levers
.venv/bin/python server.py --host 0.0.0.0 --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem
# headset: https://<mac-lan-ip>:8000/?renderer=custom
```

URL flags: `renderer=custom|vtk`, `down=1|2`, `xrscale=0.1–2`, `tex=f32|f16`
(vtk only), `xrsample=`, `adapt=1`, `crop=1`. Headset debugging: `tools/headset/README.md`.

## Open — VR

1. **Make the custom renderer the headset default** (plan Task 9): update
   `HEADSET_PERF_DEFAULTS`, write `docs/experiments/2026-10-15-vr-raymarcher-results.md`.
2. **Try `down=1` with a larger step** on the custom renderer (smooth HU should
   beat vtk's `down=2` at similar fps): `vrRenderer.benchmark({down: 1, stepScale: 2})`
   gave ~22 ms vs ~14 ms for `down=2`.
3. **72 fps not reached** (27–33). Next ideas, all about touching fewer voxels:
   occupancy fetch only on brick change; adaptive step (`adapt=1`) with the
   custom renderer; a lower-resolution "proxy" volume while the head moves fast.
4. **Recenter button** (bring the volume in front of you); today only the
   orientation resets.
5. Merge `vr-perf-levers` into `main` once happy (nothing pushed yet).
6. Flat page on the headset still waits on `navigator.xr.isSessionSupported`
   before rendering; hangs while the headset sleeps (consider a timeout).

## Open — RL (for working at home)

From `docs/STATUS.md` (TODO list) and this morning's review of it:

1. **Diagnose the `relative` regression** (40 % of instructions, negative in
   every 8-class experiment). Clue found today: E5 made it worse (−0.068 →
   **−0.145**, one seed −0.383) although the show_only reweighting doesn't touch
   the `relative` metric — points at training dynamics/budget, not the metric.
2. **Held-out test-split evaluation** of E3/E5 (only validation numbers exist).
3. **Hybrid routing**: `show_only`/`absolute` → `hill_climb`, `relative` → rule executor.
4. **Fix STATUS.md's numbers** before they reach the thesis:
   - "+0.153 = median across kinds" is wrong: it's the episode-level median per
     seed (0.179/0.143/0.153), median of 3 seeds. Median across kinds would be ~+0.09.
   - The "Current (E3/E5)" table mixes sources: brightness +0.246 is seed 0 only
     (3-seed median **+0.219**); compound +0.088 → **+0.080**; show_only "+0.096
     (E3)" is an old-metric number (correct E3: −0.238). Pick one checkpoint as
     "current" and take every number from it.
   - Speaker notes: the validation split is used for checkpoint selection, not
     seen in training (selection bias, not leakage); "metric is differentiable"
     is unsupported; SB3 SAC default is exactly `[256, 256]`.
5. **README headline** still shows the v4 4-class numbers as current.
6. **Two failing tests on `main` and this branch**: `tests/test_pretrain.py`
   `test_collect_is_deterministic_and_independent_of_workers` and
   `test_collect_search_is_deterministic_and_independent_of_workers`
   (determinism of pretraining collection) — not looked into.
7. RLHF data collection (multi-rater, Krippendorff's alpha) is location-bound —
   only possible while people are in the lab.
8. Cluster access (since 2026-10-03) isn't mentioned in STATUS.md's compute
   plan; it's likely the better option for 3-seed runs.
