# 2026-10-19: VR lab session -- frame rate at full resolution

Second session with the Vive Focus 3 (Wolvic / Gecko 105, Adreno 650), after
[the first](2026-10-15-vr-lab-session.md). Goal: a faster frame rate
without lowering the volume or framebuffer resolution. Commits 283572d,
5945c63, 35625e7, 9c51ea7, c240e03, a345c58.

## Result in one line

Head-on at 0.9 m on a 523-slice body scan, **21-22 → 57-58 fps**; while the
thumbstick turns the volume, **84-88 fps** -- at the same resolution
(volume at half size, `down=2`, framebuffer scale 0.8).

## What changed

| Change | Why | Effect |
|---|---|---|
| Custom ray marcher is the headset default | `?renderer=custom` was easy to mistype (`costum` silently ran vtk.js at ~21 fps) | no flag needed; `?renderer=vtk` still works |
| Volume placed in front of the viewer; thumbstick click recenters it | it sat 0.9 m ahead of the `local` space's origin, which after a runtime restart was where the user's head was | volume always in view -- **earlier fps numbers of this session were taken with it misplaced** |
| Foveated steps (`fovstep=2`, default) | the lens blurs the view's edge anyway | ~20 % shorter frames (benchmark and VR); `fovstep=3` no faster |
| Double step after a transparent sample (`emptyStride=2`) | the faded shell leaves 82 % of bricks occupied, but much of each is transparent under the TF | −15 % on the benchmark, no visible change; **21-22 → 44-58 fps** in VR |
| Brick skipping only below 0.6 occupancy | at 0.82, checking bricks costs more than skipping saves | −8-10 % on the benchmark; **57-58 fps** steady |
| 3× step while the volume turns (250 ms hold) | turned end-on, rays cross much more tissue against the texture's cache layout | rotating **84-88 fps** (2× was "a bit smoother", 3× "even better", no visible cost) |
| Forward-difference gradients | 3 extra fetches per lit sample instead of 6 | −10 % on one volume |
| `Cache-Control: no-cache` on `/static` | the headset browser kept a stale `xr_logic.js` across reloads | code changes arrive on a normal reload |
| VR panel lists what a command moved | the chat's per-class rows (dot, class, before → after %, ▲/▼) were missing in VR | up to six rows, largest change first |

## Measurements

Offscreen benchmark on the headset GPU (`vrRenderer.benchmark`, 1152×576,
`down=2`, `fovstep=2`, two interleaved rounds), ms per frame:

| Variant | ms | |
|---|---|---|
| baseline | 26.0-26.8 | |
| `emptyStride=2` | 22.3-22.7 | kept |
| light only samples with weight > 0.02 | 23.8-24.0 | rejected: soft tissue goes flat |
| light only samples with weight > 0.005 | 25.3-26.0 | too small |
| ray termination at 0.95 | 25.7-26.3 | no gain |
| precomputed normals (RGBA8, half size) | equal at `down=2`, slower at `down=1` | reverted |
| growing stride over transparent runs (cap 3/4/6) | 21.3-22.3 vs 22.3 | no gain, reverted |
| occupancy fetch only on entering a new brick | 22.5 vs 22.3 | no gain, reverted |
| lighting off (reference) | 21.5-22.0 | |

In VR (`[xr-perf]` from logcat, same TF, no commands in between):

| Configuration | Head-on | Rotating |
|---|---|---|
| custom, `fovstep=1`, volume misplaced | 44-52 | |
| custom, `fovstep=2`, volume misplaced | 58-59 | |
| custom, `fovstep=2`, volume in front | 21-22 | |
| + `emptyStride=2` | 44-58 | drops a lot |
| + adaptive brick skipping | 57-58 | |
| + 3× step while turning | 57-58 | 84-88 |

## Lessons

- **The TF changes frame cost about 3×** (14 vs 43 fps on the same scan
  before and after two voice commands). Every comparison has to pin the TF.
- **Check where the volume is before trusting a number** -- a shifted
  reference space made the first half of the session's fps look twice as
  good as they were.
- **`down=1` restarts the VR runtime** (browser, tab and launcher died
  together): at the 2064×2064 eye buffer, a frame takes several hundred ms.
- The flat benchmark predicts direction, not size: `emptyStride` gave −15 %
  there and doubled the frame rate in VR (more transparent space per ray).

## Open

- A scripted benchmark over 2-3 patients and two TFs each, with the volume
  placed the same way, before these numbers go into the thesis.
- Native 90 Hz is not reached; the runtime fills the gaps.
- The LLM parser was not running in the lab (Ollama not started); one
  command fell back to the rule parser and got a 400.
