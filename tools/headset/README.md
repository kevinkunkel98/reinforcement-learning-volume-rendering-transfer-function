# Headset debugging tools (Vive Focus 3 over USB)

Drive and measure the web app running in the headset's browser from the Mac.
The VIVE Browser is Wolvic on **Gecko 105** (Firefox engine), so this talks the
Firefox Remote Debugging Protocol, not Chrome DevTools.

## Setup

```bash
adb devices                       # FA1AK2R00310 ... VIVE_Focus_3
adb forward tcp:6000 localabstract:com.vive.browser.wave/firefox-debugger-socket
.venv/bin/python server.py --host 0.0.0.0 --ssl-certfile certs/lan.pem --ssl-keyfile certs/lan-key.pem
```

Re-run the `adb forward` after any USB reconnect. If a page load hangs at
"Loading local volume...", the headset is asleep (`navigator.xr` checks never
resolve then): `adb shell input keyevent KEYCODE_WAKEUP`.

## URL flags

A headset browser gets `renderer=custom down=2 xrscale=0.8 tex=f16 fovstep=2`
without flags (`static/xr_logic.js`, `HEADSET_PERF_DEFAULTS`); any flag in the
URL overrides its default:

| Flag | Values | |
|---|---|---|
| `renderer` | `custom` \| `vtk` | custom WebGL2 ray marcher or vtk.js |
| `down` | `1` \| `2` | volume resolution; `1` is too slow for the Focus 3 (restarts the runtime) |
| `xrscale` | 0.1–2 | WebXR framebuffer scale |
| `fovstep` | 1–4 | step multiplier at the edge of each eye's view (custom only) |
| `tex`, `xrsample`, `adapt`, `crop` | | vtk.js path |

## Tools

| File | Use |
|---|---|
| `rdp.py` | Protocol client; `python tools/headset/rdp.py` lists the open tabs |
| `page.py <url-substring> '<js>'` | Evaluate JS in the matching tab, print the result (no top-level `await`: stash promise results on `window` and poll) |
| `console.py <url-substring> [n]` | Last cached console messages/errors |
| `grab.py <url-substring> '<js returning a data URL>' out.png` | Save a canvas image (handles long strings) |
| `bench.js` | Paste through `page.py` to install `window.__bench({sd, w, h, shade})`: ms per frame of the **flat vtk.js** renderer at a fixed camera, `readPixels`-synced |
| `perf.sh [seconds]` | Median compositor FPS / GPU % from `VRPerfMgr` in logcat |
| `fps.sh '<js>' [seconds]` | Apply JS, then count frames of the **vtk** VR path (wraps `volumeViewer.getPhysicalRotation`; for `?renderer=custom` read the `[xr-perf]` lines instead) |

The custom renderer has its own benchmark: `page.py 8000 'vrRenderer.benchmark({w: 1152, h: 576, down: 2, stepScale: 1})'`.

In-VR frame rate is always in logcat: `adb logcat | grep xr-perf`.
Both eyes' display: `adb exec-out screencap -p > shot.png` (vertically flipped).

Benchmarks drift with headset temperature: repeat runs and compare within one sitting.
