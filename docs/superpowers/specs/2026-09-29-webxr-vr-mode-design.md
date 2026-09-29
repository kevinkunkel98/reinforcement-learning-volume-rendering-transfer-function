# WebXR VR mode — design

## Goal

View the current volume rendering in stereo on a standalone headset (target:
HTC Vive Focus 3, VIVE Browser) and change the transfer function by voice
without leaving VR. This supersedes the Vrui route in `docs/vr-integration.md`
for headset viewing: no native toolkit, no SteamVR, no Linux machine — the
existing browser client becomes the VR client.

Out of scope for this version: grabbing/rotating/scaling the volume with
controllers, an interactive (point-and-click) VR panel, dataset switching or
compare modes inside VR, server-side sync between several clients.

## Prerequisite (done)

`server.py` accepts `--host`, `--ssl-certfile`, `--ssl-keyfile`. Serving
over HTTPS on the LAN (mkcert certificate in the gitignored `certs/`) gives
the headset the secure context both WebXR and `getUserMedia` require.

## Approach

Use vtk.js's built-in WebXR support in the existing viewer
(`openGLRenderWindow.startXR(false)` / `stopXR()`, present in the pinned
vtk.js 25.11.0 bundle). VR mode renders the *same* renderer/volume as the
flat viewer, so every existing path that updates the transfer function
(`window.volumeViewer.setLabelAwareTransferFunction`, called from `app.js`
on commands, back/forward, reset) updates VR live with no new data path.

Rejected: a separate three.js volume page (duplicates the TF/label-band logic
in `viewer.js`); server-side stereo rendering streamed to the headset
(latency, ignores the headset GPU).

## Components

### `static/xr.js` (new, ~150–200 lines)

Owns everything VR-specific. Depends only on `window.volumeViewer` (XR hooks
below) and on `app:*` events / `window.appControls` from `app.js`.

- **Enter VR button** in the viewer toolbar, shown only when
  `navigator.xr?.isSessionSupported("immersive-vr")` resolves true (so it
  never appears on the Mac).
- **Enter flow:** on click, first call `getUserMedia({audio: true})` and stop
  the tracks immediately — the permission prompt cannot be shown inside an
  immersive session, so it must be granted beforehand. Then
  `volumeViewer.enterXR()`. If mic permission is denied, still enter VR and
  show "voice unavailable" on the HUD.
- **Session end** (system button or `session.end`): `volumeViewer.exitXR()`
  restores the flat viewer; listen for the session's `end` event so exits
  initiated by the headset are handled too.
- **Controller input:** read `inputSource.gamepad` each XR frame via its own
  `session.requestAnimationFrame` loop (WebXR allows several), using the
  `xr-standard` mapping:
  - trigger (`buttons[0]`), either hand: hold to record, release to send →
    `appControls.startRecording()` / `appControls.stopRecording()`.
  - A / X (`buttons[4]`): back → `appControls.back()`.
  - B / Y (`buttons[5]`): forward → `appControls.forward()`.
  - Edge-triggered (press transitions only); input ignored while a command
    is in flight (`app:busy`), matching the flat UI's disabled mic button.
- **Haptics** (`gamepad.hapticActuators?.[0]?.pulse`): short pulse on record
  start and stop, medium pulse when a command is applied, two short pulses on
  error. Missing actuators are silently skipped.
- **HUD panel:** a read-only, world-locked textured quad (`vtkPlaneSource` +
  `vtkTexture.setCanvas`) placed above and slightly in front of the volume's
  bounds, facing the initial view direction. Drawn on a 2D canvas
  (~1024×384), redrawn only when its content changes. Shows:
  - line 1: state — `Hold trigger to talk` / `● Recording…` /
    `Transcribing…` / `Applying…` / error text;
  - line 2: last transcript (`"show the lungs"`);
  - line 3: the reply summary `app.js` already builds for the chat
    (what changed), plus the dataset name;
  - footer hint: `A/X back · B/Y forward`.
  Added to the renderer on enter, removed on exit so it never shows in the
  flat viewer.
- **Recording tint:** renderer background goes dark red while recording,
  back to black otherwise (cheap peripheral cue besides the HUD).

### `static/viewer.js` (small additions)

Expose on `window.volumeViewer`:

- `enterXR()` — save the interactor style and sample distance, switch the
  interactor to a no-op style (vtk.js's trackball style binds right
  trigger/trackpad to camera "fly", which would fight push-to-talk), apply
  XR quality, call `openGLRenderWindow.startXR(false)`.
- `exitXR()` — `stopXR()`, restore style, sample distance and camera.
- `getXRSession()`, `getRenderer()`, `getVolumeBounds()` for `xr.js`.
- `xrSampleDistance(spacing)` pure helper: XR uses `min(spacing)` (flat
  viewer keeps `min(spacing)/2`) — halves ray-march cost per eye on the
  headset's mobile GPU.

### `static/app.js` (small additions)

- `window.appControls = { startRecording, stopRecording, back, forward }`
  wrapping the existing functions (`navigate("/api/back")`, etc.).
- Dispatch `window` `CustomEvent`s at existing points, no behavior change
  for the flat UI: `app:recording` (start/stop), `app:transcribing`,
  `app:transcript` (`{text}`), `app:applied` (`{summary, dataset}` — reuse the
  string `appendReply` renders), `app:error` (`{message}`), `app:busy`
  (`{busy}` from `withLoading`).

### `static/index.html`

Enter VR button (hidden by default) and `<script>` tags for
`/static/xr_logic.js` then `/static/xr.js` after `app.js`.

## Data flow (voice command in VR)

trigger down → `startRecording()` → `app:recording` → HUD `● Recording…`,
red tint, pulse → trigger up → `stopRecording()` → `/api/transcribe` →
`app:transcript` → `sendCommand` → `/api/command` → `refresh()` →
`volumeViewer.setLabelAwareTransferFunction` (volume updates in VR) →
`app:applied` → HUD summary, pulse.

## Error handling

- WebXR unsupported / `isSessionSupported` false → no button.
- `startXR` throws or session request rejected → toast in flat UI, stay flat.
- Mic denied → VR still works; HUD shows "voice unavailable", trigger does
  nothing but a double pulse.
- Parse/transcribe failure → `app:error` → HUD error line + double pulse
  (the flat toast is invisible in VR).
- Dataset too large for the headset (volume fails to load in VIVE Browser) →
  existing fallback path already shows the server PNG; VR button stays
  hidden because there is no local volume. Reducing volume resolution for
  the headset is a separate follow-up, not part of this work.

## Testing

- The repo has no JS test harness. Keep the pure pieces in a DOM-free
  `static/xr_logic.js` (UMD-style: sets `window.xrLogic` in the browser,
  `module.exports` under Node): `xrSampleDistance`, the controller
  edge-detection/state machine (fake gamepad button arrays → expected
  start/stop/back/forward calls, busy gating), and HUD line composition from
  an event sequence. Test with `node --test tests/js/xr_logic.test.js`, and
  add `tests/test_xr_logic.py`, which runs that command via `subprocess` so
  the existing `pytest` run picks it up (skipped when `node` is missing).
- Manual on-device checklist, appended to `docs/vr-integration.md`:
  1. Flat viewer loads the default dataset in VIVE Browser (local volume,
     not PNG fallback).
  2. Enter VR button visible; entering shows the volume in stereo with head
     tracking; framerate acceptable (note subjective smoothness).
  3. Hold trigger → red tint + HUD recording; release → transcript and
     summary on HUD, volume TF changes.
  4. A/X and B/Y step history; volume follows.
  5. System button exits; flat UI intact, camera restored, HUD gone.
  6. Mac browser: no Enter VR button, flat UI unchanged.
