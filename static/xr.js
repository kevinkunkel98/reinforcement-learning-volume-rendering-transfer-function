// Immersive VR mode: shows the local vtk.js volume in a WebXR headset, with
// controller push-to-talk, history buttons and a read-only status panel.
// Pure logic lives in xr_logic.js; this file only touches WebXR/vtk/DOM.
(() => {
  "use strict";

  const logic = window.xrLogic;
  const button = document.getElementById("enter-vr-btn");
  const HUD_W = 1024;
  const HUD_H = 420;
  // The panel canvas is drawn at this multiple of HUD_W x HUD_H so its text
  // stays sharp at headset resolution; layout coordinates stay in HUD units.
  const HUD_PIXEL_SCALE = 2;
  const RECORDING_BG = [0.25, 0, 0];

  let session = null;
  let hudState = logic.HUD_INITIAL;
  let busy = false;
  let hud = null; // { canvas, ctx, refresh, actor? }
  // The renderer of the running session: window.vrRenderer (?renderer=custom)
  // or window.volumeViewer (vtk.js). Both offer getPhysicalRotation,
  // setVolumeRotation, setXRSampleMultiplier and getXRDiagnostics.
  let target = window.volumeViewer;
  let flatBackground = null;
  let entering = false;
  let fps = null;
  let frameMs = null;
  let adapt = null; // controller state when ?adapt=1, else null
  let lastFrameTime = null;
  let frames = 0;
  let fpsSince = null;
  let mapper = null; // fresh per session, so a trigger held at exit can't carry over
  let sticks = null; // thumbstick rotation, fresh per session
  let lastStickTime = null;

  function createMapper() {
    return logic.createControllerMapper({
      startRecording: () => {
        if (!hudState.voice) return pulseAll(0.6, 60, 2);
        window.appControls.startRecording();
      },
      stopRecording: () => window.appControls.stopRecording(),
      back: () => window.appControls.back(),
      forward: () => window.appControls.forward(),
    });
  }

  // ---------- haptics ----------

  function pulseAll(intensity, ms, count = 1) {
    if (!session) return;
    for (const source of session.inputSources) {
      const actuator = source.gamepad?.hapticActuators?.[0];
      if (!actuator?.pulse) continue;
      for (let i = 0; i < count; i += 1) {
        setTimeout(() => actuator.pulse(intensity, ms), i * (ms + 80));
      }
    }
  }

  // ---------- HUD ----------

  function wrapLine(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let cut = text;
    while (cut.length > 1 && ctx.measureText(`${cut}…`).width > maxWidth) cut = cut.slice(0, -1);
    return `${cut}…`;
  }

  function drawHud() {
    if (!hud) return;
    const { ctx } = hud;
    ctx.setTransform(HUD_PIXEL_SCALE, 0, 0, HUD_PIXEL_SCALE, 0, 0);
    const [head, transcript, summary, footer] = logic.hudLines(hudState);
    ctx.fillStyle = "rgba(12, 12, 16, 0.92)";
    ctx.fillRect(0, 0, HUD_W, HUD_H);
    ctx.strokeStyle = hudState.status === "recording" ? "#ef4444" : "#3f3f46";
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, HUD_W - 8, HUD_H - 8);
    const pad = 40;
    const max = HUD_W - 2 * pad;
    ctx.textBaseline = "top";
    ctx.fillStyle = hudState.status === "error" ? "#fca5a5" : "#fafafa";
    ctx.font = "600 64px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, head, max), pad, 36);
    ctx.fillStyle = "#e4e4e7";
    ctx.font = "italic 44px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, transcript, max), pad, 128);
    ctx.fillStyle = "#a1a1aa";
    ctx.font = "40px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, summary, max), pad, 200);
    ctx.fillStyle = "#71717a";
    ctx.font = "34px system-ui, sans-serif";
    ctx.fillText(wrapLine(ctx, footer, max), pad, 300);
    // Debug lines: real numbers from this exact session plus the active
    // lever config, so a photo of the panel is a complete measurement.
    const [perf, config] = logic.diagnosticLines(target.getXRDiagnostics(), fps, frameMs);
    ctx.fillStyle = "#52525b";
    ctx.font = "26px monospace";
    ctx.fillText(wrapLine(ctx, perf, max), pad, 340);
    ctx.fillText(wrapLine(ctx, config, max), pad, 376);
    hud.refresh();
  }

  function addHud() {
    if (target === window.vrRenderer) {
      const canvas = document.createElement("canvas");
      canvas.width = HUD_W * HUD_PIXEL_SCALE;
      canvas.height = HUD_H * HUD_PIXEL_SCALE;
      const frame = target.getHudFrame();
      const place = logic.hudPlacement(frame.bounds, frame.up, frame.dop, HUD_W / HUD_H);
      hud = { canvas, ctx: canvas.getContext("2d"), refresh: () => target.refreshHud() };
      target.setHud(canvas, [place.origin, place.point1, place.point2]);
      drawHud();
      return;
    }
    const renderer = window.volumeViewer.getRenderer();
    const bounds = window.volumeViewer.getVolumeBounds();
    const axes = window.volumeViewer.getCameraAxes();
    const canvas = document.createElement("canvas");
    canvas.width = HUD_W * HUD_PIXEL_SCALE;
    canvas.height = HUD_H * HUD_PIXEL_SCALE;
    const place = logic.hudPlacement(bounds, axes.up, axes.dop, HUD_W / HUD_H);
    const plane = vtk.Filters.Sources.vtkPlaneSource.newInstance({
      origin: place.origin, point1: place.point1, point2: place.point2,
    });
    const planeMapper = vtk.Rendering.Core.vtkMapper.newInstance();
    planeMapper.setInputConnection(plane.getOutputPort());
    const texture = vtk.Rendering.Core.vtkTexture.newInstance();
    texture.setInterpolate(true);
    texture.setCanvas(canvas);
    const actor = vtk.Rendering.Core.vtkActor.newInstance();
    actor.setMapper(planeMapper);
    actor.addTexture(texture);
    // Unlit: the panel should read the same from any angle.
    actor.getProperty().setAmbient(1);
    actor.getProperty().setDiffuse(0);
    actor.getProperty().setSpecular(0);
    renderer.addActor(actor);
    hud = { canvas, ctx: canvas.getContext("2d"), actor, refresh: () => texture.modified() };
    drawHud();
  }

  function removeHud() {
    if (!hud) return;
    if (hud.actor) window.volumeViewer.getRenderer()?.removeActor(hud.actor);
    hud = null;
  }

  // ---------- app events ----------

  function applyEvent(event) {
    const next = logic.hudReduce(hudState, event);
    if (next === hudState) return;
    const prev = hudState;
    hudState = next;
    if (!session) return;
    drawHud();
    const setBackground = (rgb) => (target === window.vrRenderer
      ? target.setBackground(rgb) : window.volumeViewer.getRenderer().setBackground(...rgb));
    if (next.status === "recording" && prev.status !== "recording") {
      setBackground(RECORDING_BG);
      pulseAll(0.5, 40);
    } else if (prev.status === "recording" && next.status !== "recording") {
      setBackground(flatBackground);
      pulseAll(0.5, 40);
    }
    if (event.type === "applied") pulseAll(0.7, 90);
    if (event.type === "error") pulseAll(0.6, 60, 2);
  }

  for (const type of ["recording", "transcribing", "transcript", "busy", "applied", "error"]) {
    window.addEventListener(`app:${type}`, (e) => {
      if (type === "busy") busy = e.detail.busy;
      applyEvent({ type, ...e.detail });
    });
  }

  // ---------- session ----------

  function onXRFrame(time, frame) {
    if (!session) return;
    session.requestAnimationFrame(onXRFrame);
    if (adapt && lastFrameTime !== null) {
      const next = logic.adaptSampleDistance(adapt, time - lastFrameTime);
      if (next.multiplier !== adapt.multiplier) target.setXRSampleMultiplier(next.multiplier);
      adapt = next;
    }
    lastFrameTime = time;
    // Frame rate on the panel, refreshed once a second (a redraw re-uploads
    // the panel texture, so not every frame).
    frames += 1;
    if (fpsSince === null) fpsSince = time;
    if (time - fpsSince >= 1000) {
      fps = Math.round((frames * 1000) / (time - fpsSince));
      frameMs = (time - fpsSince) / frames;
      frames = 0;
      fpsSince = time;
      drawHud();
      // Same record for remote DevTools / logcat captures.
      console.info("[xr-perf]", ...logic.diagnosticLines(target.getXRDiagnostics(), fps, frameMs));
    }
    const sources = [];
    const stickSources = [];
    for (const source of frame.session.inputSources) {
      if (!source.gamepad) continue;
      const pressed = source.gamepad.buttons.map((b) => b.pressed);
      sources.push({ handedness: source.handedness, pressed });
      // xr-standard puts the thumbstick on axes 2/3 (0/1 are a touchpad) and
      // its press on button 3; fall back to 0/1 for stick-only layouts.
      const axes = source.gamepad.axes;
      const stick = axes.length >= 4 ? [axes[2], axes[3]] : [axes[0] || 0, axes[1] || 0];
      stickSources.push({ handedness: source.handedness, stick, resetPressed: Boolean(pressed[3]) });
    }
    mapper.update(sources, busy);
    const turned = sticks.update(stickSources, target.getPhysicalRotation(),
      lastStickTime === null ? 0 : time - lastStickTime);
    lastStickTime = time;
    if (turned.rotation) target.setVolumeRotation(turned.rotation);
    if (turned.event === "reset") {
      // The custom renderer also brings the volume back in front of you.
      if (target.recenter) target.recenter();
      pulseAll(0.3, 30);
    }
  }

  async function onSessionEnd() {
    if (hudState.status === "recording") window.appControls.stopRecording();
    session = null;
    mapper = null;
    sticks = null;
    adapt = null;
    removeHud();
    button.textContent = "Enter VR";
    try {
      if (target === window.vrRenderer) window.vrRenderer.stop();
      else await window.volumeViewer.exitXR();
    } catch (err) {
      window.appControls.notify(`Leaving VR failed: ${err.message}`);
    }
  }

  async function micAvailable() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      return true;
    } catch (err) {
      return false;
    }
  }

  async function enter() {
    entering = true;
    button.disabled = true;
    try {
      await startSession();
    } finally {
      entering = false;
      button.disabled = false;
    }
  }

  async function startSession() {
    // The mic prompt cannot be shown inside an immersive session: ask first.
    const voice = await micAvailable();
    let xrSession;
    try {
      xrSession = await navigator.xr.requestSession("immersive-vr");
    } catch (err) {
      // A slow permission prompt can use up the click's user activation.
      window.appControls.notify(`Could not start VR (${err.name}). Tap Enter VR again.`);
      return;
    }
    // The session can end (system button, headset taken off) while vtk.js is
    // still setting up; listen from the start so that isn't missed.
    let endedEarly = false;
    const markEnded = () => { endedEarly = true; };
    xrSession.addEventListener("end", markEnded, { once: true });
    try {
      const flags = window.volumeViewer.perfFlags;
      target = window.volumeViewer;
      if (flags.renderer === "custom") {
        try {
          await window.vrRenderer.start(xrSession, { xrscale: flags.xrscale, down: flags.down, fovstep: flags.fovstep });
          target = window.vrRenderer;
        } catch (err) {
          console.error("[vr-renderer]", err);
          window.appControls.notify(`Custom VR renderer failed (${err.message}); using vtk.js.`);
        }
      }
      if (target === window.volumeViewer) await window.volumeViewer.enterXR(xrSession);
    } catch (err) {
      xrSession.end().catch(() => {});
      window.appControls.notify(`Could not start VR: ${err.message}`);
      return;
    }
    xrSession.removeEventListener("end", markEnded);
    if (endedEarly) {
      if (target === window.vrRenderer) window.vrRenderer.stop();
      else await window.volumeViewer.exitXR();
      return;
    }
    session = xrSession;
    session.addEventListener("end", onSessionEnd, { once: true });
    mapper = createMapper();
    sticks = logic.createStickRotator();
    lastStickTime = null;
    fps = null;
    frameMs = null;
    adapt = window.volumeViewer.perfFlags.adapt ? logic.ADAPT_INITIAL : null;
    lastFrameTime = null;
    frames = 0;
    fpsSince = null;
    flatBackground = target === window.vrRenderer
      ? [0, 0, 0] : window.volumeViewer.getRenderer().getBackground().slice();
    hudState = logic.HUD_INITIAL;
    if (!voice) hudState = logic.hudReduce(hudState, { type: "voice-unavailable" });
    addHud();
    button.textContent = "Exit VR";
    session.requestAnimationFrame(onXRFrame);
  }

  button.addEventListener("click", () => {
    if (entering) return;
    // end() rejects if the session already died without an "end" event.
    if (session) session.end().catch(() => onSessionEnd());
    else enter();
  });

  if (navigator.xr?.isSessionSupported) {
    navigator.xr.isSessionSupported("immersive-vr")
      .then((supported) => { button.hidden = !supported; })
      .catch(() => {});
  }
})();
