// Immersive VR mode: shows the local vtk.js volume in a WebXR headset, with
// controller push-to-talk, history buttons and a read-only status panel.
// Pure logic lives in xr_logic.js; this file only touches WebXR/vtk/DOM.
(() => {
  "use strict";

  const logic = window.xrLogic;
  const button = document.getElementById("enter-vr-btn");
  const HUD_W = 1024;
  const HUD_H = 384;
  const RECORDING_BG = [0.25, 0, 0];

  let session = null;
  let hudState = logic.HUD_INITIAL;
  let busy = false;
  let hud = null; // { canvas, ctx, texture, actor }
  let flatBackground = null;
  let entering = false;
  let fps = null;
  let frames = 0;
  let fpsSince = null;
  let mapper = null; // fresh per session, so a trigger held at exit can't carry over

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
    if (fps !== null) {
      ctx.textAlign = "right";
      ctx.fillText(`${fps} fps`, HUD_W - pad, 300);
      ctx.textAlign = "left";
    }
    hud.texture.modified();
  }

  function addHud() {
    const renderer = window.volumeViewer.getRenderer();
    const bounds = window.volumeViewer.getVolumeBounds();
    const axes = window.volumeViewer.getCameraAxes();
    const canvas = document.createElement("canvas");
    canvas.width = HUD_W;
    canvas.height = HUD_H;
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
    hud = { canvas, ctx: canvas.getContext("2d"), texture, actor };
    drawHud();
  }

  function removeHud() {
    if (!hud) return;
    window.volumeViewer.getRenderer()?.removeActor(hud.actor);
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
    const renderer = window.volumeViewer.getRenderer();
    if (next.status === "recording" && prev.status !== "recording") {
      renderer.setBackground(...RECORDING_BG);
      pulseAll(0.5, 40);
    } else if (prev.status === "recording" && next.status !== "recording") {
      renderer.setBackground(...flatBackground);
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
    // Frame rate on the panel, refreshed once a second (a redraw re-uploads
    // the panel texture, so not every frame).
    frames += 1;
    if (fpsSince === null) fpsSince = time;
    if (time - fpsSince >= 1000) {
      fps = Math.round((frames * 1000) / (time - fpsSince));
      frames = 0;
      fpsSince = time;
      drawHud();
    }
    const sources = [];
    for (const source of frame.session.inputSources) {
      if (!source.gamepad) continue;
      sources.push({ handedness: source.handedness, pressed: source.gamepad.buttons.map((b) => b.pressed) });
    }
    mapper.update(sources, busy);
  }

  async function onSessionEnd() {
    if (hudState.status === "recording") window.appControls.stopRecording();
    session = null;
    mapper = null;
    removeHud();
    button.textContent = "Enter VR";
    try {
      await window.volumeViewer.exitXR();
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
      await window.volumeViewer.enterXR(xrSession);
    } catch (err) {
      xrSession.end().catch(() => {});
      window.appControls.notify(`Could not start VR: ${err.message}`);
      return;
    }
    xrSession.removeEventListener("end", markEnded);
    if (endedEarly) {
      await window.volumeViewer.exitXR();
      return;
    }
    session = xrSession;
    session.addEventListener("end", onSessionEnd, { once: true });
    mapper = createMapper();
    fps = null;
    frames = 0;
    fpsSince = null;
    flatBackground = window.volumeViewer.getRenderer().getBackground().slice();
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
