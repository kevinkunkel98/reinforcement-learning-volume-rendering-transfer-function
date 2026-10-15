(() => {
  // Deterministic GPU cost of one frame: flat renderer, fixed camera, VR-equivalent
  // pixel count, volume filling the view. window.__bench(config) -> ms per frame.
  const r = volumeViewer.getRenderer();
  const w = r.getVTKWindow();
  const glw = w.getViews()[0];
  const vol = r.getVolumes()[0];
  const m = vol.getMapper();
  const gl = glw.getCanvas().getContext("webgl2");
  const cam = r.getActiveCamera();
  window.__bench = (cfg) => {
    glw.setSize(cfg.w || 1152, cfg.h || 576);
    m.setAutoAdjustSampleDistances(false);
    m.setImageSampleDistance(1);
    w.getInteractor().invokeAnimationFrameRateUpdate();
    m.setSampleDistance(cfg.sd);
    vol.getProperty().setShade(Boolean(cfg.shade));
    const b = vol.getBounds();
    const c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
    cam.setFocalPoint(...c);
    cam.setPosition(c[0], c[1] - 1500, c[2]);
    cam.setViewUp(0, 0, 1);
    cam.setViewAngle(30);
    r.resetCameraClippingRange();
    const px = new Uint8Array(4);
    const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    w.render(); sync(); w.render(); sync();
    const n = cfg.n || 8;
    const t0 = performance.now();
    for (let i = 0; i < n; i += 1) { w.render(); sync(); }
    return +((performance.now() - t0) / n).toFixed(1);
  };
  return "bench ready";
})()
