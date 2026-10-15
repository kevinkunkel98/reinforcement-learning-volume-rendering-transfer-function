// The label-scoped transfer-function table shared by the flat vtk.js view
// (viewer.js) and the VR ray marcher (vr_renderer.js), so both shade every
// class the same way. Pure; loaded as window.tfTable and via require().
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.tfTable = api;
})(typeof window !== "undefined" ? window : globalThis, () => {
  "use strict";

  const CENTER_RANGE = [-1050, 2000];
  const WIDTH_RANGE = [10, 400];
  const PARAMS_PER_PEAK = 6;
  const SAMPLES = 256;
  // Mirrors transfer.CANONICAL_PEAK_ORDER and transfer.UNLABELLED_PEAKS.
  const PEAK_ORDER = ["lungs", "soft", "liver", "kidneys", "spleen", "heart", "vessels", "skeleton"];
  const UNLABELLED_PEAKS = ["skeleton", "lungs", "soft"];
  // Label id - 1 -> class (viewer.js CLASS_IDS).
  const CLASS_ORDER = ["skeleton", "lungs", "heart", "vessels", "liver", "kidneys", "spleen", "soft"];

  function internalPeak(params, index) {
    const base = index * PARAMS_PER_PEAK;
    const toRange = (value, range) => range[0] + (value + 1) / 2 * (range[1] - range[0]);
    const unit = (value) => (value + 1) / 2;
    return {
      center: toRange(params[base], CENTER_RANGE),
      width: toRange(params[base + 1], WIDTH_RANGE),
      height: unit(params[base + 2]),
      rgb: [unit(params[base + 3]), unit(params[base + 4]), unit(params[base + 5])],
    };
  }

  function blendPeaks(params, hu, peakNames, scale = 1) {
    let alpha = 0;
    let weight = 1e-6;
    const rgb = [0, 0, 0];
    for (const name of peakNames) {
      const peak = internalPeak(params, PEAK_ORDER.indexOf(name));
      const contribution = peak.height * Math.exp(-0.5 * ((hu - peak.center) / peak.width) ** 2);
      alpha += contribution;
      weight += contribution;
      peak.rgb.forEach((value, channel) => { rgb[channel] += contribution * value; });
    }
    return {
      alpha: Math.max(0, Math.min(1, alpha)) * scale,
      rgb: rgb.map((value) => Math.max(0, Math.min(1, value / weight))),
    };
  }

  // Label-scoped rows (transfer.label_scoped_tables): row 0 unlabelled,
  // row id for CLASS_ORDER[id - 1], shaded by that class's peak alone and
  // scaled by its layer opacity. Without labels: one shared HU row.
  // Each row: SAMPLES x {hu, rgb, alpha} evenly over CENTER_RANGE.
  function tfRows(params, layers, labelled) {
    const rows = labelled
      ? [{ peaks: UNLABELLED_PEAKS, scale: 1 }].concat(CLASS_ORDER.map((className) => (
        { peaks: [className], scale: layers?.[className]?.opacity ?? 1 })))
      : [{ peaks: PEAK_ORDER, scale: 1 }];
    return rows.map((row) => Array.from({ length: SAMPLES }, (_, i) => {
      const hu = CENTER_RANGE[0] + i / (SAMPLES - 1) * (CENTER_RANGE[1] - CENTER_RANGE[0]);
      return { hu, ...blendPeaks(params, hu, row.peaks, row.scale) };
    }));
  }

  // RGBA8 texture data, row after row (height = rows.length, width = SAMPLES).
  function tfTextureData(rows) {
    const out = new Uint8Array(rows.length * rows[0].length * 4);
    let o = 0;
    for (const row of rows) {
      for (const { rgb, alpha } of row) {
        out[o++] = Math.round(rgb[0] * 255);
        out[o++] = Math.round(rgb[1] * 255);
        out[o++] = Math.round(rgb[2] * 255);
        out[o++] = Math.round(alpha * 255);
      }
    }
    return out;
  }

  return {
    CENTER_RANGE, WIDTH_RANGE, SAMPLES, PEAK_ORDER, UNLABELLED_PEAKS, CLASS_ORDER,
    tfRows, tfTextureData,
  };
});
