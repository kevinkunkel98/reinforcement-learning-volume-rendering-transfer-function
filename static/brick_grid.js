// Empty-space skipping for the VR ray marcher: per-brick HU range and label
// set (once per volume), and per-brick visibility under the current
// transfer-function table (on every TF/layer change). Pure; loaded as
// window.brickGrid and via require().
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.brickGrid = api;
})(typeof window !== "undefined" ? window : globalThis, () => {
  "use strict";

  // Per brick of `size`^3 voxels: HU min/max and a bitmask of label ids,
  // over the brick plus a one-voxel apron (linear HU filtering at a brick
  // face mixes in the neighbouring voxels). `labels` may be null.
  function brickStats(hu, labels, dims, size = 8) {
    const [nx, ny, nz] = dims;
    const grid = dims.map((d) => Math.ceil(d / size));
    const count = grid[0] * grid[1] * grid[2];
    const huMin = new Float32Array(count).fill(Infinity);
    const huMax = new Float32Array(count).fill(-Infinity);
    const labelMask = new Uint16Array(count);
    const span = (v, n) => [Math.floor(Math.max(v - 1, 0) / size), Math.floor(Math.min(v + 1, n - 1) / size)];
    let index = 0;
    for (let k = 0; k < nz; k += 1) {
      const [k0, k1] = span(k, nz);
      for (let j = 0; j < ny; j += 1) {
        const [j0, j1] = span(j, ny);
        for (let i = 0; i < nx; i += 1, index += 1) {
          const [i0, i1] = span(i, nx);
          const value = hu[index];
          const bit = 1 << (labels ? labels[index] : 0);
          for (let bk = k0; bk <= k1; bk += 1) {
            for (let bj = j0; bj <= j1; bj += 1) {
              for (let bi = i0; bi <= i1; bi += 1) {
                const b = bi + grid[0] * (bj + grid[1] * bk);
                if (value < huMin[b]) huMin[b] = value;
                if (value > huMax[b]) huMax[b] = value;
                labelMask[b] |= bit;
              }
            }
          }
        }
      }
    }
    return { grid, size, huMin, huMax, labelMask };
  }

  // 255 where some label present in the brick has non-zero alpha anywhere
  // in [huMin, huMax] of its table row (HU clamped to the table's range, as
  // the shader clamps), else 0. Unlabelled tables (one row) use row 0.
  function occupancy(stats, rows, epsilon = 0.5 / 255) {
    const lo = rows[0][0].hu;
    const hi = rows[0][rows[0].length - 1].hu;
    const last = rows[0].length - 1;
    const toIndex = (h) => (Math.max(lo, Math.min(hi, h)) - lo) / (hi - lo) * last;
    const out = new Uint8Array(stats.huMin.length);
    for (let b = 0; b < out.length; b += 1) {
      const mask = stats.labelMask[b];
      if (!mask) continue;
      const i0 = Math.max(0, Math.floor(toIndex(stats.huMin[b])));
      const i1 = Math.min(last, Math.ceil(toIndex(stats.huMax[b])));
      for (let label = 0; label < 16 && !out[b]; label += 1) {
        if (!(mask & (1 << label))) continue;
        const row = rows[Math.min(label, rows.length - 1)];
        for (let i = i0; i <= i1; i += 1) {
          if (row[i].alpha > epsilon) { out[b] = 255; break; }
        }
      }
    }
    return out;
  }

  function occupiedFraction(occ) {
    let n = 0;
    for (const v of occ) if (v) n += 1;
    return occ.length ? n / occ.length : 0;
  }

  return { brickStats, occupancy, occupiedFraction };
});
