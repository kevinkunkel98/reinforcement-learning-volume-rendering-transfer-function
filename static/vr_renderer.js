// Custom WebGL2 volume ray marcher for VR (docs/superpowers/specs/
// 2026-10-15-vr-raymarcher-design.md). Separate HU (R16F, linear) and label
// (R8, nearest) textures, the shared TF table, empty-space skipping over
// 8^3-voxel bricks and early ray termination. The flat page stays on vtk.js.
(() => {
  "use strict";

  const BRICK = 8;
  const MAX_STEPS = 2048;

  const VOLUME_VS = `#version 300 es
in vec2 aPos;
out vec2 vNdc;
void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

  const VOLUME_FS = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;
uniform sampler3D uHu;
uniform sampler3D uLabel;
uniform sampler3D uOcc;
uniform sampler2D uTf;
uniform mat4 uInvMvp;      // NDC -> data mm
uniform vec3 uDims;
uniform vec3 uSpacing;
uniform vec3 uOccDims;
uniform float uStep;       // mm
uniform float uOpacityExp; // step / opacity unit distance
uniform float uRows;
uniform vec2 uHuRange;
uniform vec3 uEye;         // eye in data mm (headlight)
uniform vec4 uLight;       // ambient, diffuse, specular, specular power
uniform float uSkip;       // 1 = empty-space skipping on
uniform float uLit;        // 1 = gradient lighting on
in vec2 vNdc;
out vec4 outColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec2 boxHit(vec3 ro, vec3 inv, vec3 lo, vec3 hi) {
  vec3 t0 = (lo - ro) * inv;
  vec3 t1 = (hi - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
  vec4 nearP = uInvMvp * vec4(vNdc, -1.0, 1.0);
  vec4 farP = uInvMvp * vec4(vNdc, 1.0, 1.0);
  vec3 ro = nearP.xyz / nearP.w;
  vec3 rd = normalize(farP.xyz / farP.w - ro);
  vec3 inv = 1.0 / rd;
  vec3 hi = (uDims - 1.0) * uSpacing;
  vec2 hit = boxHit(ro, inv, vec3(0.0), hi);
  if (hit.y <= max(hit.x, 0.0)) { outColor = vec4(0.0); return; }
  float t = max(hit.x, 0.0) + hash(gl_FragCoord.xy) * uStep;
  vec4 acc = vec4(0.0);
  vec3 texel = 1.0 / uDims;
  for (int i = 0; i < ${MAX_STEPS}; i++) {
    if (t > hit.y || acc.a >= 0.98) break;
    vec3 p = ro + rd * t;
    vec3 vox = p / uSpacing;
    ivec3 iv = clamp(ivec3(floor(vox + 0.5)), ivec3(0), ivec3(uDims) - 1);
    if (uSkip > 0.5) {
      ivec3 b = iv / ${BRICK};
      if (texelFetch(uOcc, b, 0).r < 0.5) {
        vec3 blo = (vec3(b * ${BRICK}) - 0.5) * uSpacing;
        vec3 bhi = blo + float(${BRICK}) * uSpacing;
        vec2 bh = boxHit(p, inv, blo, bhi);
        t += max(bh.y, 0.0) + 0.01 * uStep;
        continue;
      }
    }
    vec3 tc = (vox + 0.5) * texel;
    float hu = texture(uHu, tc).r;
    float row = uRows > 1.5 ? texelFetch(uLabel, iv, 0).r * 255.0 : 0.0;
    float u = clamp((hu - uHuRange.x) / (uHuRange.y - uHuRange.x), 0.0, 1.0);
    vec4 c = texture(uTf, vec2(u * 255.0 / 256.0 + 0.5 / 256.0, (floor(row + 0.5) + 0.5) / uRows));
    if (c.a > 0.002) {
      float a = 1.0 - pow(1.0 - c.a, uOpacityExp);
      vec3 g = uLit < 0.5 ? vec3(0.0) : vec3(
        texture(uHu, tc + vec3(texel.x, 0.0, 0.0)).r - texture(uHu, tc - vec3(texel.x, 0.0, 0.0)).r,
        texture(uHu, tc + vec3(0.0, texel.y, 0.0)).r - texture(uHu, tc - vec3(0.0, texel.y, 0.0)).r,
        texture(uHu, tc + vec3(0.0, 0.0, texel.z)).r - texture(uHu, tc - vec3(0.0, 0.0, texel.z)).r) / uSpacing;
      float glen = length(g);
      vec3 rgb = c.rgb * (uLight.x + uLight.y);
      if (glen > 1e-3) {
        vec3 n = -g / glen;
        vec3 l = normalize(uEye - p);
        float ndl = abs(dot(n, l));
        rgb = c.rgb * (uLight.x + uLight.y * ndl) + uLight.z * pow(ndl, uLight.w);
      }
      acc.rgb += (1.0 - acc.a) * a * rgb;
      acc.a += (1.0 - acc.a) * a;
    }
    t += uStep;
  }
  outColor = acc;
}`;

  const HUD_VS = `#version 300 es
in vec3 aPos;
in vec2 aUv;
uniform mat4 uMvp;
out vec2 vUv;
void main() { vUv = aUv; gl_Position = uMvp * vec4(aPos, 1.0); }`;

  const HUD_FS = `#version 300 es
precision mediump float;
uniform sampler2D uTex;
in vec2 vUv;
out vec4 outColor;
void main() { outColor = texture(uTex, vUv); }`;

  function compile(gl, vs, fs) {
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(`shader compile failed: ${gl.getShaderInfoLog(shader)}`);
      }
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`shader link failed: ${gl.getProgramInfoLog(program)}`);
    }
    const uniforms = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i += 1) {
      const name = gl.getActiveUniform(program, i).name;
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return { program, uniforms };
  }

  function texture3D(gl, internalFormat, format, type, dims, data, filter) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, internalFormat, dims[0], dims[1], dims[2], 0, format, type, data);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, filter);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
      gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    }
    return tex;
  }

  // One renderer per GL context: textures, programs, TF state.
  function createScene(gl, data, { down = 1 } = {}) {
    const logic = window.xrLogic;
    const max3d = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
    let { hu, labels, dims } = data;
    ({ values: hu, dims } = logic.strideDownsample(hu, data.dims, down));
    if (labels) labels = logic.strideDownsample(labels, data.dims, down).values;
    if (dims.some((d) => d > max3d)) throw new Error(`volume ${dims.join("x")} exceeds MAX_3D_TEXTURE_SIZE ${max3d}`);
    const spacing = data.spacing.map((s) => s * down);
    const huClamped = Float32Array.from(hu, (v) => Math.max(-1050, Math.min(2000, v)));
    const stats = window.brickGrid.brickStats(huClamped, labels, dims, BRICK);

    const volumeProgram = compile(gl, VOLUME_VS, VOLUME_FS);
    const hudProgram = compile(gl, HUD_VS, HUD_FS);
    const quad = gl.createVertexArray();
    gl.bindVertexArray(quad);
    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(volumeProgram.program, "aPos");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    const huTex = texture3D(gl, gl.R16F, gl.RED, gl.FLOAT, dims, huClamped, gl.LINEAR);
    const labelTex = texture3D(gl, gl.R8, gl.RED, gl.UNSIGNED_BYTE, dims,
      labels || new Uint8Array(dims[0] * dims[1] * dims[2]), gl.NEAREST);
    let occTex = null;
    let tfTex = null;
    let rowCount = 1;
    let occupied = 1;

    function setTransferFunction(params, layers, labelled) {
      const rows = window.tfTable.tfRows(params, layers, labelled);
      rowCount = rows.length;
      if (tfTex) gl.deleteTexture(tfTex);
      tfTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tfTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, rows[0].length, rows.length, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        window.tfTable.tfTextureData(rows));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const occ = window.brickGrid.occupancy(stats, rows);
      occupied = window.brickGrid.occupiedFraction(occ);
      if (occTex) gl.deleteTexture(occTex);
      occTex = texture3D(gl, gl.R8, gl.RED, gl.UNSIGNED_BYTE, stats.grid, occ, gl.NEAREST);
    }
    setTransferFunction(data.params, data.layers, Boolean(labels));

    const baseStep = Math.min(...spacing);
    const state = { step: baseStep, skip: true, lit: true };

    // Draw the volume for one view into the bound framebuffer/viewport.
    // `model`: data mm -> room m; `view`: room -> eye; `proj`: eye -> clip.
    function drawVolume(model, view, proj) {
      const mv = window.vrMath.multiply(view, model);
      const invMvp = window.vrMath.invert(window.vrMath.multiply(proj, mv));
      const eye = window.vrMath.transformPoint(window.vrMath.invert(mv), [0, 0, 0]);
      const { program, uniforms: u } = volumeProgram;
      gl.useProgram(program);
      [[huTex, gl.TEXTURE_3D, "uHu"], [labelTex, gl.TEXTURE_3D, "uLabel"],
        [occTex, gl.TEXTURE_3D, "uOcc"], [tfTex, gl.TEXTURE_2D, "uTf"]].forEach(([tex, target, name], unit) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(target, tex);
        gl.uniform1i(u[name], unit);
      });
      gl.uniformMatrix4fv(u.uInvMvp, false, invMvp);
      gl.uniform3fv(u.uDims, dims);
      gl.uniform3fv(u.uSpacing, spacing);
      gl.uniform1f(u.uStep, state.step);
      gl.uniform1f(u.uOpacityExp, state.step / data.unitDistance);
      gl.uniform1f(u.uRows, rowCount);
      gl.uniform2f(u.uHuRange, -1050, 2000);
      gl.uniform3fv(u.uEye, eye);
      gl.uniform4f(u.uLight, data.light.ambient, data.light.diffuse, data.light.specular, data.light.specularPower);
      gl.uniform1f(u.uSkip, state.skip ? 1 : 0);
      gl.uniform1f(u.uLit, state.lit ? 1 : 0);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(quad);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

    // HUD quad (corners in data mm, placed by `model` without rotation).
    let hud = null;
    function setHud(canvas, corners) {
      const [origin, p1, p2] = corners;
      const p3 = p1.map((v, i) => v + p2[i] - origin[i]);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from([
        ...origin, 0, 1, ...p1, 1, 1, ...p2, 0, 0,
        ...p2, 0, 0, ...p1, 1, 1, ...p3, 1, 0,
      ]), gl.STATIC_DRAW);
      const posLoc = gl.getAttribLocation(hudProgram.program, "aPos");
      const uvLoc = gl.getAttribLocation(hudProgram.program, "aUv");
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 20, 0);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 20, 12);
      gl.bindVertexArray(null);
      const tex = gl.createTexture();
      hud = { canvas, vao, buffer, tex };
      refreshHud();
    }
    function refreshHud() {
      if (!hud) return;
      gl.bindTexture(gl.TEXTURE_2D, hud.tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, hud.canvas);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    function drawHud(model, view, proj) {
      if (!hud) return;
      const { program, uniforms: u } = hudProgram;
      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, hud.tex);
      gl.uniform1i(u.uTex, 0);
      gl.uniformMatrix4fv(u.uMvp, false, window.vrMath.multiply(proj, window.vrMath.multiply(view, model)));
      gl.bindVertexArray(hud.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
    }

    function dispose() {
      [huTex, labelTex, occTex, tfTex, hud?.tex].forEach((t) => t && gl.deleteTexture(t));
      [quadBuffer, hud?.buffer].forEach((b) => b && gl.deleteBuffer(b));
      gl.deleteProgram(volumeProgram.program);
      gl.deleteProgram(hudProgram.program);
    }

    return {
      dims, spacing, baseStep, state, drawVolume, drawHud, setHud, refreshHud,
      setTransferFunction, dispose, get occupied() { return occupied; },
    };
  }

  // Headset benchmark without XR: offscreen canvas, fixed camera looking at
  // the volume centre from -y (as the Stage C rdp/bench.js did), volume
  // filling the view. Returns ms per frame (readPixels-synced) and,
  // optionally, a PNG data URL for screenshot parity.
  function benchmark({ w = 1152, h = 576, n = 8, skip = true, lit = true, stepScale = 1, down = 1, image = false } = {}) {
    const data = window.volumeViewer.getVolumeData();
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: image });
    const scene = createScene(gl, data, { down });
    scene.state.skip = skip;
    scene.state.lit = lit;
    scene.state.step = scene.baseStep * stepScale;
    const center = data.dims.map((d, i) => (d - 1) * data.spacing[i] / 2);
    const V = window.vrMath.lookAt([center[0], center[1] - 1500, center[2]], center, [0, 0, 1]);
    const P = window.vrMath.perspective(Math.PI / 6, w / h, 10, 10000);
    const M = window.vrMath.identity();
    const px = new Uint8Array(4);
    const frame = () => {
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      scene.drawVolume(M, V, P);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    };
    frame(); frame();
    const t0 = performance.now();
    for (let i = 0; i < n; i += 1) frame();
    const ms = +((performance.now() - t0) / n).toFixed(1);
    const result = { ms, occupied: +scene.occupied.toFixed(3), dims: scene.dims, step: scene.state.step };
    if (image) result.image = canvas.toDataURL("image/png");
    scene.dispose();
    return JSON.stringify(result);
  }

  window.vrRenderer = { createScene, benchmark };
})();
