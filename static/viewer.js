(() => {
  "use strict";

  const CENTER_RANGE = [-1050, 2000];
  const WIDTH_RANGE = [10, 400];
  const viewerEl = document.getElementById("vtk-viewer");
  const statusEl = document.getElementById("viewer-status");
  const errorEl = document.getElementById("viewer-error");
  const fallbackEl = document.getElementById("current-image");
  let renderer;
  let renderWindow;
  let volume;
  let mapper;
  let imageData;
  let camera;
  let datasetName;
  let transferFunction;

  function setStatus(message, error = false) {
    statusEl.textContent = message;
    statusEl.hidden = error;
    errorEl.textContent = error ? message : "";
    errorEl.hidden = !error;
  }

  async function fetchVolume(name) {
    const metadataResponse = await fetch(`/api/datasets/${encodeURIComponent(name)}/metadata`);
    if (!metadataResponse.ok) throw new Error(`volume metadata request failed (${metadataResponse.status})`);
    const metadata = await metadataResponse.json();
    const chunks = await Promise.all(metadata.chunks.map(async (chunk) => {
      const response = await fetch(`/api/datasets/${encodeURIComponent(name)}/chunks/${chunk.index}`);
      if (!response.ok) throw new Error(`volume chunk ${chunk.index} request failed (${response.status})`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength !== chunk.byte_length) throw new Error(`volume chunk ${chunk.index} has invalid length`);
      return { ...chunk, bytes };
    }));
    const raw = new Uint8Array(metadata.total_bytes);
    chunks.forEach(({ byte_offset: offset, bytes }) => raw.set(bytes, offset));
    if (metadata.byte_order !== "little" || metadata.scalar_type !== "float32") {
      throw new Error("unsupported volume scalar format");
    }
    if (metadata.order !== "F") throw new Error("unsupported volume storage order");
    return { metadata, values: new Float32Array(raw.buffer) };
  }

  function buildImageData(metadata, values) {
    imageData = vtk.Common.DataModel.vtkImageData.newInstance();
    imageData.setDimensions(...metadata.dimensions);
    imageData.setSpacing(...metadata.spacing);
    imageData.getPointData().setScalars(vtk.Common.Core.vtkDataArray.newInstance({
      name: "Scalars", values, numberOfComponents: 1,
    }));
    return imageData;
  }

  function internalPeak(params, index) {
    const base = index * 6;
    const toRange = (value, range) => range[0] + (value + 1) / 2 * (range[1] - range[0]);
    const unit = (value) => (value + 1) / 2;
    return {
      center: toRange(params[base], CENTER_RANGE),
      width: toRange(params[base + 1], WIDTH_RANGE),
      height: unit(params[base + 2]),
      rgb: [unit(params[base + 3]), unit(params[base + 4]), unit(params[base + 5])],
    };
  }

  function setTransferFunction(params) {
    if (!Array.isArray(params) && !(params instanceof Float32Array) && !(params instanceof Float64Array)) {
      throw new Error("transfer function must be an array");
    }
    if (params.length !== 24) throw new Error("transfer function must contain 24 values");
    const color = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    const opacity = vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();
    for (let i = 0; i < 256; i += 1) {
      const hu = CENTER_RANGE[0] + i / 255 * (CENTER_RANGE[1] - CENTER_RANGE[0]);
      let alpha = 0;
      const rgb = [0, 0, 0];
      let weight = 1e-6;
      for (let peakIndex = 0; peakIndex < 4; peakIndex += 1) {
        const peak = internalPeak(params, peakIndex);
        const contribution = peak.height * Math.exp(-0.5 * ((hu - peak.center) / peak.width) ** 2);
        alpha += contribution;
        weight += contribution;
        peak.rgb.forEach((value, channel) => { rgb[channel] += contribution * value; });
      }
      color.addRGBPoint(hu, ...rgb.map((value) => Math.max(0, Math.min(1, value / weight))));
      opacity.addPoint(hu, Math.max(0, Math.min(1, alpha)));
    }
    transferFunction = params.slice();
    volume.getProperty().setRGBTransferFunction(0, color);
    volume.getProperty().setScalarOpacity(0, opacity);
    renderWindow.render();
  }

  function getCamera() {
    return camera ? {
      position: camera.getPosition().slice(),
      focal_point: camera.getFocalPoint().slice(),
      view_up: camera.getViewUp().slice(),
      zoom: camera.getParallelScale(),
    } : null;
  }

  function setCamera(value) {
    if (!camera || !value) return;
    if (value.position) camera.setPosition(...value.position);
    if (value.focal_point) camera.setFocalPoint(...value.focal_point);
    if (value.view_up) camera.setViewUp(...value.view_up);
    if (value.parallel_scale) camera.setParallelScale(value.parallel_scale);
    if (value.zoom && value.zoom !== 1) camera.dolly(value.zoom);
    if (value.azimuth) camera.azimuth(value.azimuth);
    if (value.elevation) camera.elevation(value.elevation);
    renderWindow.render();
  }

  async function load(name, params, cameraState) {
    if (!window.vtk) throw new Error("vtk.js unavailable");
    if (volume && datasetName === name) {
      setCamera(cameraState);
      setTransferFunction(params);
      return true;
    }
    datasetName = name;
    viewerEl.hidden = false;
    fallbackEl.hidden = false;
    setStatus("Loading local volume...");
    try {
      const loaded = await fetchVolume(name);
      if (!renderer) {
        const openGLRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
        renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [0, 0, 0] });
        renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
        renderWindow.addRenderer(renderer);
        renderWindow.addView(openGLRenderWindow);
        openGLRenderWindow.setContainer(viewerEl);
        openGLRenderWindow.setSize(viewerEl.clientWidth || 640, viewerEl.clientHeight || 480);
        camera = renderer.getActiveCamera();
      }
      buildImageData(loaded.metadata, loaded.values);
      mapper = vtk.Rendering.Core.vtkVolumeMapper.newInstance();
      mapper.setInputData(imageData);
      volume = vtk.Rendering.Core.vtkVolume.newInstance();
      volume.setMapper(mapper);
      renderer.removeAllVolumes();
      renderer.addVolume(volume);
      renderer.resetCamera();
      setCamera(cameraState);
      setTransferFunction(params);
      fallbackEl.hidden = true;
      setStatus(`Local ${datasetName} volume`);
      return true;
    } catch (error) {
      viewerEl.hidden = true;
      fallbackEl.hidden = false;
      setStatus(`Local viewer unavailable: ${error.message}`, true);
      return false;
    }
  }

  window.volumeViewer = { load, setTransferFunction, getCamera, setCamera, render: () => renderWindow?.render(), get dataset() { return datasetName; }, get transferFunction() { return transferFunction; } };
})();
