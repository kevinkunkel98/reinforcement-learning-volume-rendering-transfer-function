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
  let openGLRenderWindow;
  let interactor;
  let interactorBound = false;
  let volume;
  let mapper;
  let imageData;
  let camera;
  let datasetName;
  let transferFunction;
  let loadGeneration = 0;
  let loadController;
  let appliedCameraState = null;
  let cameraBaseScale = 1;

  function destroyViewer() {
    if (interactor) {
      if (interactorBound) interactor.unbindEvents();
      interactor.delete();
    }
    if (openGLRenderWindow) openGLRenderWindow.delete();
    if (renderWindow) renderWindow.delete();
    if (renderer) renderer.delete();
    interactor = undefined;
    interactorBound = false;
    openGLRenderWindow = undefined;
    renderWindow = undefined;
    renderer = undefined;
    camera = undefined;
    volume = undefined;
    mapper = undefined;
    imageData = undefined;
  }

  function setStatus(message, error = false) {
    statusEl.textContent = message;
    statusEl.hidden = error;
    errorEl.textContent = error ? message : "";
    errorEl.hidden = !error;
  }

  function validateMetadata(metadata) {
    if (!metadata || !Array.isArray(metadata.dimensions) || metadata.dimensions.length !== 3 ||
        metadata.dimensions.some((value) => !Number.isInteger(value) || value <= 0)) {
      throw new Error("invalid volume dimensions");
    }
    const expectedBytes = metadata.dimensions.reduce((product, value) => product * value, 1) * 4;
    if (!Number.isSafeInteger(expectedBytes) || metadata.total_bytes !== expectedBytes) {
      throw new Error("volume byte count does not match dimensions");
    }
    if (!Array.isArray(metadata.chunks) || metadata.chunks.length === 0) {
      throw new Error("volume has no chunks");
    }
    let offset = 0;
    metadata.chunks.forEach((chunk, index) => {
      if (chunk.index !== index || chunk.byte_offset !== offset ||
          !Number.isInteger(chunk.byte_length) || chunk.byte_length <= 0 ||
          chunk.byte_offset < 0 || chunk.byte_offset + chunk.byte_length > metadata.total_bytes) {
        throw new Error("invalid volume chunk descriptors");
      }
      offset += chunk.byte_length;
    });
    if (offset !== metadata.total_bytes) throw new Error("volume chunks do not cover payload");
    return metadata;
  }

  function isCurrentLoad(generation) {
    return generation === loadGeneration;
  }

  async function fetchVolume(name, signal, generation) {
    const metadataResponse = await fetch(`/api/datasets/${encodeURIComponent(name)}/metadata`, { signal });
    if (!metadataResponse.ok) throw new Error(`volume metadata request failed (${metadataResponse.status})`);
    const metadata = validateMetadata(await metadataResponse.json());
    if (metadata.name !== name) throw new Error("volume metadata dataset mismatch");
    if (metadata.byte_order !== "little") throw new Error("unsupported non-little-endian volume");
    const chunks = await Promise.all(metadata.chunks.map(async (chunk) => {
      const response = await fetch(`/api/datasets/${encodeURIComponent(name)}/chunks/${chunk.index}`, { signal });
      if (!response.ok) throw new Error(`volume chunk ${chunk.index} request failed (${response.status})`);
      if (response.headers.get("X-Dataset-Version") !== metadata.version) {
        throw new Error(`volume chunk ${chunk.index} has mismatched dataset version`);
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength !== chunk.byte_length) throw new Error(`volume chunk ${chunk.index} has invalid length`);
      return { ...chunk, bytes };
    }));
    if (!isCurrentLoad(generation)) throw new DOMException("stale volume load", "AbortError");
    const raw = new Uint8Array(metadata.total_bytes);
    chunks.forEach(({ byte_offset: offset, bytes }) => raw.set(bytes, offset));
    if (metadata.scalar_type !== "float32") throw new Error("unsupported volume scalar format");
    if (metadata.order !== "F") throw new Error("unsupported volume storage order");
    const values = new Float32Array(metadata.total_bytes / 4);
    const dataView = new DataView(raw.buffer);
    for (let index = 0; index < values.length; index += 1) values[index] = dataView.getFloat32(index * 4, true);
    return { metadata, values };
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
    if (!camera) return appliedCameraState ? { ...appliedCameraState } : null;
    return {
      position: camera.getPosition().slice(),
      focal_point: camera.getFocalPoint().slice(),
      view_up: camera.getViewUp().slice(),
      zoom: cameraBaseScale / camera.getParallelScale(),
    };
  }

  function toRendererCamera(value) {
    if (!value) return null;
    if (value.azimuth === undefined && value.elevation === undefined) return { ...value };
    if (!Number.isFinite(value.azimuth) || !Number.isFinite(value.elevation) || !Number.isFinite(value.zoom)) {
      throw new Error("legacy camera values must be finite");
    }
    if (value.zoom <= 0) throw new Error("legacy camera zoom must be positive");
    renderer.resetCamera();
    if (value.azimuth !== 0) camera.azimuth(value.azimuth);
    if (value.elevation !== 0) camera.elevation(value.elevation);
    if (value.zoom !== 1) camera.setParallelScale(camera.getParallelScale() / value.zoom);
    return { ...fromRendererCamera(), zoom: value.zoom || 1 };
  }

  function fromRendererCamera() {
    return {
      position: camera.getPosition().slice(),
      focal_point: camera.getFocalPoint().slice(),
      view_up: camera.getViewUp().slice(),
      zoom: 1,
    };
  }

  function setCamera(value) {
    if (!camera || !value) return;
    appliedCameraState = toRendererCamera(value);
    renderer.resetCamera();
    cameraBaseScale = camera.getParallelScale();
    if (appliedCameraState.position) camera.setPosition(...appliedCameraState.position);
    if (appliedCameraState.focal_point) camera.setFocalPoint(...appliedCameraState.focal_point);
    if (appliedCameraState.view_up) camera.setViewUp(...appliedCameraState.view_up);
    if (appliedCameraState.zoom) camera.setParallelScale(cameraBaseScale / appliedCameraState.zoom);
    renderWindow.render();
  }

  async function load(name, params, cameraState) {
    const generation = ++loadGeneration;
    loadController?.abort();
    loadController = new AbortController();
    viewerEl.hidden = false;
    fallbackEl.hidden = false;
    setStatus("Loading local volume...");
    try {
      if (!window.vtk) throw new Error("vtk.js unavailable");
      if (volume && datasetName === name) {
        setCamera(cameraState);
        setTransferFunction(params);
        fallbackEl.hidden = true;
        setStatus(`Local ${datasetName} volume`);
        return true;
      }
      const loaded = await fetchVolume(name, loadController.signal, generation);
      if (!isCurrentLoad(generation)) return false;
      datasetName = name;
      if (!renderer) {
        openGLRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
        renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [0, 0, 0] });
        renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
        interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
        renderWindow.addRenderer(renderer);
        renderWindow.addView(openGLRenderWindow);
        renderWindow.setInteractor(interactor);
        interactor.setView(openGLRenderWindow);
        openGLRenderWindow.setContainer(viewerEl);
        openGLRenderWindow.setSize(viewerEl.clientWidth || 640, viewerEl.clientHeight || 480);
        interactor.initialize();
        interactor.bindEvents(viewerEl);
        interactorBound = true;
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
      if (!isCurrentLoad(generation) || error.name === "AbortError") return false;
      destroyViewer();
      viewerEl.hidden = true;
      fallbackEl.hidden = false;
      setStatus(`Local viewer unavailable: ${error.message}`, true);
      return false;
    }
  }

  window.volumeViewer = { load, setTransferFunction, getCamera, setCamera, render: () => renderWindow?.render(), get dataset() { return datasetName; }, get transferFunction() { return transferFunction; } };
})();
