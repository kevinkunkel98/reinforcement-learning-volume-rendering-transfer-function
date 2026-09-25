(() => {
  "use strict";

  const CENTER_RANGE = [-1050, 2000];
  const WIDTH_RANGE = [10, 400];
  const TOTAL_PARAMS = 48;
  const PARAMS_PER_PEAK = 6;
  const N_PEAKS = TOTAL_PARAMS / PARAMS_PER_PEAK;
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
  let labelStatus = "";
  let labelValues;
  let labelMetadata;
  let activeLayers = {};

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
    lastRequestedCamera = null;
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

  function validateLabelMetadata(metadata) {
    if (!metadata || !Array.isArray(metadata.dimensions) || metadata.dimensions.length !== 3 ||
        metadata.dimensions.some((value) => !Number.isInteger(value) || value <= 0)) {
      throw new Error("invalid label dimensions");
    }
    const expectedBytes = metadata.dimensions.reduce((product, value) => product * value, 1);
    if (metadata.scalar_type !== "uint8" || metadata.order !== "F" ||
        metadata.label_layout_version !== "anatomy-v2" || metadata.total_bytes !== expectedBytes) {
      throw new Error("unsupported label transport format");
    }
    if (!Number.isInteger(metadata.chunk_count) || metadata.chunk_count <= 0 ||
        !Array.isArray(metadata.chunks) || metadata.chunks.length !== metadata.chunk_count) {
      throw new Error("labels have no chunks");
    }
    let offset = 0;
    metadata.chunks.forEach((chunk, index) => {
      if (chunk.index !== index || chunk.byte_offset !== offset ||
          !Number.isInteger(chunk.byte_length) || chunk.byte_length <= 0 ||
          chunk.byte_offset < 0 || chunk.byte_offset + chunk.byte_length > metadata.total_bytes) {
        throw new Error("invalid label chunk descriptors");
      }
      offset += chunk.byte_length;
    });
    if (offset !== metadata.total_bytes) throw new Error("label chunks do not cover payload");
    return metadata;
  }

  function reconstructLabels(metadata, chunks) {
    if (chunks.length !== metadata.chunk_count) throw new Error("label chunk count mismatch");
    const raw = new Uint8Array(metadata.total_bytes);
    chunks.forEach(({ index, byte_offset: offset, byte_length: length, bytes }) => {
      const expected = metadata.chunks[index];
      if (!expected || expected.byte_offset !== offset || expected.byte_length !== length ||
          bytes.byteLength !== length || offset < 0 || offset + length > raw.byteLength) {
        throw new Error("invalid label chunk payload");
      }
      raw.set(bytes, offset);
    });
    return new Uint8Array(raw.buffer);
  }

  async function fetchLabelMetadata(name, signal) {
    const response = await fetch(`/api/datasets/${encodeURIComponent(name)}/labels/metadata`, { signal });
    if (response.status === 404) throw new Error("no anatomical labels (label-unavailable)");
    if (!response.ok) throw new Error(`label metadata request failed (${response.status})`);
    const metadata = validateLabelMetadata(await response.json());
    if (metadata.name !== name) throw new Error("label metadata dataset mismatch");
    const chunks = await Promise.all(metadata.chunks.map(async (chunk) => {
      const chunkResponse = await fetch(`/api/datasets/${encodeURIComponent(name)}/labels/chunks/${chunk.index}`, { signal });
      if (!chunkResponse.ok) throw new Error(`label chunk ${chunk.index} request failed (${chunkResponse.status})`);
      if (chunkResponse.headers.get("X-Dataset-Version") !== metadata.dataset_version) {
        throw new Error(`label chunk ${chunk.index} has mismatched dataset version`);
      }
      const bytes = new Uint8Array(await chunkResponse.arrayBuffer());
      if (bytes.byteLength !== chunk.byte_length) throw new Error(`label chunk ${chunk.index} has invalid length`);
      return { ...chunk, bytes };
    }));
    return { metadata, labels: reconstructLabels(metadata, chunks) };
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
    // metadata.byte_order is validated "little" above, and every JS engine
    // that runs a browser is little-endian natively, so reinterpreting the
    // buffer directly is equivalent to (and vastly faster than) reading it
    // one float at a time through a DataView -- the latter took ~15s+ for
    // a 145MB real CT volume (ct_chest), long enough that switching
    // datasets looked broken: the previous dataset's stale canvas frame
    // stayed visible the whole time this loop ran.
    const values = new Float32Array(raw.buffer);
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

  function setTransferFunction(params) {
    if (!Array.isArray(params) && !(params instanceof Float32Array) && !(params instanceof Float64Array)) {
      throw new Error("transfer function must be an array");
    }
    if (params.length !== TOTAL_PARAMS) throw new Error(`transfer function must contain ${TOTAL_PARAMS} values`);
    const color = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    const opacity = vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();
    for (let i = 0; i < 256; i += 1) {
      const hu = CENTER_RANGE[0] + i / 255 * (CENTER_RANGE[1] - CENTER_RANGE[0]);
      let alpha = 0;
      const rgb = [0, 0, 0];
      let weight = 1e-6;
      for (let peakIndex = 0; peakIndex < N_PEAKS; peakIndex += 1) {
        const peak = internalPeak(params, peakIndex);
        const className = ["lungs", "soft", "liver", "kidneys", "spleen", "heart", "vessels", "skeleton"][peakIndex];
        const layerOpacity = activeLayers[className]?.opacity ?? 1;
        const contribution = peak.height * layerOpacity * Math.exp(-0.5 * ((hu - peak.center) / peak.width) ** 2);
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

  function setLabelAwareTransferFunction(params, labels, metadata, layers = {}) {
    labelValues = labels;
    labelMetadata = metadata;
    activeLayers = layers || {};
    // Labels are retained for the label-aware path; HU transfer remains the
    // fallback for datasets whose label transport is unavailable.
    setTransferFunction(params);
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

  // The camera the *server* last asked for. A step whose camera is unchanged
  // (every transfer-function command -- only camera commands and dataset
  // switches change it) must leave the camera exactly where it is, including
  // wherever the user has dragged it to: re-seating it would re-frame the
  // picture that the user is trying to compare against the previous step.
  let lastRequestedCamera = null;

  function sameCameraRequest(value) {
    try {
      return lastRequestedCamera !== null
        && JSON.stringify(value) === JSON.stringify(lastRequestedCamera);
    } catch (err) {
      return false;
    }
  }

  function setCamera(value) {
    if (!camera || !value) return;
    if (sameCameraRequest(value)) return;
    lastRequestedCamera = JSON.parse(JSON.stringify(value));
    appliedCameraState = toRendererCamera(value);
    renderer.resetCamera();
    cameraBaseScale = camera.getParallelScale();
    if (appliedCameraState.position) camera.setPosition(...appliedCameraState.position);
    if (appliedCameraState.focal_point) camera.setFocalPoint(...appliedCameraState.focal_point);
    if (appliedCameraState.view_up) camera.setViewUp(...appliedCameraState.view_up);
    if (appliedCameraState.zoom) camera.setParallelScale(cameraBaseScale / appliedCameraState.zoom);
    renderWindow.render();
  }

  async function load(name, params, cameraState, layers = {}) {
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
        activeLayers = layers || {};
        setLabelAwareTransferFunction(params, labelValues, labelMetadata, layers);
        fallbackEl.hidden = true;
        setStatus(`Local ${datasetName} volume${labelStatus}`);
        return true;
      }
      labelStatus = "";
      const loaded = await fetchVolume(name, loadController.signal, generation);
      if (!isCurrentLoad(generation)) return false;
      // Labels are transport-ready now; rendering remains HU-only until Task 3.
      // A missing label volume must never disable the existing volume path.
      try {
        const labels = await fetchLabelMetadata(name, loadController.signal);
        labelValues = labels.labels;
        labelMetadata = labels.metadata;
        labelStatus = "";
      } catch (labelError) {
        if (labelError.name === "AbortError") throw labelError;
        labelStatus = `; ${labelError.message}`;
      }
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
        // Without an explicit style, the interactor captures mouse/touch
        // events (bindEvents below) but never turns them into camera
        // movement -- drag/scroll are silently no-ops otherwise.
        interactor.setInteractorStyle(vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance());
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
      // Match render.py's server-side quality settings -- vtk.js's defaults
      // (no shading, auto-adjusted/coarser sampling) are what made this
      // viewer look flat and "hologram"-like next to the reference PNGs.
      mapper.setAutoAdjustSampleDistances(false);
      mapper.setSampleDistance(Math.min(...loaded.metadata.spacing) / 2.0);
      volume = vtk.Rendering.Core.vtkVolume.newInstance();
      volume.setMapper(mapper);
      volume.getProperty().setShade(true);
      volume.getProperty().setInterpolationTypeToLinear();
      volume.getProperty().setAmbient(0.1);
      volume.getProperty().setDiffuse(0.7);
      volume.getProperty().setSpecular(0.2);
      renderer.removeAllVolumes();
      renderer.addVolume(volume);
      renderer.resetCamera();
      setCamera(cameraState);
      setLabelAwareTransferFunction(params, labelValues, labelMetadata, layers);
      fallbackEl.hidden = true;
      setStatus(`Local ${datasetName} volume${labelStatus}`);
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

  window.volumeViewer = { load, setTransferFunction, setLabelAwareTransferFunction, getCamera, setCamera, render: () => renderWindow?.render(), get dataset() { return datasetName; }, get transferFunction() { return transferFunction; } };
})();
