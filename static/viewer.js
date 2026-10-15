(() => {
  "use strict";

  const CENTER_RANGE = [-1050, 2000];
  const TOTAL_PARAMS = 48;
  // Mirrors render.LABEL_BAND: labelled HU is packed into one band per label
  // so a single 1D transfer function (rows from tf_table.js) can shade each
  // class by its own peak.
  const LABEL_BAND = 4096;
  // Headset performance levers (?tex ?crop ?down ?fov ?xrscale ?xrsample
  // ?adapt). Re-parsed with the headset defaults once load() knows this is a
  // VR-capable device; explicit URL flags win either way.
  let PERF = window.xrLogic.parsePerfFlags(window.location.search);
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
  let rawVolumeValues;
  let volumeMetadata;
  let anatomyAvailability;
  // True on a VR-capable (e.g. headset) browser, resolved once at load. Used
  // to pick a mobile-GPU-appropriate sampling policy -- the fixed, fine
  // sample distance below is tuned for a desktop GPU to match render.py's
  // reference screenshots, and is far too slow on a standalone headset's
  // mobile GPU, even for the flat (non-XR) page.
  let isXRCapableDevice = false;
  // Fixed-foveation level actually applied in the running XR session (or
  // "unsupported"); null outside XR.
  let xrFoveation = null;
  let floatLinear;
  // A browser without navigator.xr at all (e.g. desktop Safari/Firefox)
  // makes the optional-chained call below evaluate to undefined rather than
  // a promise -- fall back to one so `.then` is always safe to call.
  const xrCapabilityCheck = (navigator.xr?.isSessionSupported?.("immersive-vr") ?? Promise.resolve(false))
    .then((supported) => { isXRCapableDevice = supported; })
    .catch(() => {});

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
    builtFrom = { raw: null, labels: null };
    floatLinear = undefined;
    rawVolumeValues = undefined;
    volumeMetadata = undefined;
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

  function validateLabelDimensions(metadata, volumeMetadata) {
    if (metadata.dimensions.some((value, index) => value !== volumeMetadata.dimensions[index])) {
      throw new Error("label dimensions do not match volume dimensions");
    }
    return metadata;
  }

  function bandLabeledHuValues(values, labels, metadata, volumeMetadata) {
    // label-aware banding: each voxel's HU moves into its label's band, the
    // browser twin of render.label_scoped_volume.
    validateLabelDimensions(metadata, volumeMetadata);
    const banded = new Float32Array(values.length);
    for (let index = 0; index < labels.length; index += 1) {
      const hu = Math.max(CENTER_RANGE[0], Math.min(CENTER_RANGE[1], values[index]));
      banded[index] = hu - CENTER_RANGE[0] + labels[index] * LABEL_BAND;
    }
    return banded;
  }

  const CLASS_IDS = {
    skeleton: 1, lungs: 2, heart: 3, vessels: 4,
    liver: 5, kidneys: 6, spleen: 7, soft: 8,
  };
  const CLASS_ORDER = Object.keys(CLASS_IDS);

  function hasLabels() {
    return labelValues instanceof Uint8Array && labelMetadata !== undefined;
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

  function buildImageData(metadata, values, origin = [0, 0, 0]) {
    imageData = vtk.Common.DataModel.vtkImageData.newInstance();
    imageData.setDimensions(...metadata.dimensions);
    imageData.setSpacing(...metadata.spacing);
    imageData.setOrigin(...origin);
    imageData.getPointData().setScalars(vtk.Common.Core.vtkDataArray.newInstance({
      name: "Scalars", values, numberOfComponents: 1,
    }));
    return imageData;
  }

  // Identity of the inputs the current imageData was built from: rebanding
  // and re-uploading the whole volume takes seconds on a headset, and a
  // transfer-function change alone needs neither.
  let builtFrom = { raw: null, labels: null };

  // ?crop: anything at or below this HU is air outside the body.
  const AIR_HU = -500;

  function buildVolumeImage() {
    const logic = window.xrLogic;
    let values = hasLabels()
      ? bandLabeledHuValues(rawVolumeValues, labelValues, labelMetadata, volumeMetadata)
      : rawVolumeValues;
    let dims = volumeMetadata.dimensions;
    let origin = [0, 0, 0];
    if (PERF.crop) {
      // On raw HU, so a label band's offset can't make air look like tissue.
      const box = logic.bodyBoundingBox(rawVolumeValues, dims, AIR_HU);
      ({ values, dims } = logic.cropVolume(values, dims, box));
      // Shift the origin so world coordinates (camera, XR placement) match
      // the uncropped volume's.
      origin = box.min.map((index, axis) => index * volumeMetadata.spacing[axis]);
    }
    ({ values, dims } = logic.strideDownsample(values, dims, PERF.down));
    const spacing = volumeMetadata.spacing.map((step) => step * PERF.down);
    if (PERF.tex === "f16") values = logic.toSixteenBit(values);
    buildImageData({ dimensions: dims, spacing }, values, origin);
    mapper.setPreferSizeOverAccuracy(PERF.tex === "f16");
    mapper.setInputData(imageData);
    builtFrom = { raw: rawVolumeValues, labels: hasLabels() ? labelValues : null };
  }

  // Base XR ray-march step for the current image (finest spacing after any
  // downsampling), times the fixed ?xrsample multiplier.
  function xrBaseSampleDistance() {
    return window.xrLogic.xrSampleDistance(imageData.getSpacing()) * PERF.xrsample;
  }

  function setTransferFunction(params) {
    if (!Array.isArray(params) && !(params instanceof Float32Array) && !(params instanceof Float64Array)) {
      throw new Error("transfer function must be an array");
    }
    if (params.length !== TOTAL_PARAMS) throw new Error(`transfer function must contain ${TOTAL_PARAMS} values`);
    const color = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    const opacity = vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();
    // The same table the VR ray marcher uses (tf_table.js), laid out over
    // the banded scalar: row r covers [r * LABEL_BAND, r * LABEL_BAND + 3050].
    window.tfTable.tfRows(params, activeLayers, hasLabels()).forEach((row, rowIndex) => {
      for (const { hu, rgb, alpha } of row) {
        const x = hasLabels() ? rowIndex * LABEL_BAND + hu - CENTER_RANGE[0] : hu;
        color.addRGBPoint(x, ...rgb);
        opacity.addPoint(x, alpha);
      }
    });
    transferFunction = params.slice();
    volume.getProperty().setRGBTransferFunction(0, color);
    volume.getProperty().setScalarOpacity(0, opacity);
    tfListeners.forEach((listener) => listener(transferFunction, activeLayers, hasLabels()));
    // In XR, vtk.js's frame loop picks the new function up on its next frame;
    // a render() here would ray-march into the bound XR framebuffer for nothing.
    if (!inXR()) renderWindow.render();
  }

  function applyLabelInterpolation() {
    // Interpolating between two label bands lands in a third class's band.
    if (hasLabels()) volume.getProperty().setInterpolationTypeToNearest();
    else volume.getProperty().setInterpolationTypeToLinear();
  }

  function setLabelAwareTransferFunction(params, labels, metadata, layers = {}) {
    labelValues = labels;
    labelMetadata = metadata;
    activeLayers = layers || {};
    if (rawVolumeValues && volumeMetadata && mapper) {
      const labels = hasLabels() ? labelValues : null;
      if (builtFrom.raw !== rawVolumeValues || builtFrom.labels !== labels) buildVolumeImage();
      applyLabelInterpolation();
    }
    // HU-only transfer remains the fallback for datasets whose label
    // transport is unavailable.
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
    // In XR the headset pose drives the view, and resetCamera() (used for
    // legacy camera values) would also reset vtk.js's physical scale.
    if (inXR()) return;
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

  // Saved flat-viewer settings while an immersive session is running.
  let xrSaved = null;
  // The camera's physical frame right after XR placement; thumbstick
  // rotation turns it about the volume centre (see setVolumeRotation).
  let xrBaseFrame = null;
  // Bounding-sphere radius and distance of the volume in VR, in metres. A
  // ~39 cm-wide chest CT has a ~33 cm bounding radius, so this is about life-size.
  const XR_RADIUS_M = 0.3;
  const XR_DISTANCE_M = 0.9;
  // Per-eye render resolution relative to the headset's default is
  // PERF.xrscale (?xrscale=0.6 etc.).
  // Multiplies the XR ray-march sample distance (coarser = cheaper, fewer
  // samples per ray). AutoAdjustSampleDistances doesn't reliably kick in
  // inside the XR session (it's tuned against vtk's own interactor render
  // loop, not the session.requestAnimationFrame loop xr.js drives), so this
  // is a manual escape hatch: PERF.xrsample (?xrsample=2 etc.).

  function inXR() {
    return xrSaved !== null;
  }

  // Enters VR with a session the caller requested (it needs the click's user
  // activation, and requesting it there makes rejection catchable -- vtk.js's
  // own startXR() swallows it inside a promise).
  async function enterXR(session) {
    if (!openGLRenderWindow || !mapper || !volumeMetadata) throw new Error("no local volume loaded");
    xrSaved = {
      style: interactor.getInteractorStyle(),
      sampleDistance: mapper.getSampleDistance(),
      autoAdjust: mapper.getAutoAdjustSampleDistances(),
      imageSampleDistance: mapper.getImageSampleDistance(),
      shade: volume.getProperty().getShade(),
      camera: fromRendererCamera(),
      background: renderer.getBackground().slice(),
    };
    // vtk.js's trackball style binds the right trigger/trackpad to a camera
    // "fly" in XR, which would fight push-to-talk; the base style has no 3D
    // button handlers at all.
    interactor.setInteractorStyle(vtk.Rendering.Core.vtkInteractorStyle.newInstance());
    mapper.setSampleDistance(xrBaseSampleDistance());
    // vtk.js's auto-adjust measures its own interactor loop, which doesn't
    // run in XR (it reads ~2 fps), so it pins the volume to its smallest
    // offscreen size -- 1/20 resolution per axis -- and that offscreen path
    // sizes itself from the canvas, not the XR eye viewports: one eye loses
    // the volume, the other gets a black crop. In XR the WebXR layer's
    // framebuffer scale sets the resolution instead. The scale is only
    // recomputed on vtk's frame-rate event, so fire it once by hand.
    mapper.setAutoAdjustSampleDistances(false);
    mapper.setImageSampleDistance(1);
    interactor.invokeAnimationFrameRateUpdate();
    // Keep the orientation the user was looking at in the flat viewer. The
    // stored view-up needn't be perpendicular to the view direction (a saved
    // camera can look almost straight along it), and a near-parallel
    // up/north pair makes vtk's physical frame degenerate -- the volume then
    // lands tilted and off to the side. The flat camera is restored from
    // xrSaved on exit, so orthogonalizing it here is safe.
    camera.orthogonalizeViewUp();
    camera.setPhysicalViewUp(...camera.getViewUp());
    camera.setPhysicalViewNorth(...camera.getDirectionOfProjection());
    try {
      // vtk.js requests the reference space without awaiting it, and its
      // frame loop calls frame.getPose(grip, space) before re-arming -- a
      // null (first entry) or stale (re-entry) space throws there and stops
      // rendering for good. Resolve it up front; set() with noWarning writes
      // the model field, which has no setter.
      const space = await session.requestReferenceSpace("local");
      openGLRenderWindow.set({ xrReferenceSpace: space }, true);
      // getContext() on the same canvas returns the existing (unproxied) context.
      const rawContext = openGLRenderWindow.getCanvas()
        .getContext(openGLRenderWindow.getWebgl2() ? "webgl2" : "webgl");
      await window.xrLogic.withXRLayerContext(window, rawContext,
        () => openGLRenderWindow.enterXR(session),
        { framebufferScaleFactor: PERF.xrscale });
      // Replace vtk's resetXRScene() placement (viewer inside a 1 m-radius
      // volume): life-size-ish, in front along the direction the flat view faced.
      const place = window.xrLogic.xrPlacement(volume.getBounds(), camera.getPhysicalViewNorth(),
        XR_RADIUS_M, XR_DISTANCE_M);
      camera.setPhysicalScale(place.scale);
      camera.setPhysicalTranslation(...place.translation);
      camera.setClippingRange(0.1 * place.scale, 100 * place.scale);
      xrBaseFrame = {
        north: camera.getPhysicalViewNorth().slice(),
        up: camera.getPhysicalViewUp().slice(),
        translation: camera.getPhysicalTranslation().slice(),
      };
    } catch (error) {
      await exitXR();
      throw error;
    }
  }

  async function exitXR() {
    if (!xrSaved) return;
    const saved = xrSaved;
    try {
      await openGLRenderWindow.stopXR();
    } finally {
      // Restore the flat viewer even if stopXR() fails (e.g. context lost),
      // or inXR() would suppress flat rendering until a reload.
      openGLRenderWindow.set({ xrReferenceSpace: null }, true);
      xrSaved = null;
      xrBaseFrame = null;
      interactor.setInteractorStyle(saved.style);
      mapper.setSampleDistance(saved.sampleDistance);
      mapper.setAutoAdjustSampleDistances(saved.autoAdjust);
      mapper.setImageSampleDistance(saved.imageSampleDistance);
      interactor.invokeAnimationFrameRateUpdate();
      volume.getProperty().setShade(saved.shade);
      renderer.setBackground(...saved.background);
      camera.setPosition(...saved.camera.position);
      camera.setFocalPoint(...saved.camera.focal_point);
      camera.setViewUp(...saved.camera.view_up);
      renderer.resetCameraClippingRange();
      renderWindow.render();
    }
  }

  async function load(name, params, cameraState, layers = {}, availability = null) {
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
        anatomyAvailability = availability;
        setLabelAwareTransferFunction(params, labelValues, labelMetadata, layers);
        fallbackEl.hidden = true;
        setStatus(`Local ${datasetName} volume${labelStatus}`);
        return true;
      }
      labelStatus = "";
      labelValues = undefined;
      labelMetadata = undefined;
      activeLayers = {};
      anatomyAvailability = availability;
      const loaded = await fetchVolume(name, loadController.signal, generation);
      if (!isCurrentLoad(generation)) return false;
      // Labels are transport-ready now; rendering remains HU-only until Task 3.
      // A missing label volume must never disable the existing volume path.
      const knownUnlabeled = availability && availability.label_available === false;
      if (!knownUnlabeled) {
        try {
          const labels = await fetchLabelMetadata(name, loadController.signal);
          labelMetadata = validateLabelDimensions(labels.metadata, loaded.metadata);
          labelValues = labels.labels;
          labelStatus = "";
        } catch (labelError) {
          if (labelError.name === "AbortError") throw labelError;
          if (!String(labelError.message).includes("label-unavailable")) {
            labelStatus = `; ${labelError.message}`;
          }
        }
      }
      await xrCapabilityCheck;
      PERF = window.xrLogic.parsePerfFlags(window.location.search,
        isXRCapableDevice ? window.xrLogic.HEADSET_PERF_DEFAULTS : window.xrLogic.PERF_DEFAULTS);
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
        if (isXRCapableDevice) {
          // vtk.js's auto-adjust logic (enabled below) coarsens sampling to
          // hit this interactor-level target. The defaults (~15 fps moving,
          // near-zero i.e. "take as long as you like" once still) assume a
          // desktop GPU with a human occasionally letting go of the mouse --
          // a headset is always "moving" (head tracking) and needs a real
          // frame-rate target the whole time, both on the flat page and in VR.
          interactor.setDesiredUpdateRate(60);
          interactor.setStillUpdateRate(60);
        }
      }
      rawVolumeValues = loaded.values;
      volumeMetadata = loaded.metadata;
      mapper = vtk.Rendering.Core.vtkVolumeMapper.newInstance();
      buildVolumeImage();
      // Match render.py's server-side quality settings -- vtk.js's defaults
      // (no shading, auto-adjusted/coarser sampling) are what made this
      // viewer look flat and "hologram"-like next to the reference PNGs.
      // On a VR-capable device's mobile GPU, that fixed fine sampling is
      // far too slow (observed 2-5 fps, even on this flat page) -- auto-
      // adjust lets vtk.js coarsen it dynamically to hit the update rate
      // set above instead, falling back toward this same quality target
      // whenever the GPU can sustain it.
      mapper.setAutoAdjustSampleDistances(isXRCapableDevice);
      mapper.setSampleDistance(Math.min(...imageData.getSpacing()) / 2.0);
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

  // The 3D texture vtk.js 25.11 builds from our scalars under WebGL2:
  // Float32 is uploaded as R32F; 16-bit integers become R16F (half float)
  // with preferSizeOverAccuracy on (Rendering/OpenGL/Texture.js,
  // create3DFilterableFromRaw). vtk.js doesn't expose the texture itself.
  function textureFormat() {
    const type = imageData?.getPointData().getScalars()?.getDataType();
    if (!type) return null;
    const sixteenBit = type === "Uint16Array" || type === "Int16Array";
    return sixteenBit && mapper.getPreferSizeOverAccuracy() ? "R16F" : "R32F";
  }

  // Whether this GPU can linearly filter float textures natively.
  function floatLinearSupported() {
    if (floatLinear === undefined && openGLRenderWindow) {
      const gl = openGLRenderWindow.getCanvas()
        .getContext(openGLRenderWindow.getWebgl2() ? "webgl2" : "webgl");
      floatLinear = Boolean(gl?.getExtension("OES_texture_float_linear"));
    }
    return floatLinear;
  }

  // Called by xr.js's adaptive controller (?adapt=1) when its multiplier moves.
  function setXRSampleMultiplier(multiplier) {
    if (inXR()) mapper.setSampleDistance(xrBaseSampleDistance() * multiplier);
  }

  // Thumbstick rotation (xr.js). The world -> physical rotation of the XR
  // placement the turns are measured against, and the turn itself: vtk.js 25
  // mis-clips a volume whose prop matrix is rotated, so the room frame turns
  // about the volume centre instead (xrLogic.rotatedPhysicalFrame), and the
  // HUD gets the inverse turn so it stays put in the room.
  function getPhysicalRotation() {
    return window.xrLogic.physicalRotation(xrBaseFrame.north, xrBaseFrame.up);
  }

  function setVolumeRotation(rotation) {
    if (!xrBaseFrame) return;
    const logic = window.xrLogic;
    const bounds = volume.getBounds();
    const center = [0, 2, 4].map((i) => (bounds[i] + bounds[i + 1]) / 2);
    const frame = logic.rotatedPhysicalFrame(xrBaseFrame, rotation, center);
    camera.setPhysicalViewNorth(...frame.north);
    camera.setPhysicalViewUp(...frame.up);
    camera.setPhysicalTranslation(...frame.translation);
    const inverse = [0, 1, 2].map((c) => [rotation[0][c], rotation[1][c], rotation[2][c]]);
    const hudMatrix = logic.centeredUserMatrix(inverse, center);
    renderer.getActors().forEach((actor) => actor.setUserMatrix(hudMatrix));
  }

  // For the custom VR ray marcher (vr_renderer.js): the raw volume, labels
  // and current TF state, plus the flat view's lighting and orientation so
  // VR looks and faces the same way.
  const tfListeners = new Set();

  function getVolumeData() {
    if (!rawVolumeValues || !volumeMetadata || !volume) return null;
    const property = volume.getProperty();
    camera.orthogonalizeViewUp();
    return {
      hu: rawVolumeValues,
      labels: hasLabels() ? labelValues : null,
      dims: volumeMetadata.dimensions.slice(),
      spacing: volumeMetadata.spacing.slice(),
      params: transferFunction.slice(),
      layers: { ...activeLayers },
      unitDistance: property.getScalarOpacityUnitDistance(0),
      light: {
        ambient: property.getAmbient(), diffuse: property.getDiffuse(),
        specular: property.getSpecular(), specularPower: property.getSpecularPower(),
      },
      north: camera.getDirectionOfProjection().slice(),
      up: camera.getViewUp().slice(),
    };
  }

  function onTransferFunctionChange(listener) {
    tfListeners.add(listener);
    return () => tfListeners.delete(listener);
  }

  window.volumeViewer = {
    load, setTransferFunction, setLabelAwareTransferFunction, getCamera, setCamera,
    // The XR frame loop renders on its own; a flat render() mid-session would
    // draw into the XR-sized canvas for nothing.
    render: () => { if (!inXR()) renderWindow?.render(); },
    enterXR, exitXR, inXR,
    get perfFlags() { return PERF; },
    setXRSampleMultiplier,
    getVolumeData, onTransferFunctionChange,
    getPhysicalRotation, setVolumeRotation,
    getRenderer: () => renderer,
    getVolumeBounds: () => volume?.getBounds(),
    // In XR the camera's own axes follow the head every frame; the physical
    // up/north are the fixed room axes the volume was placed along.
    getCameraAxes: () => {
      if (!camera) return null;
      return inXR()
        ? { up: camera.getPhysicalViewUp().slice(), dop: camera.getPhysicalViewNorth().slice() }
        : { up: camera.getViewUp().slice(), dop: camera.getDirectionOfProjection().slice() };
    },
    get dataset() { return datasetName; },
    get transferFunction() { return transferFunction; },
    // Surfaced on the in-VR HUD (xr.js) so a headset test reports real
    // numbers on its own screen instead of a subjective "laggy".
    getXRDiagnostics: () => ({
      xrCapable: isXRCapableDevice,
      sampleDistance: mapper?.getSampleDistance(),
      autoAdjust: mapper?.getAutoAdjustSampleDistances(),
      shade: volume?.getProperty().getShade(),
      dimensions: imageData?.getDimensions(),
      texture: textureFormat(),
      floatLinear: floatLinearSupported(),
      foveation: xrFoveation,
      framebufferScale: inXR() ? PERF.xrscale : null,
      flags: PERF,
    }),
  };
})();
