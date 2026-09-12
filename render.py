"""Offscreen VTK volume rendering, pixel grab, image features."""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from transfer import N_PEAKS, peak_internal, vector_to_vtk

WIDTH, HEIGHT = 1024, 800
_MAPPER_ANNOUNCED = False
MAPPER_NAME = None  # set on first render(); real value, not a guess -- read by server.py for the UI

VISIBLE_BOUNDS_STRIDE = 4  # measured: ~10ms strided vs ~1-2s at full resolution on real CT
VISIBLE_BOUNDS_THRESHOLD = 0.05
VISIBLE_BOUNDS_MARGIN = 0.15  # fraction of extent added as padding so content isn't cropped tight


def _visible_bounds(volume: np.ndarray, params: np.ndarray, spacing) -> list | None:
    """Physical-space bounds [xmin, xmax, ymin, ymax, zmin, zmax] of voxels
    whose composited opacity exceeds VISIBLE_BOUNDS_THRESHOLD, so the camera
    can fit to what's actually visible instead of the full volume extent
    (mostly transparent "air" padding, especially once a command isolates
    one small tissue). Returns None if nothing is visible, so the caller can
    fall back to the full-volume framing. Downsampled by
    VISIBLE_BOUNDS_STRIDE per axis -- see the module constant's comment for
    why full resolution isn't viable here."""
    sub = volume[::VISIBLE_BOUNDS_STRIDE, ::VISIBLE_BOUNDS_STRIDE, ::VISIBLE_BOUNDS_STRIDE]
    opacity = np.zeros(sub.shape, dtype=np.float64)
    for i in range(N_PEAKS):
        p = peak_internal(params, i)
        gauss = np.exp(-0.5 * ((sub - p["center"]) / p["width"]) ** 2)
        opacity += p["height"] * gauss
    mask = np.clip(opacity, 0.0, 1.0) > VISIBLE_BOUNDS_THRESHOLD
    idx = np.nonzero(mask)
    if idx[0].size == 0:
        return None

    bounds = []
    for axis in range(3):
        lo = idx[axis].min() * VISIBLE_BOUNDS_STRIDE
        hi = idx[axis].max() * VISIBLE_BOUNDS_STRIDE
        lo_phys, hi_phys = lo * spacing[axis], hi * spacing[axis]
        pad = max((hi_phys - lo_phys) * VISIBLE_BOUNDS_MARGIN, spacing[axis] * VISIBLE_BOUNDS_STRIDE)
        bounds.extend([lo_phys - pad, hi_phys + pad])
    return bounds


def _make_mapper(vtk_image):
    global _MAPPER_ANNOUNCED, MAPPER_NAME
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(vtk_image)
    try:
        ok = bool(mapper.IsRenderSupported(vtk.vtkRenderWindow(), None))
    except Exception:
        ok = False
    if not ok:
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        mapper.SetInputData(vtk_image)
        name = "vtkFixedPointVolumeRayCastMapper (CPU fallback)"
        MAPPER_NAME = "CPU"
    else:
        name = "vtkGPUVolumeRayCastMapper"
        MAPPER_NAME = "GPU"
    if not _MAPPER_ANNOUNCED:
        print(f"[render] using {name}")
        _MAPPER_ANNOUNCED = True
    return mapper


_PIPELINE_CACHE = None  # (key, prop, renderer, win) -- see _get_pipeline


def _get_pipeline(volume: np.ndarray, spacing):
    """Build (or reuse) the offscreen render pipeline for `volume`.

    Every vtkRenderWindow.Render() call re-binds a native offscreen GL
    surface via a macOS WindowServer/IOKit round trip (vtkCocoaRenderWindow
    ::Start() -> -[NSOpenGLContext update] -> IOAccelCreateSurface). That
    round trip is fine occasionally, but a process that creates a *new*
    vtkRenderWindow for every render (the old behavior here) issues a fresh
    surface-bind request every single call -- and after roughly a minute of
    sustained requests from one non-windowed process, IOAccelCreateSurface
    stops responding and the next Render() blocks forever (confirmed via
    `sample` on the hung process: stuck in mach_msg2_trap inside
    IOServiceOpen). Reusing one window/renderer/mapper per volume avoids
    creating new surfaces at all after the first render, which also makes
    each subsequent render ~150x faster (no context-creation overhead).

    Keyed on `id(volume)`: every hot-loop caller in this codebase holds one
    volume array alive for the caller's entire lifetime, so identity is a
    safe, cheap cache key here -- it is not a general-purpose memoization.
    """
    global _PIPELINE_CACHE
    key = (id(volume), tuple(spacing))
    if _PIPELINE_CACHE is not None and _PIPELINE_CACHE[0] == key:
        return _PIPELINE_CACHE[1:]
    if _PIPELINE_CACHE is not None:
        _PIPELINE_CACHE[3].Finalize()

    dx, dy, dz = volume.shape
    flat = np.ascontiguousarray(volume.ravel(order="F"))
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)

    image = vtk.vtkImageData()
    image.SetDimensions(dx, dy, dz)
    image.SetSpacing(*spacing)
    image.GetPointData().SetScalars(vtk_arr)

    # The grainy look in real CT renders is the scan's own HU sensor noise,
    # passed straight through the transfer function -- not a sampling
    # artifact, so no amount of ray-sampling quality fixes it. A light
    # Gaussian smooth of the volume itself does: measured negligible cost on
    # the synthetic phantom (96^3, ~2ms) and ~115ms on real CT (512x512x139).
    # Done once per volume here, not per render, since it doesn't depend on
    # the transfer function.
    smoother = vtk.vtkImageGaussianSmooth()
    smoother.SetInputData(image)
    smoother.SetStandardDeviations(1.0, 1.0, 1.0)
    smoother.SetRadiusFactors(2.0, 2.0, 2.0)
    smoother.Update()
    image = smoother.GetOutput()

    prop = vtk.vtkVolumeProperty()
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()

    mapper = _make_mapper(image)
    # This render is offscreen batch rendering, not an interactive loop --
    # there's no reason to let VTK trade sampling quality for frame rate the
    # way it does by default for interactive rendering.
    mapper.AutoAdjustSampleDistancesOff()
    mapper.SetSampleDistance(min(spacing) / 2.0)
    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(mapper)
    volume_actor.SetProperty(prop)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume_actor)
    renderer.SetBackground(0.0, 0.0, 0.0)

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(renderer)
    win.SetSize(WIDTH, HEIGHT)

    _PIPELINE_CACHE = (key, prop, renderer, win)
    return prop, renderer, win


def render(volume: np.ndarray, params: np.ndarray, spacing=(1.0, 1.0, 1.0), camera: dict | None = None) -> vtk.vtkRenderWindow:
    prop, renderer, win = _get_pipeline(volume, spacing)

    ctf, otf = vector_to_vtk(params)
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)

    bounds = _visible_bounds(volume, params, spacing)
    if bounds is not None:
        renderer.ResetCamera(bounds)
    else:
        renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam_state = camera or {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    cam.Azimuth(cam_state["azimuth"])
    cam.Elevation(cam_state["elevation"])
    cam.Zoom(cam_state["zoom"])
    renderer.ResetCameraClippingRange()

    win.Render()
    return win


def grab(win: vtk.vtkRenderWindow) -> np.ndarray:
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    vtk_image = w2i.GetOutput()
    w, h, _ = vtk_image.GetDimensions()
    arr = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    arr = arr.reshape(h, w, 3).astype(np.uint8)
    return arr[::-1]  # VTK draws bottom-up


def features(rgb: np.ndarray) -> dict:
    gray = rgb.astype(np.float64).mean(axis=2)
    coverage = float((gray > 2.0).mean())
    hist, _ = np.histogram(gray, bins=32, range=(0, 255), density=False)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "coverage": coverage,
        "entropy": entropy,
    }
