"""Offscreen VTK volume rendering, pixel grab, image features."""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from transfer import vector_to_vtk

WIDTH, HEIGHT = 1024, 800
_MAPPER_ANNOUNCED = False
MAPPER_NAME = None  # set on first render(); real value, not a guess -- read by server.py for the UI


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


def render(volume: np.ndarray, params: np.ndarray, spacing=(1.0, 1.0, 1.0), camera: dict | None = None) -> vtk.vtkRenderWindow:
    dx, dy, dz = volume.shape
    flat = np.ascontiguousarray(volume.ravel(order="F"))
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)

    image = vtk.vtkImageData()
    image.SetDimensions(dx, dy, dz)
    image.SetSpacing(*spacing)
    image.GetPointData().SetScalars(vtk_arr)

    ctf, otf = vector_to_vtk(params)
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()

    mapper = _make_mapper(image)
    # This render is one-shot and offscreen, not an interactive loop -- there's
    # no reason to let VTK trade sampling quality for frame rate the way it
    # does by default for interactive rendering.
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
