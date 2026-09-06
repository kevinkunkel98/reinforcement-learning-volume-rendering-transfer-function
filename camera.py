"""Camera state (azimuth/elevation/zoom) and relative camera commands."""
import numpy as np

DEFAULT_CAMERA = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}  # copy before use, e.g. dict(DEFAULT_CAMERA) -- never hold a live reference to this dict

ELEVATION_RANGE = (-85.0, 85.0)
ZOOM_RANGE = (0.3, 4.0)

ROTATE_STEP_DEGREES = {"slightly": 15.0, "moderately": 30.0, "strongly": 60.0}
ZOOM_STEP_FACTOR = {"slightly": 1.15, "moderately": 1.35, "strongly": 1.7}


def apply_camera_command(cam_cmd: dict, camera: dict) -> dict:
    """Apply one relative camera adjustment, returning a new camera dict.

    cam_cmd: {"action": "rotate"|"tilt"|"zoom",
              "direction": "left"|"right" (rotate) | "up"|"down" (tilt) | "in"|"out" (zoom),
              "strength": "slightly"|"moderately"|"strongly"}

    Azimuth wraps (mod 360) since it has no natural bound. Elevation and
    zoom clamp instead, since both have real physical limits -- clamping
    elevation specifically avoids the camera flipping through the poles.
    Does not mutate the input `camera` dict.
    """
    camera = dict(camera)
    action = cam_cmd["action"]
    direction = cam_cmd["direction"]
    strength = cam_cmd.get("strength") or "moderately"

    if action == "rotate":
        delta = ROTATE_STEP_DEGREES[strength]
        sign = 1.0 if direction == "right" else -1.0
        camera["azimuth"] = (camera["azimuth"] + sign * delta) % 360.0
    elif action == "tilt":
        delta = ROTATE_STEP_DEGREES[strength]
        sign = 1.0 if direction == "up" else -1.0
        camera["elevation"] = float(np.clip(camera["elevation"] + sign * delta, *ELEVATION_RANGE))
    elif action == "zoom":
        factor = ZOOM_STEP_FACTOR[strength]
        if direction == "in":
            camera["zoom"] = float(np.clip(camera["zoom"] * factor, *ZOOM_RANGE))
        else:
            camera["zoom"] = float(np.clip(camera["zoom"] / factor, *ZOOM_RANGE))
    return camera
