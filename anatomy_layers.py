"""Versioned anatomical layer settings and validation."""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any, Mapping

import anatomy


LAYER_LAYOUT_VERSION = "anatomy-layers-v1"
DEFAULT_LAYERS = {
    "skeleton": {"opacity": 1.0, "rgb": [0.95, 0.95, 0.90]},
    "lungs": {"opacity": 1.0, "rgb": [0.55, 0.70, 0.95]},
    "heart": {"opacity": 1.0, "rgb": [0.85, 0.25, 0.25]},
    "vessels": {"opacity": 1.0, "rgb": [0.90, 0.45, 0.40]},
    "liver": {"opacity": 1.0, "rgb": [0.75, 0.45, 0.25]},
    "kidneys": {"opacity": 1.0, "rgb": [0.70, 0.40, 0.35]},
    "spleen": {"opacity": 1.0, "rgb": [0.55, 0.30, 0.45]},
    "soft": {"opacity": 1.0, "rgb": [0.85, 0.35, 0.35]},
}


def default_layers() -> dict[str, dict[str, Any]]:
    """Return an independent copy of canonical layer defaults."""
    return copy.deepcopy(DEFAULT_LAYERS)


def _bounded_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return value


def normalize_layers(
    value: Mapping[str, Any], available_classes: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Validate layer overrides and return all canonical classes with defaults."""
    if not isinstance(value, Mapping):
        raise ValueError("anatomy_layers must be an object")
    unknown = set(value) - set(anatomy.CANONICAL_CLASSES)
    if unknown:
        raise ValueError(f"anatomy_layers has unknown class: {', '.join(sorted(unknown))}")
    if available_classes is not None:
        available = set(available_classes)
        unknown_available = available - set(anatomy.CANONICAL_CLASSES)
        if unknown_available:
            raise ValueError(
                "available_classes has unknown class: "
                + ", ".join(sorted(unknown_available))
            )
        unsupported = set(value) - available
        if unsupported:
            raise ValueError(
                "anatomy_layers contains unsupported class: "
                + ", ".join(sorted(unsupported))
            )

    result = default_layers()
    for class_name, settings in value.items():
        if not isinstance(settings, Mapping):
            raise ValueError(f"anatomy_layers.{class_name} must be an object")
        unknown_settings = set(settings) - {"opacity", "rgb"}
        if unknown_settings:
            raise ValueError(
                f"anatomy_layers.{class_name} has unknown fields: "
                + ", ".join(sorted(unknown_settings))
            )
        layer = result[class_name]
        if "opacity" in settings:
            layer["opacity"] = _bounded_number(
                settings["opacity"], f"anatomy_layers.{class_name}.opacity"
            )
        if "rgb" in settings:
            rgb = settings["rgb"]
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                raise ValueError(f"anatomy_layers.{class_name}.rgb must contain 3 values")
            layer["rgb"] = [
                _bounded_number(component, f"anatomy_layers.{class_name}.rgb[{index}]")
                for index, component in enumerate(rgb)
            ]
    return result
