"""Canonical anatomy classes and structure-to-class registry."""

CANONICAL_CLASSES = (
    "skeleton",
    "lungs",
    "heart",
    "vessels",
    "liver",
    "kidneys",
    "spleen",
    "soft",
)
LAYOUT_VERSION = "anatomy-v2"

STRUCTURE_MAPPINGS = {
    "liver": "liver",
    "kidney_left": "kidneys",
    "kidney_right": "kidneys",
    "heart": "heart",
    "pancreas": "soft",
}

STRUCTURE_PREFIXES = {
    "skeleton": (
        "rib_",
        "vertebrae_",
        "hip_",
        "femur_",
        "humerus_",
        "scapula_",
        "clavicula_",
        "sacrum",
        "sternum",
        "skull",
        "costal_cartilages",
    ),
    "lungs": ("lung_",),
    "vessels": (
        "vessel_",
        "aorta",
        "atrial_appendage_",
        "brachiocephalic_",
        "common_carotid_",
        "subclavian_",
        "pulmonary_",
        "vena_cava",
        "portal_vein",
        "iliac_artery",
        "iliac_vena",
    ),
}


def class_for_structure(structure: str) -> str | None:
    """Return canonical class for a TotalSegmentator structure name."""
    if structure in STRUCTURE_MAPPINGS:
        return STRUCTURE_MAPPINGS[structure]
    for anatomy_class, prefixes in STRUCTURE_PREFIXES.items():
        if structure.startswith(prefixes):
            return anatomy_class
    return None
