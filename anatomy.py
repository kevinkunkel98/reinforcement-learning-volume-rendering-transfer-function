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
CLASS_NAMES = CANONICAL_CLASSES
MEASURED_CLASSES = CANONICAL_CLASSES + ("other",)
LAYOUT_VERSION = "anatomy-v2"
CLASS_LAYOUT_VERSION = LAYOUT_VERSION
PROMOTED_ORGAN_CLASSES = ("liver", "kidneys", "spleen", "heart")
CANONICAL_PEAK_ORDER = (
    "lungs",
    "soft",
    "liver",
    "kidneys",
    "spleen",
    "heart",
    "vessels",
    "skeleton",
)

STRUCTURE_MAPPINGS = {
    "liver": "liver",
    "kidney_left": "kidneys",
    "kidney_right": "kidneys",
    "spleen": "spleen",
    "heart": "heart",
}

STRUCTURE_RULES = {
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
        "patella",
        "tibia",
        "fibula",
        "carpal",
        "metacarpal",
        "phalanges",
        "tarsal",
        "metatarsal",
    ),
    "lungs": ("lung_",),
    "heart": ("heart", "atrial_appendage"),
    "liver": ("liver",),
    "kidneys": ("kidney_",),
    "spleen": ("spleen",),
    "vessels": (
        "vessel_",
        "aorta",
        "brachiocephalic_",
        "common_carotid_",
        "subclavian_",
        "pulmonary_",
        "vena_cava",
        "portal_vein",
        "iliac_artery",
        "iliac_vena",
        "superior_vena_cava",
        "inferior_vena_cava",
    ),
    "soft": (
        "stomach",
        "pancreas",
        "gallbladder",
        "colon",
        "small_bowel",
        "duodenum",
        "esophagus",
        "urinary_bladder",
        "prostate",
        "adrenal_gland_",
        "thyroid_gland",
        "brain",
        "spinal_cord",
        "trachea",
        "autochthon_",
        "gluteus_",
        "iliopsoas_",
    ),
}

STRUCTURE_PREFIXES = STRUCTURE_RULES


def class_for_structure(structure: str) -> str | None:
    """Return canonical class for a TotalSegmentator structure name."""
    if structure in STRUCTURE_MAPPINGS:
        return STRUCTURE_MAPPINGS[structure]
    for anatomy_class, prefixes in STRUCTURE_RULES.items():
        if structure.startswith(prefixes):
            return anatomy_class
    return None
