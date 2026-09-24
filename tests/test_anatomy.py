import anatomy


def test_registry_exposes_canonical_classes_and_layout_version():
    assert anatomy.CANONICAL_CLASSES == (
        "skeleton",
        "lungs",
        "heart",
        "vessels",
        "liver",
        "kidneys",
        "spleen",
        "soft",
    )
    assert anatomy.LAYOUT_VERSION == "anatomy-v2"


def test_structure_class_maps_exact_and_prefixed_structures():
    assert anatomy.class_for_structure("rib_1") == "skeleton"
    assert anatomy.class_for_structure("vertebrae_L1") == "skeleton"
    assert anatomy.class_for_structure("sacrum") == "skeleton"
    assert anatomy.class_for_structure("lung_upper_lobe_left") == "lungs"
    assert anatomy.class_for_structure("vessel_aorta") == "vessels"
    assert anatomy.class_for_structure("aorta") == "vessels"
    assert anatomy.class_for_structure("liver") == "liver"
    assert anatomy.class_for_structure("kidney_left") == "kidneys"
    assert anatomy.class_for_structure("kidney_right") == "kidneys"
    assert anatomy.class_for_structure("heart") == "heart"
    assert anatomy.class_for_structure("pancreas") == "soft"


def test_unknown_structure_has_no_class():
    assert anatomy.class_for_structure("not_a_structure") is None
