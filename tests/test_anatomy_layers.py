import copy

import pytest

import anatomy_layers


def test_default_layers_returns_the_eight_canonical_classes_and_defaults():
    layers = anatomy_layers.default_layers()

    assert tuple(layers) == (
        "skeleton",
        "lungs",
        "heart",
        "vessels",
        "liver",
        "kidneys",
        "spleen",
        "soft",
    )
    assert layers["skeleton"] == {"opacity": 1.0, "rgb": [0.95, 0.95, 0.90]}
    assert layers["soft"] == {"opacity": 1.0, "rgb": [0.85, 0.35, 0.35]}


def test_default_layers_is_independent():
    layers = anatomy_layers.default_layers()
    layers["lungs"]["rgb"][0] = 0.0

    assert anatomy_layers.default_layers()["lungs"]["rgb"][0] == 0.55


def test_normalize_layers_fills_omitted_classes_from_defaults():
    layers = anatomy_layers.normalize_layers({"liver": {"opacity": 0.25}})

    assert tuple(layers) == tuple(anatomy_layers.default_layers())
    assert layers["liver"] == {"opacity": 0.25, "rgb": [0.75, 0.45, 0.25]}


@pytest.mark.parametrize(
    "value",
    [
        {"liver": {"opacity": -0.1}},
        {"liver": {"opacity": 1.1}},
        {"liver": {"opacity": float("nan")}},
        {"liver": {"rgb": [0.0, 0.0, 1.1]}},
        {"liver": {"rgb": [0.0, float("inf"), 0.0]}},
        {"liver": {"rgb": [0.0, 0.0]}},
    ],
)
def test_normalize_layers_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        anatomy_layers.normalize_layers(value)


def test_normalize_layers_rejects_unknown_class():
    with pytest.raises(ValueError, match="unknown class"):
        anatomy_layers.normalize_layers({"pancreas": {}})


def test_normalize_layers_rejects_non_string_class_key_with_value_error():
    with pytest.raises(ValueError, match="keys must be strings"):
        anatomy_layers.normalize_layers({1: {}})


def test_normalize_layers_rejects_unavailable_class():
    with pytest.raises(ValueError, match="unsupported.*liver"):
        anatomy_layers.normalize_layers({"liver": {}}, available_classes={"skeleton"})


def test_normalize_layers_does_not_mutate_input():
    source = {"liver": {"opacity": 0.25, "rgb": [0.1, 0.2, 0.3]}}
    original = copy.deepcopy(source)

    anatomy_layers.normalize_layers(source)

    assert source == original
