import json
import subprocess
import sys

import numpy as np
import pytest

from anatomy import CANONICAL_CLASSES
from tools.compare_layer_renders import (
    compare_samples,
    compare_dataset_samples,
    load_json_record,
    representative_label_ids,
    sampled_layer_contributions,
)


def _sample_record():
    return {
        "image": np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float64),
        "class_visibility": {name: float(index + 1) / 36 for index, name in enumerate(CANONICAL_CLASSES)},
        "cross_class_leakage": 0.0,
    }


def test_identical_sampled_browser_and_reference_records_pass():
    result = compare_samples(_sample_record(), _sample_record())

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["image"]["max_abs"] == 0.0
    assert result["per_class"]["liver"]["max_abs"] == 0.0


def test_image_visibility_and_leakage_tolerances_are_reported_and_fail():
    browser = _sample_record()
    browser["image"] = browser["image"].copy()
    browser["image"][0, 0] = 0.1
    browser["class_visibility"] = dict(browser["class_visibility"])
    browser["class_visibility"]["liver"] += 0.1
    browser["cross_class_leakage"] = 0.2

    result = compare_samples(
        _sample_record(),
        browser,
        image_tolerance=0.01,
        visibility_tolerance=0.01,
        leakage_tolerance=0.01,
    )

    assert result["passed"] is False
    assert {failure["kind"] for failure in result["failures"]} == {
        "image", "per_class_visibility"
    }
    assert result["tolerances"] == {
        "image": 0.01, "per_class_visibility": 0.01, "cross_class_leakage": 0.01
    }


def test_representative_label_ids_use_shared_eight_class_contract():
    labels = np.array([3, 4, 5, 6, 7, 1], dtype=np.uint8)

    assert representative_label_ids(labels) == {
        "liver": 5, "kidneys": 6, "spleen": 7, "heart": 3,
        "vessels": 4, "skeleton": 1,
    }


def test_missing_representative_label_fails_validation():
    assert representative_label_ids(np.array([1, 5], dtype=np.uint8)) == {
        "liver": 5, "skeleton": 1,
    }


def test_sampled_layer_contributions_are_deterministic_and_do_not_leak_classes():
    labels = np.array([[1, 1, 2, 2], [5, 5, 6, 6]], dtype=np.uint8)
    weights = np.ones(labels.shape, dtype=np.float64)
    layers = {name: {"opacity": 0.0} for name in CANONICAL_CLASSES}
    layers["liver"] = {"opacity": 1.0}

    first = sampled_layer_contributions(labels, weights, layers)
    second = sampled_layer_contributions(labels, weights, layers)

    assert first == second
    assert first["liver"] == pytest.approx(0.25)
    assert all(first[name] == 0.0 for name in CANONICAL_CLASSES if name != "liver")


def test_dataset_comparison_derives_leakage_from_contributions_not_input_scalar():
    labels = np.array([[5, 6], [5, 6]], dtype=np.uint8)
    weights = np.ones(labels.shape, dtype=np.float64)
    layers = {name: {"opacity": 0.0} for name in CANONICAL_CLASSES}
    layers["liver"] = {"opacity": 1.0}
    layers["kidneys"] = {"opacity": 1.0}

    result = compare_dataset_samples(labels, weights, layers, reference={
        "image": [[0.0]], "class_visibility": {name: 0.0 for name in CANONICAL_CLASSES},
        "cross_class_leakage": 999.0,
    })

    assert result["derived"]["cross_class_leakage"] == pytest.approx(0.5)
    assert result["reference"]["cross_class_leakage"] == pytest.approx(999.0)


def test_json_loader_rejects_nonfinite_values_and_tolerances():
    with pytest.raises(ValueError, match="finite"):
        load_json_record('{"image": [NaN]}')
    with pytest.raises(ValueError, match="finite"):
        compare_samples(_sample_record(), _sample_record(), image_tolerance=float("inf"))


def test_json_loader_rejects_nonfinite_class_visibility(tmp_path):
    path = tmp_path / "record.json"
    path.write_text(json.dumps({
        "image": [[0.0]],
        "class_visibility": {name: 0.0 for name in CANONICAL_CLASSES} | {"liver": "NaN"},
        "cross_class_leakage": 0.0,
    }))

    with pytest.raises(ValueError, match="finite"):
        load_json_record(str(path))


def test_optional_reference_is_compared_against_derived_browser_record():
    labels = np.array([[5, 6]], dtype=np.uint8)
    weights = np.ones(labels.shape, dtype=np.float64)
    layers = {name: {"opacity": 0.0} for name in CANONICAL_CLASSES}
    layers["liver"] = {"opacity": 1.0}

    result = compare_dataset_samples(labels, weights, layers, reference={
        "image": [[0.0, 0.0]],
        "class_visibility": {name: 0.0 for name in CANONICAL_CLASSES},
        "cross_class_leakage": 0.0,
    })

    assert result["reference_comparison"]["passed"] is False


def test_dataset_report_is_json_serializable():
    labels = np.array([[5, 6]], dtype=np.uint8)
    result = compare_dataset_samples(
        labels, np.ones(labels.shape), {name: {"opacity": 1.0} for name in CANONICAL_CLASSES}
    )
    json.dumps(result)


def test_cli_input_records_are_json_serializable(tmp_path):
    reference = _sample_record()
    browser = _sample_record()
    for record, name in ((reference, "reference"), (browser, "browser")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "image": record["image"].tolist(),
            "class_visibility": record["class_visibility"],
            "cross_class_leakage": record["cross_class_leakage"],
        }))
        assert path.exists()


def test_cli_returns_nonzero_for_difference(tmp_path):
    reference = tmp_path / "reference.json"
    browser = tmp_path / "browser.json"
    reference.write_text(json.dumps({
        "image": [[0.0]],
        "class_visibility": {name: 0.0 for name in CANONICAL_CLASSES},
        "cross_class_leakage": 0.0,
    }))
    browser.write_text(json.dumps({
        "image": [[1.0]],
        "class_visibility": {name: 0.0 for name in CANONICAL_CLASSES},
        "cross_class_leakage": 0.0,
    }))

    result = subprocess.run(
        [sys.executable, "-m", "tools.compare_layer_renders", "fixture",
         "--fixture", str(reference), str(browser)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode != 0
    assert '"passed": false' in result.stdout
