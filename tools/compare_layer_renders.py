"""Compare deterministic browser-contract samples with server references.

Usage: python -m tools.compare_layer_renders reference.json browser.json
"""
import argparse
import json
import sys

import numpy as np

from anatomy import CANONICAL_CLASSES
from tools import validate_visibility
from tools.validate_visibility import (
    IMAGE_TOLERANCE,
    LEAKAGE_TOLERANCE,
    VISIBILITY_TOLERANCE,
)

representative_label_ids = validate_visibility.representative_label_ids
sampled_layer_contributions = validate_visibility.sampled_layer_contributions


def _difference(reference, browser) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    browser = np.asarray(browser, dtype=np.float64)
    if reference.shape != browser.shape:
        return {"max_abs": float("inf"), "mean_abs": float("inf"), "nonzero": -1}
    difference = np.abs(reference - browser)
    return {"max_abs": float(difference.max(initial=0.0)),
            "mean_abs": float(difference.mean()), "nonzero": int(np.count_nonzero(difference))}


def compare_samples(reference: dict, browser: dict, image_tolerance=IMAGE_TOLERANCE,
                    visibility_tolerance=VISIBILITY_TOLERANCE,
                    leakage_tolerance=LEAKAGE_TOLERANCE) -> dict:
    """Return a machine-readable comparison report; never hide nonzero failures."""
    image = _difference(reference["image"], browser["image"])
    per_class = {}
    failures = []
    for name in CANONICAL_CLASSES:
        difference = _difference(reference["class_visibility"][name], browser["class_visibility"][name])
        per_class[name] = difference
        if difference["max_abs"] > visibility_tolerance:
            failures.append({"kind": "per_class_visibility", "class": name, **difference})
    if image["max_abs"] > image_tolerance:
        failures.append({"kind": "image", **image})
    leakage = abs(float(reference["cross_class_leakage"]) - float(browser["cross_class_leakage"]))
    leakage_report = {"abs": leakage, "reference": float(reference["cross_class_leakage"]),
                      "browser": float(browser["cross_class_leakage"])}
    if leakage > leakage_tolerance:
        failures.append({"kind": "cross_class_leakage", **leakage_report})
    return {
        "passed": not failures,
        "failures": failures,
        "tolerances": {"image": image_tolerance, "per_class_visibility": visibility_tolerance,
                        "cross_class_leakage": leakage_tolerance},
        "image": image,
        "per_class": per_class,
        "cross_class_leakage": leakage_report,
    }


def _load(path):
    with open(path) as stream:
        record = json.load(stream)
    record["image"] = np.asarray(record["image"], dtype=np.float64)
    record["class_visibility"] = {
        name: float(record["class_visibility"][name]) for name in CANONICAL_CLASSES
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("browser")
    parser.add_argument("--out")
    parser.add_argument("--image-tolerance", type=float, default=IMAGE_TOLERANCE)
    parser.add_argument("--visibility-tolerance", type=float, default=VISIBILITY_TOLERANCE)
    parser.add_argument("--leakage-tolerance", type=float, default=LEAKAGE_TOLERANCE)
    args = parser.parse_args()
    report = compare_samples(_load(args.reference), _load(args.browser), args.image_tolerance,
                             args.visibility_tolerance, args.leakage_tolerance)
    payload = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as stream:
            stream.write(payload + "\n")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
