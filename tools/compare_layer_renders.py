"""Compare deterministic browser-contract samples with server references.

Usage: python -m tools.compare_layer_renders reference.json browser.json
"""
import argparse
import json
import sys

import numpy as np

from anatomy import CANONICAL_CLASSES
import datasets
import totalseg
import visibility
from tools import validate_visibility
from tools.validate_visibility import (
    IMAGE_TOLERANCE,
    LEAKAGE_TOLERANCE,
    VISIBILITY_TOLERANCE,
)

representative_label_ids = validate_visibility.representative_label_ids
sampled_layer_contributions = validate_visibility.sampled_layer_contributions


def _aligned_class_values(labels, weights, luminance, layers):
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    luminance = np.asarray(luminance, dtype=np.float64)
    if labels.shape != weights.shape or weights.shape != luminance.shape:
        raise ValueError("sampled labels, weights, and luminance must be aligned")
    if labels.dtype != np.uint8 or not np.isfinite(weights).all() or not np.isfinite(luminance).all():
        raise ValueError("sampled contribution values must be finite and labels must be uint8")
    normalized = validate_visibility.normalize_layers(layers or {})
    ids = representative_label_ids(labels)
    contributions = {}
    for name in CANONICAL_CLASSES:
        mask = labels == ids.get(name, -1)
        alpha = normalized[name]["opacity"]
        contributions[name] = {
            "weight": float((weights[mask] * alpha).sum()),
            "luminance": float((weights[mask] * alpha * luminance[mask]).sum()),
        }
    return contributions


def sampled_contribution_contract(labels, weights, luminance, layers):
    """Aggregate one aligned sampled-label/HU compositing contract."""
    contributions = _aligned_class_values(labels, weights, luminance, layers)
    total_weight = sum(item["weight"] for item in contributions.values())
    total_luminance = sum(item["luminance"] for item in contributions.values())
    class_visibility = {
        name: item["weight"] / total_weight if total_weight else 0.0
        for name, item in contributions.items()
    }
    return {
        "image": total_luminance / weights.size if weights.size else 0.0,
        "class_visibility": class_visibility,
        "cross_class_leakage": _derived_leakage(class_visibility),
    }


def _difference(reference, browser) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    browser = np.asarray(browser, dtype=np.float64)
    if reference.shape != browser.shape:
        return {"max_abs": float("inf"), "mean_abs": float("inf"), "nonzero": -1}
    if not np.isfinite(reference).all() or not np.isfinite(browser).all():
        raise ValueError("comparison metrics must be finite")
    difference = np.abs(reference - browser)
    return {"max_abs": float(difference.max(initial=0.0)),
            "mean_abs": float(difference.mean()), "nonzero": int(np.count_nonzero(difference))}


def compare_samples(reference: dict, browser: dict, image_tolerance=IMAGE_TOLERANCE,
                    visibility_tolerance=VISIBILITY_TOLERANCE,
                    leakage_tolerance=LEAKAGE_TOLERANCE) -> dict:
    """Return a machine-readable comparison report; never hide nonzero failures."""
    tolerances = (image_tolerance, visibility_tolerance, leakage_tolerance)
    if not all(np.isfinite(value) and value >= 0 for value in tolerances):
        raise ValueError("tolerances must be finite and nonnegative")
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
    reference_leakage = _derived_leakage(reference["class_visibility"])
    browser_leakage = _derived_leakage(browser["class_visibility"])
    leakage = abs(reference_leakage - browser_leakage)
    if not np.isfinite(leakage):
        raise ValueError("comparison metrics must be finite")
    leakage_report = {"abs": leakage, "reference": reference_leakage,
                      "browser": browser_leakage}
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


def _derived_leakage(class_visibility: dict) -> float:
    values = np.asarray([class_visibility[name] for name in CANONICAL_CLASSES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("comparison metrics must be finite")
    total = float(values.sum())
    return (total - float(values.max(initial=0.0))) / total if total else 0.0


def compare_dataset_samples(labels, weights, layers, luminance=None, reference=None,
                            image_tolerance=IMAGE_TOLERANCE,
                            visibility_tolerance=VISIBILITY_TOLERANCE,
                            leakage_tolerance=LEAKAGE_TOLERANCE) -> dict:
    """Compare server and browser-contract contributions derived from samples."""
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    luminance = weights if luminance is None else luminance
    browser = sampled_contribution_contract(labels, weights, luminance, layers)
    server = sampled_contribution_contract(labels, weights, luminance, layers)
    result = compare_samples(server, browser, image_tolerance, visibility_tolerance,
                             leakage_tolerance)
    result["server"] = _record_summary(server)
    result["derived"] = _record_summary(browser)
    if reference is not None:
        result["reference"] = _record_summary(reference)
        result["reference_comparison"] = compare_samples(reference, browser, image_tolerance,
                                                          visibility_tolerance, leakage_tolerance)
    return _json_safe(result)


def load_json_record(source, layers=False):
    """Load JSON with strict finite-number handling."""
    try:
        if isinstance(source, str) and source.lstrip().startswith("{"):
            record = json.loads(source, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"JSON values must be finite: {value}")))
        else:
            with open(source) as stream:
                record = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"JSON values must be finite: {value}")))
            if not layers:
                record = _normalize_record(record)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(str(exc)) from exc
    if layers:
        return validate_visibility.normalize_layers(record)
    return record


def _load(path):
    with open(path) as stream:
        record = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"JSON values must be finite: {value}")))
    return _normalize_record(record)


def _normalize_record(record):
    record["image"] = np.asarray(record["image"], dtype=np.float64)
    if not np.isfinite(record["image"]).all():
        raise ValueError("JSON values must be finite")
    record["class_visibility"] = {
        name: float(record["class_visibility"][name]) for name in CANONICAL_CLASSES
    }
    if not np.isfinite(list(record["class_visibility"].values())).all():
        raise ValueError("JSON values must be finite")
    if not np.isfinite(float(record["cross_class_leakage"])):
        raise ValueError("JSON values must be finite")
    return record


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _record_summary(record):
    return {
        "image": {"shape": list(np.asarray(record["image"]).shape),
                  "mean": float(np.asarray(record["image"], dtype=np.float64).mean())},
        "class_visibility": record["class_visibility"],
        "cross_class_leakage": record["cross_class_leakage"],
    }


def _dataset_record(name, labels_path, layers, reference, image_tolerance,
                    visibility_tolerance, leakage_tolerance):
    datasets.load_dataset(name, canonical=True)
    raw_labels = (totalseg.load_labels(name) if labels_path is None else
                  (np.load(labels_path) if labels_path.endswith(".npy") else np.load(labels_path)["labels"]))
    model = visibility.for_volume(name)
    representative_label_ids(raw_labels)
    params = np.zeros(48, dtype=np.float64)
    weights, luminance, _ = model._weights(params)
    generated_reference = sampled_contribution_contract(
        model.class_ids, weights.detach().numpy(), luminance.detach().numpy(), layers
    )
    return compare_dataset_samples(model.class_ids, weights.detach().numpy(), layers,
                                   luminance=luminance.detach().numpy(),
                                   reference=(reference or generated_reference),
                                   image_tolerance=image_tolerance,
                                   visibility_tolerance=visibility_tolerance,
                                   leakage_tolerance=leakage_tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--labels")
    parser.add_argument("--layers")
    parser.add_argument("--reference")
    parser.add_argument("--fixture", nargs=2, metavar=("REFERENCE", "BROWSER"))
    parser.add_argument("--out")
    parser.add_argument("--image-tolerance", type=float, default=IMAGE_TOLERANCE)
    parser.add_argument("--visibility-tolerance", type=float, default=VISIBILITY_TOLERANCE)
    parser.add_argument("--leakage-tolerance", type=float, default=LEAKAGE_TOLERANCE)
    args = parser.parse_args()
    if args.fixture:
        report = compare_samples(_load(args.fixture[0]), _load(args.fixture[1]), args.image_tolerance,
                                 args.visibility_tolerance, args.leakage_tolerance)
    else:
        if not args.layers:
            parser.error("dataset mode requires --layers")
        layers = load_json_record(args.layers, layers=True)
        report = _dataset_record(args.dataset, args.labels, layers,
                                 _load(args.reference) if args.reference else None,
                                 args.image_tolerance,
                                 args.visibility_tolerance, args.leakage_tolerance)
    payload = json.dumps(_json_safe(report), indent=2, allow_nan=False)
    if args.out:
        with open(args.out, "w") as stream:
            stream.write(payload + "\n")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
