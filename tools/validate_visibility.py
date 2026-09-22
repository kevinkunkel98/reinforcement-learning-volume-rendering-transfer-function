"""Does the visibility estimate track real renders?

For each anatomical class a volume's label volume carries, sweep 10 transfer
functions that isolate that class (the peak whose single-peak transfer
function shows the most of it -- the same peak `visibility.solo_max` would
find -- swept from height 0.02 to 1.0, the other peaks jittered by a seeded
generator), and compare the estimate's vis*bright against a real reference:
mean luminance over the 6 standard views, normal render minus a render where
that class's label colour is blacked out (opacity left untouched) using
VTK's label-map masking. Blackening, not deleting the class's voxels, keeps
occlusion the same -- deleting a class opens a hole that reveals whatever
sits behind it, which is not that class's actual contribution to the
picture.

Coverage is checked separately, by rank agreement rather than Pearson: a
global opacity sweep (every peak's height together, 0 -> 1) compared between
the estimate's `coverage` and the rendered fraction of lit pixels, since only
the ordering (more opacity -> more of the frame covered), not the raw scale,
needs to match.

A class whose real contribution barely moves across the sampled transfer
functions cannot be validated this way (it sits near the renderer's noise
floor) and is reported unvalidated rather than counted as a failure. Volumes
without a label volume (`label_source == "intensity"`) are skipped entirely
-- this validation method only applies to anatomical labels.

    python -m tools.validate_visibility ts_s1379 ts_s1337 ts_s0454
"""
import argparse
import json
import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk  # pyright: ignore[reportMissingImports]

import datasets
import render
import totalseg
import views
import visibility
from transfer import CENTER_RANGE, N_PEAKS, PARAMS_PER_PEAK, _from_unit, anatomical_params, vector_to_vtk

CONTRIBUTION_FLOOR = 0.002       # luminance range below this is renderer noise
DEFAULT_THRESHOLD = 0.7          # Pearson correlation required per validated class
COVERAGE_THRESHOLD = 0.9         # rank agreement required for coverage
N_TRIALS = 10
OUT_PATH = "out/visibility_validation.json"


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def rank_agreement(a, b) -> float:
    """Spearman rank correlation (Pearson over each series' ranks), used for
    coverage since only relative ordering needs to match, not raw scale."""
    a_ranks = np.argsort(np.argsort(np.asarray(a)))
    b_ranks = np.argsort(np.argsort(np.asarray(b)))
    return pearson(a_ranks, b_ranks)


def summarize(estimate: dict, reference: dict, threshold=DEFAULT_THRESHOLD) -> dict:
    summary = {}
    for name, values in reference.items():
        values = np.asarray(values, dtype=np.float64)
        spread = float(values.max() - values.min())
        entry = {"contribution_range": spread, "validated": spread >= CONTRIBUTION_FLOOR,
                 "pearson": pearson(estimate[name], values)}
        entry["passed"] = (entry["pearson"] >= threshold) if entry["validated"] else None
        summary[name] = entry
    return summary


def _best_peak(model: "visibility.VisibilityModel", name: str) -> int:
    """Which single peak, alone at full height, shows the most of `name` --
    the same peak visibility.VisibilityModel.solo_max finds, and the peak
    sweep_params sweeps to isolate that class."""
    best_peak, best_vis = 0, -1.0
    for peak in range(N_PEAKS):
        params = anatomical_params().copy()
        for i in range(N_PEAKS):
            params[i * PARAMS_PER_PEAK + 2] = 1.0 if i == peak else -1.0
        vis = model.features(params)["vis"][name]
        if vis > best_vis:
            best_vis, best_peak = vis, peak
    return best_peak


def sweep_params(rng: np.random.Generator, peak: int, n: int = N_TRIALS) -> list:
    """n transfer functions with `peak`'s height swept 0.02 -> 1.0 and the
    other peaks' height/width jittered by `rng`."""
    functions = []
    for height in np.linspace(0.02, 1.0, n):
        params = anatomical_params().copy()
        for i in range(N_PEAKS):
            if i == peak:
                params[i * PARAMS_PER_PEAK + 2] = _from_unit(float(height))
            else:
                params[i * PARAMS_PER_PEAK + 2] = np.clip(
                    params[i * PARAMS_PER_PEAK + 2] + rng.uniform(-1.0, 1.0), -1.0, 1.0)
                params[i * PARAMS_PER_PEAK + 1] = np.clip(
                    params[i * PARAMS_PER_PEAK + 1] + rng.uniform(-0.5, 0.5), -1.0, 1.0)
        functions.append(params)
    return functions


def sweep_global_opacity(n: int = N_TRIALS) -> list:
    """n transfer functions with every peak's height swept together 0 -> 1,
    for the coverage check -- no single class is isolated here."""
    functions = []
    for height in np.linspace(0.0, 1.0, n):
        params = anatomical_params().copy()
        for i in range(N_PEAKS):
            params[i * PARAMS_PER_PEAK + 2] = _from_unit(float(height))
        functions.append(params)
    return functions


def _black_colour() -> vtk.vtkColorTransferFunction:
    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(CENTER_RANGE[0], 0.0, 0.0, 0.0)
    ctf.AddRGBPoint(CENTER_RANGE[1], 0.0, 0.0, 0.0)
    return ctf


def _render_with_label_mask(volume, spacing, params, cameras, labels, black_label=None):
    """Mean luminance over the views; black_label renders that class's colour as
    black while leaving every opacity untouched, so occlusion is unchanged."""
    prop, renderer, window = render._get_pipeline(volume, spacing)
    # vtk's stubs type GetLastProp()'s return as the generic vtkProp base
    # class; at runtime it's the vtkVolume this pipeline actually added.
    mapper = renderer.GetVolumes().GetLastProp().GetMapper()  # pyright: ignore[reportAttributeAccessIssue]
    label_image = vtk.vtkImageData()
    label_image.SetDimensions(*labels.shape)
    label_image.SetSpacing(*spacing)
    label_image.GetPointData().SetScalars(numpy_to_vtk(
        np.ascontiguousarray(labels.ravel(order="F")), deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR))
    mapper.SetMaskInput(label_image)
    mapper.SetMaskTypeToLabelMap()
    mapper.SetMaskBlendFactor(1.0)
    try:
        colour, opacity = vector_to_vtk(params)
        for label in range(1, len(visibility.CLASSES) + 1):
            prop.SetLabelColor(label, _black_colour() if label == black_label else colour)
            prop.SetLabelScalarOpacity(label, opacity)
        frames = [render.grab(render.render(volume, params, spacing, camera)) for camera in cameras]
    finally:
        mapper.SetMaskInput(None)          # never leak the mask into later renders
    return float(np.mean([frame.astype(np.float64).mean() / 255.0 for frame in frames]))


def _rendered_coverage(volume, spacing, params, cameras) -> float:
    frames = [render.grab(render.render(volume, params, spacing, camera)) for camera in cameras]
    return float(np.mean([(frame.astype(np.float64).mean(axis=2) > 2).mean() for frame in frames]))


def validate_volume(name: str, seed: int = 0, threshold: float = DEFAULT_THRESHOLD,
                     n_trials: int = N_TRIALS) -> dict:
    model = visibility.for_volume(name)
    if model.label_source != "anatomy":
        return {"volume": name, "label_source": model.label_source, "skipped": True}

    volume, spacing = datasets.load_dataset(name, canonical=True)
    labels = totalseg.load_labels(name)
    cameras = views.cameras_for_volume(volume, spacing)
    rng = np.random.default_rng(seed)

    estimate, reference = {}, {}
    for class_name in totalseg.classes_present(name):
        class_id = visibility.CLASSES.index(class_name) + 1
        peak = _best_peak(model, class_name)
        sampled = sweep_params(rng, peak, n_trials)
        feats = [model.features(p) for p in sampled]
        estimate[class_name] = np.array([f["vis"][class_name] * f["bright"][class_name] for f in feats])
        normal = np.array([_render_with_label_mask(volume, spacing, p, cameras, labels)
                           for p in sampled])
        blackened = np.array([_render_with_label_mask(volume, spacing, p, cameras, labels, black_label=class_id)
                              for p in sampled])
        reference[class_name] = normal - blackened

    classes = summarize(estimate, reference, threshold)

    coverage_params = sweep_global_opacity(n_trials)
    coverage_estimate = np.array([model.features(p)["coverage"] for p in coverage_params])
    coverage_reference = np.array([_rendered_coverage(volume, spacing, p, cameras) for p in coverage_params])
    coverage = {"rank_agreement": rank_agreement(coverage_estimate, coverage_reference)}
    coverage["passed"] = coverage["rank_agreement"] >= COVERAGE_THRESHOLD

    return {"volume": name, "label_source": model.label_source, "trials": n_trials, "seed": seed,
            "classes": classes, "coverage": coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    reports, failures = [], []
    for name in args.names:
        report = validate_volume(name, args.seed, args.threshold, args.trials)
        reports.append(report)
        if report.get("skipped"):
            print(f"{name}: skipped (label_source={report['label_source']!r})")
            continue
        print(f"{name}: coverage rank_agreement={report['coverage']['rank_agreement']:+.3f} "
              f"{'PASS' if report['coverage']['passed'] else 'FAIL'}")
        if not report["coverage"]["passed"]:
            failures.append(f"{name}: coverage")
        for class_name, entry in report["classes"].items():
            state = "unvalidated" if not entry["validated"] else ("PASS" if entry["passed"] else "FAIL")
            print(f"    {class_name:8s} pearson={entry['pearson']:+.3f} "
                  f"range={entry['contribution_range']:.4f} {state}")
            if entry["validated"] and not entry["passed"]:
                failures.append(f"{name}: {class_name}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump({"threshold": args.threshold, "coverage_threshold": COVERAGE_THRESHOLD,
                   "contribution_floor": CONTRIBUTION_FLOOR, "reports": reports}, stream, indent=2)
    print(f"[validate_visibility] wrote {args.out}")
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))
    print("[validate_visibility] gate passed")


if __name__ == "__main__":
    main()
