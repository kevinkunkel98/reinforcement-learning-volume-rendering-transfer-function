"""Select, split and extract a TotalSegmentator subset for RL v2.

Reads the official small-subset zip (Zenodo record 10047263) directly, picks
subjects stratified by body region, extracts only the selected CT volumes to
data/totalseg/<id>/ct.nii.gz and writes data/totalseg_manifest.json.
Deterministic for a given --seed.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import zipfile

import nibabel as nib
import numpy as np

DEFAULT_ZIP = "data/Totalsegmentator_dataset_small_v201.zip"
DEFAULT_OUT_DIR = "data/totalseg"
DEFAULT_MANIFEST = "data/totalseg_manifest.json"
ZENODO_RECORD = "10047263"
NIFTI_HEADER_BYTES = 352

REGION_BY_STUDY_TYPE = {
    "ct thorax": "thorax",
    "ct thorax-neck": "thorax",
    "ct abdomen": "abdomen_pelvis",
    "ct pelvis": "abdomen_pelvis",
    "ct hip right": "abdomen_pelvis",
    "ct angiography abdomen-pelvis": "abdomen_pelvis",
    "ct thorax-abdomen-pelvis": "trunk",
    "ct neck-thorax-abdomen-pelvis": "trunk",
    "ct angiography thorax-abdomen-pelvis": "trunk",
    "ct polytrauma": "whole_body",
    "ct whole body": "whole_body",
}

# (train, val, test) subjects per region; totals 20 / 4 / 6.
SPLIT_COUNTS = {
    "whole_body": (5, 1, 2),
    "trunk": (5, 1, 2),
    "thorax": (5, 1, 1),
    "abdomen_pelvis": (5, 1, 1),
}

# The dataset is resampled to 1.5 mm isotropic, so the filter is on physical
# extent: it drops heavily cropped scans (e.g. a single hip).
MIN_INPLANE_MM = 250.0
MIN_SUPERIOR_INFERIOR_MM = 150.0

# A class counts as "present" in a subject only once its mask clears this many
# voxels, so stray single-voxel segmentation noise doesn't count as coverage.
MIN_LABEL_VOXELS = 1000

CLASS_NAMES = ("skeleton", "lungs", "organs", "muscle", "vessels")   # label ids 1..5; 0 = other

CLASS_RULES = {
    "skeleton": ("rib_", "vertebrae_", "hip_", "femur_", "humerus_", "scapula_",
                 "clavicula_", "sacrum", "sternum", "skull", "costal_cartilages", "patella",
                 "tibia", "fibula", "carpal", "metacarpal", "phalanges", "tarsal", "metatarsal"),
    "lungs": ("lung_",),
    "organs": ("liver", "spleen", "kidney_", "stomach", "pancreas", "gallbladder", "colon",
               "small_bowel", "duodenum", "esophagus", "urinary_bladder", "prostate",
               "adrenal_gland_", "thyroid_gland", "brain", "spinal_cord", "trachea"),
    "muscle": ("autochthon_", "gluteus_", "iliopsoas_"),
    "vessels": ("aorta", "heart", "atrial_appendage", "brachiocephalic_", "common_carotid_",
                "subclavian_", "pulmonary_", "vena_cava", "portal_vein", "iliac_artery",
                "iliac_vena", "superior_vena_cava", "inferior_vena_cava"),
}


def class_for_structure(structure: str):
    """Which anatomical class a TotalSegmentator structure belongs to, or None."""
    for name in CLASS_NAMES:
        if any(structure.startswith(prefix) or structure == prefix.rstrip("_")
               for prefix in CLASS_RULES[name]):
            return name
    return None


def build_label_volume(masks: dict, shape) -> np.ndarray:
    """Collapse {structure name: boolean mask} into one uint8 label volume.

    Ids are 1..len(CLASS_NAMES) in CLASS_NAMES order, 0 for everything else
    ("other": fat, skin, bowel contents, the scanner table). Classes later in
    CLASS_NAMES overwrite earlier ones where masks overlap.
    """
    labels = np.zeros(shape, dtype=np.uint8)
    for index, name in enumerate(CLASS_NAMES, start=1):
        for structure, mask in masks.items():
            if class_for_structure(structure) == name:
                labels[mask] = index
    return labels


def _subject_label_volume(zf, subject_id, shape):
    """Read every mask of one subject and collapse it into a label volume."""
    masks = {}
    affine = None
    prefix = f"{subject_id}/segmentations/"
    for name in zf.namelist():
        if not name.startswith(prefix) or not name.endswith(".nii.gz"):
            continue
        structure = name[len(prefix):-len(".nii.gz")]
        if class_for_structure(structure) is None:
            continue
        image = nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(name)))
        affine = image.affine
        mask = np.asarray(image.dataobj) > 0
        if mask.shape == tuple(shape):
            masks[structure] = mask
    if affine is None:
        raise ValueError(
            f"{subject_id}: no recognized structure found in its segmentation masks")
    return build_label_volume(masks, tuple(shape)), affine


def region_for_study_type(study_type: str) -> str:
    try:
        return REGION_BY_STUDY_TYPE[study_type.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown study type {study_type!r}") from None


def passes_extent_filter(shape, zooms) -> bool:
    extent = [float(n) * float(z) for n, z in zip(shape[:3], zooms[:3])]
    return min(extent[0], extent[1]) >= MIN_INPLANE_MM and extent[2] >= MIN_SUPERIOR_INFERIOR_MM


def select_and_split(candidates, split_counts=SPLIT_COUNTS, seed=0) -> dict:
    """Map subject id -> "train" | "val" | "test".

    `candidates` is a list of {"id", "region"} dicts. Each region contributes
    exactly its (train, val, test) counts, drawn with a seeded permutation of
    the region's sorted ids, so the result does not depend on input order.
    """
    rng = np.random.default_rng(seed)
    by_region = {}
    for candidate in candidates:
        by_region.setdefault(candidate["region"], []).append(candidate["id"])
    assignment = {}
    for region, (n_train, n_val, n_test) in sorted(split_counts.items()):
        ids = sorted(by_region.get(region, []))
        need = n_train + n_val + n_test
        if len(ids) < need:
            raise ValueError(f"region {region!r} has {len(ids)} candidates, needs {need}")
        picked = [ids[i] for i in rng.permutation(len(ids))[:need]]
        for sid in picked[:n_test]:
            assignment[sid] = "test"
        for sid in picked[n_test:n_test + n_val]:
            assignment[sid] = "val"
        for sid in picked[n_test + n_val:]:
            assignment[sid] = "train"
    return assignment


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_meta(zf: zipfile.ZipFile) -> list:
    lines = zf.read("meta.csv").decode("utf-8-sig").splitlines()
    header = lines[0].split(";")
    return [dict(zip(header, line.split(";"))) for line in lines[1:] if line.strip()]


def read_ct_header(zf: zipfile.ZipFile, subject_id: str):
    """(shape, zooms) from the NIfTI header only, without decompressing the volume."""
    with zf.open(f"{subject_id}/ct.nii.gz") as raw, gzip.open(raw) as stream:
        header = nib.Nifti1Header.from_fileobj(io.BytesIO(stream.read(NIFTI_HEADER_BYTES)))
    return header.get_data_shape(), header.get_zooms()


def build_manifest(zip_path, out_dir=DEFAULT_OUT_DIR, seed=0, split_counts=SPLIT_COUNTS) -> dict:
    """Select subjects, extract their ct.nii.gz into out_dir, return the manifest dict."""
    with zipfile.ZipFile(zip_path) as zf:
        candidates, info = [], {}
        for row in read_meta(zf):
            sid = row["image_id"]
            shape, zooms = read_ct_header(zf, sid)
            if not passes_extent_filter(shape, zooms):
                continue
            try:
                region = region_for_study_type(row["study_type"])
            except ValueError as exc:
                raise ValueError(f"{sid}: {exc}") from exc
            candidates.append({"id": sid, "region": region})
            info[sid] = {
                "study_type": row["study_type"],
                "region": region,
                "shape": [int(v) for v in shape[:3]],
                "spacing": [round(float(v), 6) for v in zooms[:3]],
            }
        assignment = select_and_split(candidates, split_counts, seed)
        subjects = []
        for sid in sorted(assignment):
            path = os.path.join(out_dir, sid, "ct.nii.gz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp"
            with zf.open(f"{sid}/ct.nii.gz") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp_path, path)

            labels, affine = _subject_label_volume(zf, sid, info[sid]["shape"])
            labels_path = os.path.join(out_dir, sid, "labels.nii.gz")
            labels_tmp_path = labels_path + ".tmp"
            with open(labels_tmp_path, "wb") as f:
                f.write(gzip.compress(nib.Nifti1Image(labels, affine).to_bytes()))
            os.replace(labels_tmp_path, labels_path)
            class_counts = {name: int(np.count_nonzero(labels == index))
                             for index, name in enumerate(CLASS_NAMES, start=1)}
            classes_present = sorted(name for name, count in class_counts.items()
                                     if count >= MIN_LABEL_VOXELS)
            print(f"  {sid}: " + ", ".join(f"{name}={count}" for name, count in class_counts.items()))

            subjects.append({"id": sid, "name": f"ts_{sid}", "split": assignment[sid],
                             **info[sid], "path": os.path.relpath(path), "sha256": _sha256_file(path),
                             "labels_path": os.path.relpath(labels_path),
                             "classes_present": classes_present,
                             "contrast": "angiography" in info[sid]["study_type"].lower()})
    return {
        "source": {
            "dataset": "TotalSegmentator small subset v2.0.1",
            "zenodo_record": ZENODO_RECORD,
            "zip": os.path.basename(zip_path),
            "zip_sha256": _sha256_file(zip_path),
            "license": "CC-BY-4.0",
        },
        "selection": {
            "seed": seed,
            "split_counts": {region: list(counts) for region, counts in split_counts.items()},
            "min_inplane_mm": MIN_INPLANE_MM,
            "min_superior_inferior_mm": MIN_SUPERIOR_INFERIOR_MM,
            "n_candidates": len(candidates),
            "numpy_version": np.__version__,
        },
        "subjects": subjects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=DEFAULT_ZIP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.zip):
        raise SystemExit(
            f"{args.zip} not found. Download it first:\n"
            f"  curl -L -o {args.zip} "
            f"'https://zenodo.org/records/{ZENODO_RECORD}/files/{os.path.basename(args.zip)}?download=1'"
        )
    manifest = build_manifest(args.zip, args.out_dir, args.seed)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    counts = {}
    for s in manifest["subjects"]:
        counts[(s["split"], s["region"])] = counts.get((s["split"], s["region"]), 0) + 1
    print(f"[select_totalseg] {len(manifest['subjects'])} subjects from "
          f"{manifest['selection']['n_candidates']} candidates -> {args.manifest}")
    for (split, region), n in sorted(counts.items()):
        print(f"  {split:5s} {region:15s} {n}")


if __name__ == "__main__":
    main()
