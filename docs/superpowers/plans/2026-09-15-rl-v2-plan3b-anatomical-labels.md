# RL v2 Plan 3b: Anatomical Labels and a Corrected Validation Gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute visibility to anatomical classes (skeleton, lungs, organs, muscle, vessels) from the TotalSegmentator masks instead of Hounsfield bands, and validate the estimate against VTK renders with a reference that actually measures a class's contribution.

**Architecture:** The selection tool also extracts each subject's 117 masks and collapses them into one label volume. `totalseg.py` serves those labels; `visibility.py` labels samples by anatomy when labels exist and by intensity bands otherwise (the four Slicer CTs, the phantom), and reports which source it used. The validation gate renders each state twice with VTK's label-map masking — once normally, once with one class's colour black and its opacity untouched — and compares the difference against the estimate.

**Tech Stack:** Python 3.14 (`.venv`), nibabel, numpy, torch, VTK 9.7, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md` (sections "Data", "Visibility and brightness estimate", "Goals").

**Why this plan exists (measured, 2026-09-15):**
- The intensity-band labels failed validation on 2 of 3 volumes: fat 0.46/0.68 and spongy 0.49/0.58 Pearson on the TotalSegmentator volumes, while bone (0.98–1.00) and soft (0.92–0.96) held. The bands are catch-alls: "fat" (−550..−30 HU) also collects lung edges and partial-volume voxels, "spongy" (170..600) collects contrast-filled vessels.
- Two flaws in the first validation method, both fixed here: deleting a class's voxels opens holes that reveal what is behind (blacken the colour instead, keeping opacity), and random transfer functions barely move a given class or the coverage (sweep the class's own peak; sweep global opacity for coverage).
- Measured class separability (subjects s0454 plain, s1379 contrast; median HU): skeleton 238/268, lungs −/−856, organs −49/57, muscle 15/45, vessels 38/133. Vessels are only separable with contrast — on a plain scan no intensity-based transfer function can single them out, so vessels are only a goal on contrast scans.
- Only 7–20% of body voxels carry any mask, so `other` (fat, skin, bowel contents, table) is most of what occludes.
- VTK 9.7 supports label-map masking (`vtkGPUVolumeRayCastMapper.SetMaskInput` + `SetMaskTypeToLabelMap`, `vtkVolumeProperty.SetLabelColor/SetLabelScalarOpacity`). Verified end to end: with a bone-dominant TF on ts_s0454, mean luminance is 0.1481 normally, 0.1065 with the skeleton blacked out (its contribution), and 0.1194 with the skeleton made invisible — the last being higher precisely because removing it reveals the background.
- Building one subject's skeleton mask from its 63 structure files takes ~0.4 s, so all 117 structures for 30 subjects is minutes, not hours.

**Repo rules:**
- Run from the repository root with `.venv/bin/python`.
- Commit messages: plain `git commit -m "..."`, **no** trailers.
- Stage files explicitly by path; never `git add -A`/`.`. Leave untracked `docs/prompts_report.pdf` and `docs/vr-integration.md` alone.
- Branch `rl-v2`.

---

## Class definitions (used by every task)

```python
# tools/select_totalseg.py
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
```

A structure belongs to the first class whose prefixes match; unmatched structures stay `other`. Later ids win on overlap in the order above (skeleton first, vessels last), so a voxel claimed by two masks gets the later class — overlaps are rare and this keeps the rule deterministic.

---

### Task 1: Extract masks and build label volumes

**Files:** modify `tools/select_totalseg.py`, `tests/test_select_totalseg.py`; regenerate `data/totalseg_manifest.json`.

- [ ] **Step 1: Write failing tests** in `tests/test_select_totalseg.py`:
  - `test_class_for_structure_maps_prefixes`: `class_for_structure("rib_left_4") == "skeleton"`, `("lung_upper_lobe_left") == "lungs"`, `("aorta") == "vessels"`, `("autochthon_left") == "muscle"`, `("liver") == "organs"`, `("unknown_thing") is None`.
  - `test_build_label_volume_assigns_ids`: given a dict `{"rib_left_1": mask_a, "liver": mask_b, "unknown": mask_c}` of boolean arrays, `build_label_volume(masks, shape)` returns a `uint8` array with `CLASS_NAMES.index("skeleton") + 1` where `mask_a`, `organs` id where `mask_b`, and `0` where only `mask_c`.
  - `test_build_label_volume_later_class_wins_on_overlap`: a voxel in both a rib mask and an aorta mask gets the `vessels` id.
  - Extend the fake-zip end-to-end test: each fake subject also gets `segmentations/rib_left_1.nii.gz` and `segmentations/liver.nii.gz`; assert the manifest entry has `labels_path` pointing at an existing file, `classes_present == ["organs", "skeleton"]` (sorted), and `contrast is False`; assert the saved label volume has the expected ids.

- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_select_totalseg.py` — expect the new tests to fail (`ImportError`/`AttributeError`).

- [ ] **Step 3: Implement** in `tools/select_totalseg.py`: add `CLASS_NAMES`/`CLASS_RULES` from the section above, plus:

```python
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
    prefix = f"{subject_id}/segmentations/"
    for name in zf.namelist():
        if not name.startswith(prefix) or not name.endswith(".nii.gz"):
            continue
        structure = name[len(prefix):-len(".nii.gz")]
        if class_for_structure(structure) is None:
            continue
        image = nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(name)))
        mask = np.asarray(image.dataobj) > 0
        if mask.shape == tuple(shape):
            masks[structure] = mask
    return build_label_volume(masks, tuple(shape)), image.affine
```

In `build_manifest`, after extracting a subject's `ct.nii.gz`, build its label volume and save it next to the CT as `labels.nii.gz` (`nib.Nifti1Image(labels, affine)`, same affine as the masks so the canonical-RAS loader reorients it identically). Add to each subject entry: `"labels_path"`, `"classes_present"` (sorted names with at least 1000 labelled voxels), `"contrast"` (`"angiography" in study_type`). Print the class counts per subject in `main()`.

- [ ] **Step 4:** Run the tests — all pass.

- [ ] **Step 5:** Re-run the real extraction: `.venv/bin/python -m tools.select_totalseg`. Expect the same 30 subjects, splits and CT sha256 values as before (only new fields and new `labels.nii.gz` files). Verify with `git diff data/totalseg_manifest.json`: subject ids, splits, paths and `sha256` unchanged; new keys added. If any subject or split changed, STOP and report.

- [ ] **Step 6:** Commit `tools/select_totalseg.py tests/test_select_totalseg.py data/totalseg_manifest.json` with message `feat(data): extract anatomical label volumes for the RL v2 subjects`.

---

### Task 2: Serve labels from `totalseg.py`

**Files:** modify `totalseg.py`, `tests/test_totalseg.py`.

- [ ] **Step 1: Write failing tests**: `has_labels(name)` is True only when the manifest entry has a `labels_path` that exists; `load_labels(name)` returns a `uint8` array in canonical RAS with the same shape as `load_volume(name)[0]`; `classes_present(name)` returns the manifest list; `is_contrast(name)` returns the flag; `load_labels` raises `FileNotFoundError` naming `tools.select_totalseg` when the file is missing.

- [ ] **Step 2:** Run them — expect failures.

- [ ] **Step 3: Implement** in `totalseg.py`:

```python
def has_labels(name: str) -> bool:
    entry = _subjects().get(name)
    return bool(entry and entry.get("labels_path") and os.path.exists(entry["labels_path"]))


def classes_present(name: str) -> list:
    return list(subject(name).get("classes_present", []))


def is_contrast(name: str) -> bool:
    return bool(subject(name).get("contrast", False))


def load_labels(name: str) -> np.ndarray:
    """Anatomical class ids per voxel, canonical RAS, matching load_volume()."""
    entry = subject(name)
    path = entry.get("labels_path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{name} has no label volume; {_FETCH_HINT}")
    image = nib.as_closest_canonical(nib.load(path))
    return np.ascontiguousarray(np.asarray(image.dataobj, dtype=np.uint8))
```

- [ ] **Step 4:** Tests pass. Also check on real data that `load_labels(name).shape == load_volume(name)[0].shape` for one train volume and that the class ids present match `classes_present(name)`.

- [ ] **Step 5:** Commit `totalseg.py tests/test_totalseg.py` with message `feat(data): serve anatomical labels and contrast flag`.

---

### Task 3: Anatomical classes in `visibility.py`

**Files:** modify `visibility.py`, `tools/build_visibility_cache.py`, `tests/test_visibility.py`.

Changes:
- `CLASSES = ("skeleton", "lungs", "organs", "muscle", "vessels")` replaces `TISSUES`. `features()` returns `vis`/`bright` keyed by these names.
- The cube cache stores a second `uint8` array, `class_ids`, sampled from the label volume with the same geometry and `mode="nearest"`; `label_source` (`"anatomy"` or `"intensity"`) is stored with it and exposed as `VisibilityModel.label_source`.
- Volumes without labels (the four Slicer CTs, the phantom, any manifest entry lacking `labels_path`) fall back to intensity bands mapped onto the same class names: `skeleton` = HU ≥ 300, `lungs` = HU ≤ −500, `organs` = −30..300, and `muscle`/`vessels` empty. `VisibilityModel.label_source` is then `"intensity"`, and anything reported from such a volume must say so.
- `solo_max(c)` = the largest `vis_c` over the four single-peak transfer functions (peak *i* at height 1, others at 0), cached per class.
- `CACHE_VERSION` bumps to 2 so old caches are ignored.

- [ ] **Step 1: Write failing tests** in `tests/test_visibility.py`, reusing the existing slab/core helpers but building models from explicit label volumes:
  - `test_classes_come_from_the_label_volume`: a volume where a bone-HU core carries the `organs` id shows its contribution under `organs`, not `skeleton` — labels, not intensities, decide.
  - `test_label_source_is_reported`: a model built with labels reports `label_source == "anatomy"`; one built without reports `"intensity"`.
  - `test_intensity_fallback_maps_to_the_same_class_names`: a bone-HU slab with no labels lands in `skeleton`, a lung-HU slab in `lungs`.
  - `test_solo_max_uses_the_best_single_peak`: for a bone-HU core labelled `skeleton`, `solo_max("skeleton")` is > 0 and ≥ the visibility under the default transfer function.
  - Update the existing occlusion/brightness/coverage tests to the new class names.

- [ ] **Step 2:** Run them — expect failures.

- [ ] **Step 3: Implement.** `VisibilityModel.from_volume(volume, spacing, labels=None, ...)` samples `labels` (when given) through the same `_sample_cubes` geometry with `mode="nearest"` and stores the result as `class_ids`; when `labels is None` it derives ids from the quantized intensities with the fallback thresholds above. `_labels` becomes that stored array, so `features()` needs no other change. `for_volume(name)` passes `totalseg.load_labels(name)` when `totalseg.has_labels(name)`.

- [ ] **Step 4:** Tests pass.

- [ ] **Step 5:** Rebuild every cache: `.venv/bin/python -m tools.build_visibility_cache`. Expect 34 volumes, the 30 TotalSegmentator ones reporting `anatomy` and the 4 Slicer ones `intensity` (print the source per volume). Then check on `ts_s1379` (a contrast scan) that `features(default_params())["vis"]` has non-zero `skeleton`, `lungs` and `vessels`, and that clearing the soft-tissue peaks raises `skeleton` visibility at least 3×.

- [ ] **Step 6:** Commit `visibility.py tools/build_visibility_cache.py tests/test_visibility.py` with message `feat(rl): attribute visibility to anatomical classes`.

---

### Task 4: Corrected validation gate

**Files:** rewrite `tools/validate_visibility.py` (currently uncommitted), `tests/test_validate_visibility.py`.

Method per volume:
1. For each class present, sample 10 transfer functions by sweeping the peak that best matches that class (the peak whose single-peak transfer function maximises `vis_c`, i.e. the same peak `solo_max` found) from height 0.02 to 1.0, with the other peaks jittered by a seeded generator.
2. Estimate = `vis_c * bright_c`.
3. Reference = mean luminance over the 6 views of (normal render) − (render with class `c`'s label colour black, opacity untouched), using VTK label-map masking. Volumes without a label volume are skipped with `label_source: "intensity"` noted in the report.
4. Coverage: a separate global opacity sweep (all peaks together, height 0 → 1), rank agreement between estimated and rendered coverage.

Thresholds: Pearson ≥ 0.7 per class whose reference range ≥ `CONTRIBUTION_FLOOR` (0.002); rank agreement ≥ 0.9 for coverage. Classes below the floor are reported `unvalidated`, not failures.

The renderer helper (label mask handling) lives in this tool, since it reaches into `render._get_pipeline`:

```python
def _render_with_label_mask(volume, spacing, params, cameras, labels, black_label=None):
    """Mean luminance over the views; black_label renders that class's colour as
    black while leaving every opacity untouched, so occlusion is unchanged."""
    prop, renderer, window = render._get_pipeline(volume, spacing)
    mapper = renderer.GetVolumes().GetLastProp().GetMapper()
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
```

`_black_colour()` returns a `vtkColorTransferFunction` that is black over `CENTER_RANGE`.

- [ ] **Step 1:** Keep the existing helper tests (`pearson`, `summarize`, noise floor) and add: `test_sweep_params_varies_the_matching_peak` (the swept peak's height is monotonically increasing across the returned transfer functions) and `test_rank_agreement_helper` for the coverage measure.
- [ ] **Step 2:** Run — expect failures.
- [ ] **Step 3:** Implement as described.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5: Run the gate** on `ts_s1379` (contrast; has all five classes), `ts_s1337` and `ts_s0454`:
  `.venv/bin/python -m tools.validate_visibility ts_s1379 ts_s1337 ts_s0454`
  Print every class line and write `out/visibility_validation.json`. **This is a gate:** if a validated class fails, STOP and report the JSON. Do not lower the threshold or tune the estimate.
- [ ] **Step 6:** Commit `tools/validate_visibility.py tests/test_validate_visibility.py` with message `feat(rl): validate anatomical visibility against label-masked renders`.

---

### Task 5: Documentation

**Files:** modify `README.md`.

- [ ] **Step 1:** Rewrite the "Visibility estimate" subsection (or add it, if Plan 3's Task 6 never ran): explain anatomical classes from TotalSegmentator masks, the `other` majority, the intensity fallback for the Slicer CTs, the label-mask validation method, and the measured correlations from `out/visibility_validation.json` (fill in real numbers, no placeholders). State plainly that vessels are only a goal on contrast scans.
- [ ] **Step 2:** `.venv/bin/python -m pytest -q` — all pass.
- [ ] **Step 3:** Commit `README.md` with message `docs: describe anatomical visibility classes and their validation`.
