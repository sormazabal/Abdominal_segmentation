# Abdominal CT Annotation Pipeline — Design

## Goal

Replicate the chest CT pipeline (`chest_ct_annotation.py` + `chest_classes.py` +
`build_pseudo_labels.py` + `kaggle_train_chest_nnunet.ipynb` +
`finetune_chest_nnunet.py`), which itself replicates the brain MRI reference
pipeline pattern in `reference_examples/`, for abdominal CT.

## Scope

- Organ annotation only. No pathology classifier (no nodule/stone/tumor
  detector slot) in this pass.
- Six core classes: liver, kidney_right, kidney_left, spleen, pancreas, aorta.
- Both pipeline layers, as separate new files — no shared/parametrized module
  with the chest scripts. The codebase's own convention (brain vs. chest) is
  full per-domain duplication, and the existing chest_* files stay untouched.

## Files

### `abdomen_classes.py`

Mirrors `chest_classes.py`.

- `BASE_MODEL_STRUCTURES = ("liver", "kidney_right", "kidney_left", "spleen", "pancreas", "aorta")`
  — these are already atomic single structures in TotalSegmentator's `total`
  task class map, so no left/right part-merging (like `LEFT_LUNG_PARTS`) is
  needed.
- `ABDOMEN_LABELS = {"background": 0, "liver": 1, "kidney_right": 2, "kidney_left": 3, "spleen": 4, "pancreas": 5, "aorta": 6}`
- `DEFAULT_COLORS` / `DISPLAY_NAMES` for the 6 classes.
- `combine_anatomical_masks(raw_mask, label_map, extra_structures)` — same
  signature and extras-append-from-7 behavior as chest's version, but the
  per-class loop is a straight `raw_mask == source_label` remap (no grouping
  tuples) since every base structure is 1:1.

### `abdomen_ct_annotation.py`

Mirrors `chest_ct_annotation.py` structure-for-structure:

- Same `LoadedCT` dataclass, `configure_logging`, `is_nifti`,
  `looks_like_dicom`, `load_dicom_series`, `inspect_dicom_spacing`, `load_ct`,
  `validate_ct`, `resolve_device`, `parse_extra_structures`,
  `run_anatomical_segmentation`, `run_nnunet_segmentation`, `load_colors`,
  `window_ct`, `to_radiological`, `find_contours`, `decimate_contour`,
  `bbox_for_mask`, `hex_to_rgb`, `compose_overlay`, `save_overlays`,
  `label_position`, `build_slice_annotations`, `save_annotated_slice`,
  `save_mask`, `build_parser`, `run`, `main` — same shapes and same output
  contract (`segmentation_mask.nii.gz`, `annotated_slice.png`,
  `annotations.json`, `overlays/slice_NNN.png`).
- Removed entirely: `NoduleDetection`, `run_nodule_detector`,
  `nodule_annotations`, `--enable-nodules`, `--nodule-checkpoint`, and all
  nodule-specific branches in `run()` / the JSON payload / `save_annotated_slice`.
- `WINDOW_CENTER = 40.0`, `WINDOW_WIDTH = 400.0` (abdominal soft-tissue window,
  replacing chest's lung window of -600/1500).
- `informative_slice()` scores on the 6 abdomen classes instead of chest's
  `("lung_right", "lung_left", "heart", "trachea", "aorta", "vertebrae")`
  tuple, using `("liver", "kidney_right", "kidney_left", "spleen", "pancreas", "aorta")`.
- `label_position()` layout tweaks: no `trachea`/`vertebrae` special-cases;
  keep the generic left/right leader-line placement for all 6 classes.
- Imports from `abdomen_classes` instead of `chest_classes`.
- Docstring/log strings/JSON `"anatomy"` field say "abdomen"/"abdominal CT"
  instead of "chest".

### `build_pseudo_labels_abdomen.py`

Mirrors `build_pseudo_labels.py`:

- Imports from `abdomen_classes` and `abdomen_ct_annotation` instead of the
  chest equivalents.
- `ABDOMEN_CLASS_NAMES = [name for name in ABDOMEN_LABELS if name != "background"]`.
- Same `discover_cases`, `class_voxel_counts`, `build_one_case`,
  `MAX_MISSING_CLASSES = 1` quality gate, `build_parser`, `run`, `main`.
- `dataset.json["labels"]` uses `ABDOMEN_LABELS`.

### `kaggle_train_abdomen_nnunet.ipynb`

Mirrors `kaggle_train_chest_nnunet.ipynb` cell-for-cell: same
env-var/dataset-id setup, same "paste each `# Cell N` block" structure, same
`nnUNetv2_plan_and_preprocess` → `nnUNetv2_train` flow. Only the dataset name/id
and the pseudo-label source (public abdominal CT volumes with liver/kidney/
spleen/pancreas in view, e.g. Medical Segmentation Decathlon Task03_Liver or
Task09_Spleen images-only, run through `build_pseudo_labels_abdomen.py`)
change.

### `finetune_abdomen_nnunet.py`

Mirrors `finetune_chest_nnunet.py`: same
pretrain → `nnUNetv2_move_plans_between_datasets` → preprocess →
`nnUNetv2_train -pretrained_weights` sequence, same "can't add a 7th class"
caveat, same `<case>_ct.nii.gz`/`<case>_mask.nii.gz` correction-pair contract.
Imports `ABDOMEN_LABELS` from `abdomen_classes` instead of `CHEST_LABELS`.

### README.md

New "Abdominal CT Annotation" section, structured like the existing chest
section: what the script does, install step (same `uv sync` — no new
dependencies, TotalSegmentator's `total` task already covers all 6 organs),
usage examples (NIfTI, DICOM, `--fast --device cpu`), output files table,
common options table, and the same 4-step "Training Your Own 6-Class Model"
subsection (build pseudo-labels → train on Kaggle → run your model →
fine-tune on corrections) pointed at the new abdomen scripts/notebook.

## Non-goals

- No pathology/lesion classifier (kidney stone, tumor, etc.) — explicitly
  deferred per user's scope choice.
- No changes to any existing chest_* or brain reference file.
- No shared library extraction between chest and abdomen scripts.

## Testing

Each new `.py` file gets the same manual smoke check the chest script
implicitly relies on: `python abdomen_ct_annotation.py --help` parses, and
`build_pseudo_labels_abdomen.py --help` / `finetune_abdomen_nnunet.py --help`
parse without import errors. No local dataset with all 6 abdominal organs
in view is available for a full end-to-end run in this environment (no
GPU/TotalSegmentator weights fetch has been verified here), so the
implementation plan should call this out explicitly rather than claim a full
run was verified.
