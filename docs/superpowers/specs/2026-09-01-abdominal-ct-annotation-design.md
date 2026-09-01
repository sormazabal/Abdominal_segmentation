# Abdominal CT Annotation Pipeline — Design

## Goal

Replicate the chest CT pipeline (`chest_ct_annotation.py` + `chest_classes.py` +
`build_pseudo_labels.py` + `kaggle_train_chest_nnunet.ipynb` +
`finetune_chest_nnunet.py`), which itself replicates the brain MRI reference
pipeline pattern in `reference_examples/`, for abdominal CT.

## Scope

- Organ annotation only. No pathology classifier (no nodule/stone/tumor
  detector slot) in this pass.
- Class set: **user-confirmed to be exactly the 6 chest structures** —
  Right Lung, Left Lung, Heart, Trachea, Aorta, Spine — i.e. the same
  structures as `chest_classes.py`, not liver/kidney/spleen/pancreas. This was
  confirmed twice against a screenshot of that exact class list after I
  flagged that these are chest, not abdominal, structures.
  **Consequence:** with an identical class set and (see below) an identical
  HU window, `abdomen_ct_annotation.py`/`abdomen_classes.py` become a
  functional duplicate of `chest_ct_annotation.py`/`chest_classes.py` under
  new module names — no different organs are actually segmented. This is
  flagged here for visibility; proceeding per explicit, twice-confirmed
  instruction.
- Both pipeline layers, as separate new files — no shared/parametrized module
  with the chest scripts. The codebase's own convention (brain vs. chest) is
  full per-domain duplication, and the existing chest_* files stay untouched.

## Files

### `abdomen_classes.py`

Mirrors `chest_classes.py` with identical content (see Scope note above):

- `LEFT_LUNG_PARTS`, `RIGHT_LUNG_PARTS`, `THORACIC_VERTEBRAE`,
  `BASE_MODEL_STRUCTURES` — same TotalSegmentator `total`-task structure
  names and groupings as `chest_classes.py`.
- `ABDOMEN_LABELS = {"background": 0, "lung_right": 1, "lung_left": 2, "heart": 3, "trachea": 4, "aorta": 5, "vertebrae": 6}`
  — same keys/values as `CHEST_LABELS`, renamed constant only.
- `DEFAULT_COLORS` / `DISPLAY_NAMES` — same 6 entries as chest's.
- `combine_anatomical_masks(raw_mask, label_map, extra_structures)` — same
  body as chest's version (lung-lobe and thoracic-vertebrae grouping
  included), since the source structures are identical.

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
- `WINDOW_CENTER = -600.0`, `WINDOW_WIDTH = 1500.0` — same lung window as
  chest, since the structures being displayed (lungs, heart, trachea, aorta,
  spine) are the same.
- `informative_slice()` uses the same `("lung_right", "lung_left", "heart", "trachea", "aorta", "vertebrae")`
  preferred-structures tuple as chest's.
- `label_position()` keeps chest's `trachea`/`vertebrae`/`heart`
  special-case placements, unchanged.
- Imports from `abdomen_classes` instead of `chest_classes`.
- Docstring/log strings/JSON `"anatomy"` field say
  `"TotalSegmentator/total"` same as chest; file/CLI naming (`abdomen_*`) is
  the only distinguishing signal from the chest script, per Scope note above.

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
`nnUNetv2_plan_and_preprocess` → `nnUNetv2_train` flow, same pseudo-label
source data (chest CT volumes with lungs/heart/trachea/aorta/spine in view —
identical requirement to the chest notebook, since the class set is
identical). Only the dataset name/id and file names change.

### `finetune_abdomen_nnunet.py`

Mirrors `finetune_chest_nnunet.py`: same
pretrain → `nnUNetv2_move_plans_between_datasets` → preprocess →
`nnUNetv2_train -pretrained_weights` sequence, same "can't add a 7th class"
caveat, same `<case>_ct.nii.gz`/`<case>_mask.nii.gz` correction-pair contract.
Imports `ABDOMEN_LABELS` from `abdomen_classes` instead of `CHEST_LABELS`.

### README.md

New "Abdominal CT Annotation" section, structured like the existing chest
section: what the script does, install step (same `uv sync` — no new
dependencies), usage examples (NIfTI, DICOM, `--fast --device cpu`), output
files table, common options table, and the same 4-step "Training Your Own
6-Class Model" subsection (build pseudo-labels → train on Kaggle → run your
model → fine-tune on corrections) pointed at the new abdomen
scripts/notebook. Notes explicitly that the class set matches the chest
pipeline's (Right Lung, Left Lung, Heart, Trachea, Aorta, Spine) per your
confirmed decision.

## Non-goals

- No pathology/lesion classifier (kidney stone, tumor, etc.) — explicitly
  deferred per user's scope choice.
- No changes to any existing chest_* or brain reference file.
- No shared library extraction between chest and abdomen scripts.
- No actual abdominal organs (liver, kidney, spleen, pancreas) in this pass —
  explicitly superseded by the confirmed chest-identical class set.

## Testing

Each new `.py` file gets the same manual smoke check the chest script
implicitly relies on: `python abdomen_ct_annotation.py --help` parses, and
`build_pseudo_labels_abdomen.py --help` / `finetune_abdomen_nnunet.py --help`
parse without import errors. Running `abdomen_ct_annotation.py` end-to-end
against the repo's existing `Dataset/LIDC-IDRI-0021` chest CT (same input
`chest_ct_annotation.py` already validates against) is a reasonable
additional check, given the class set is chest-identical — but no
GPU/TotalSegmentator weights fetch has been verified in this environment, so
the implementation plan should call that out explicitly rather than claim a
full run was verified.
