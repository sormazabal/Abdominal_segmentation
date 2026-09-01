#!/usr/bin/env python3
"""
Build an nnU-Net v2 raw dataset of 6-class abdomen pseudo-labels.

Runs TotalSegmentator over a directory of CT volumes, collapses its output to
the 6 classes in abdomen_classes.ABDOMEN_LABELS (Right Lung, Left Lung,
Heart, Trachea, Aorta, Spine), and writes an nnUNet_raw/DatasetXXX_Name/
layout ready for `nnUNetv2_plan_and_preprocess`.

This is knowledge distillation, not ground truth: the student model this
trains can only approach TotalSegmentator's accuracy, never exceed it, until
some of its output is hand-corrected and fed back in via
finetune_abdomen_nnunet.py. See the project README for that loop.

Example:
    uv run python build_pseudo_labels_abdomen.py \\
        --input Dataset/LIDC-IDRI-0021 \\
        --output /kaggle/working/nnUNet_raw/Dataset503_AbdomenCT \\
        --device cpu --fast

--input accepts a directory containing one or more cases, where each case is
either a .nii/.nii.gz file or a subdirectory of DICOM files (the same two
input shapes abdomen_ct_annotation.py accepts) — or a single case path directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from abdomen_classes import BASE_MODEL_STRUCTURES, ABDOMEN_LABELS, combine_anatomical_masks
from abdomen_ct_annotation import (
    LOGGER,
    configure_logging,
    is_nifti,
    load_ct,
    resolve_device,
    run_anatomical_segmentation,
    save_mask,
)

# A class with zero voxels teaches the model that class doesn't exist. Two or
# more missing classes usually means the scan's field of view doesn't cover
# the full thorax (a common LIDC-IDRI/cropped-CT pattern) rather than the
# structure being genuinely absent, so such cases are dropped rather than fed
# in as-is.
MAX_MISSING_CLASSES = 1

ABDOMEN_CLASS_NAMES = [name for name in ABDOMEN_LABELS if name != "background"]


def discover_cases(input_dir: Path) -> List[Tuple[str, Path]]:
    """Find one case per top-level entry: a NIfTI file, or a DICOM directory."""
    if input_dir.is_file() or (input_dir.is_dir() and any(input_dir.glob("*.dcm"))):
        return [(input_dir.stem.removesuffix(".nii"), input_dir)]

    cases: List[Tuple[str, Path]] = []
    for entry in sorted(input_dir.iterdir()):
        if entry.is_file() and is_nifti(entry):
            case_id = entry.name
            for suffix in (".nii.gz", ".nii"):
                if case_id.lower().endswith(suffix):
                    case_id = case_id[: -len(suffix)]
                    break
            cases.append((case_id, entry))
        elif entry.is_dir():
            cases.append((entry.name, entry))
    return cases


def class_voxel_counts(mask: np.ndarray, labels: Dict[str, int]) -> Dict[str, int]:
    return {name: int(np.count_nonzero(mask == index)) for name, index in labels.items()}


def build_one_case(
    case_id: str,
    case_path: Path,
    images_dir: Path,
    labels_dir: Path,
    ts_device: str,
    fast: bool,
    verbose: bool,
) -> Tuple[bool, Dict[str, int]]:
    loaded = load_ct(str(case_path))
    for warning in loaded.warnings:
        LOGGER.warning("%s: %s", case_id, warning)

    raw_mask, label_map = run_anatomical_segmentation(
        image=loaded.image,
        model_structures=BASE_MODEL_STRUCTURES,
        ts_device=ts_device,
        fast=fast,
        verbose=verbose,
    )
    mask, labels = combine_anatomical_masks(raw_mask, label_map, extra_structures=[])
    counts = class_voxel_counts(mask, labels)

    missing = [name for name in ABDOMEN_CLASS_NAMES if counts.get(name, 0) == 0]
    if len(missing) > MAX_MISSING_CLASSES:
        LOGGER.warning(
            "%s: dropped, missing class(es) %s (field of view likely incomplete)",
            case_id,
            ", ".join(missing),
        )
        return False, counts

    import nibabel as nib

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    nib.save(loaded.image, str(images_dir / f"{case_id}_0000.nii.gz"))
    save_mask(labels_dir / f"{case_id}.nii.gz", mask, loaded.image)
    return True, counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True, help="Directory of cases, or a single case.")
    parser.add_argument("--output", "-o", required=True, help="nnUNet_raw dataset directory to create, e.g. .../Dataset503_AbdomenCT.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fast", action="store_true", help="Use 3 mm TotalSegmentator inference.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)
    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input does not exist: {input_dir}")

    output_dir = Path(args.output).expanduser().resolve()
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"

    _, ts_device = resolve_device(args.device)

    cases = discover_cases(input_dir)
    if not cases:
        raise FileNotFoundError(f"No cases (NIfTI files or DICOM directories) found under: {input_dir}")

    kept = 0
    dropped = 0
    rows: List[Tuple[str, Dict[str, int]]] = []
    for case_id, case_path in cases:
        print(f"[{case_id}] segmenting {case_path.name}...")
        try:
            ok, counts = build_one_case(
                case_id, case_path, images_dir, labels_dir, ts_device, args.fast, args.verbose
            )
        except Exception as exc:
            LOGGER.error("%s: skipped after error: %s", case_id, exc)
            dropped += 1
            continue
        rows.append((case_id, counts))
        if ok:
            kept += 1
        else:
            dropped += 1

    header = "case".ljust(24) + "".join(name.ljust(14) for name in ABDOMEN_CLASS_NAMES)
    print("\nVoxel counts per class (0 = class not found in this scan):")
    print(header)
    for case_id, counts in rows:
        print(case_id.ljust(24) + "".join(str(counts.get(name, 0)).ljust(14) for name in ABDOMEN_CLASS_NAMES))

    print(f"\n{kept} case(s) kept, {dropped} case(s) dropped.")
    if kept == 0:
        raise RuntimeError("No case passed the quality gate; no dataset was written.")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": ABDOMEN_LABELS,
        "numTraining": kept,
        "file_ending": ".nii.gz",
    }
    (output_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Dataset written to {output_dir}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
