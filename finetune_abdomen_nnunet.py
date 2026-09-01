#!/usr/bin/env python3
"""
Fine-tune a trained 6-class abdomen nnU-Net model on hand-corrected masks.

This closes the loop this project is built around: pseudo-label with
TotalSegmentator (build_pseudo_labels_abdomen.py) -> train (kaggle_train_abdomen_nnunet.ipynb)
-> review predictions in abdomen_ct_annotation.py's annotation tool -> correct
the mask -> fine-tune here on the correction -> repeat.

SCOPE: same 6 classes, new data only. nnU-Net requires a fine-tuning dataset
to share the base run's plans and preprocessing fingerprint, so this follows
its documented pretraining -> finetuning sequence (nnUNetv2_move_plans_between_datasets,
then -pretrained_weights), rather than a bare -pretrained_weights call, which
fails outright on a plans mismatch.

Adding a 7th class is NOT supported by this script or by nnU-Net's
fine-tuning path in general: the trained segmentation head has a fixed
number of output channels, and resizing it requires surgically editing the
checkpoint — nnU-Net has no built-in support for that. If you need new
classes, evaluate switching to a MONAI-based model instead, where swapping the
output head is a normal PyTorch operation.

Prerequisites: nnUNet_raw / nnUNet_preprocessed / nnUNet_results env vars set
(as in the training notebook), and the base model's dataset id known.

Example:
    uv run python finetune_abdomen_nnunet.py \\
        --corrected corrected_cases/ \\
        --base-dataset-id 503 \\
        --new-dataset-id 504 \\
        --fold 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from abdomen_classes import ABDOMEN_LABELS


def find_pretrained_checkpoint(results_root: Path, dataset_name: str, configuration: str, trainer: str, fold: int) -> Path:
    model_dir = results_root / dataset_name / f"{trainer}__nnUNetPlans__{configuration}"
    checkpoint = model_dir / f"fold_{fold}" / "checkpoint_final.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Train the base model first "
            f"(kaggle_train_abdomen_nnunet.ipynb), or check --trainer/--configuration."
        )
    return checkpoint


def build_raw_dataset(corrected_dir: Path, raw_dataset_dir: Path) -> int:
    """Copy <case>_ct.nii.gz / <case>_mask.nii.gz pairs into nnU-Net's layout.

    Expects `corrected_dir` to hold one CT + one corrected mask per case, named
    `<case>_ct.nii.gz` and `<case>_mask.nii.gz` — the pairing abdomen_ct_annotation.py's
    `segmentation_mask.nii.gz` output naturally falls into once you rename it
    per case and place it alongside the source CT.
    """
    images_dir = raw_dataset_dir / "imagesTr"
    labels_dir = raw_dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(corrected_dir.glob("*_ct.nii.gz"))
    if not cases:
        raise FileNotFoundError(
            f"No '*_ct.nii.gz' files under {corrected_dir}. Expected pairs of "
            "<case>_ct.nii.gz and <case>_mask.nii.gz."
        )

    count = 0
    for ct_path in cases:
        case_id = ct_path.name.removesuffix("_ct.nii.gz")
        mask_path = corrected_dir / f"{case_id}_mask.nii.gz"
        if not mask_path.exists():
            print(f"Skipping {case_id}: no matching {mask_path.name}", file=sys.stderr)
            continue
        shutil.copy(ct_path, images_dir / f"{case_id}_0000.nii.gz")
        shutil.copy(mask_path, labels_dir / f"{case_id}.nii.gz")
        count += 1

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": ABDOMEN_LABELS,
        "numTraining": count,
        "file_ending": ".nii.gz",
    }
    (raw_dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    return count


def print_validation_dice(results_root: Path, dataset_name: str, trainer: str, configuration: str, fold: int) -> None:
    """Print per-class Dice from nnU-Net's own held-out validation split.

    nnUNetv2_train writes this summary automatically after training finishes,
    so no extra scoring code is needed.
    """
    summary_path = (
        results_root / dataset_name / f"{trainer}__nnUNetPlans__{configuration}"
        / f"fold_{fold}" / "validation" / "summary.json"
    )
    if not summary_path.exists():
        print(f"No validation summary at {summary_path}.", file=sys.stderr)
        return
    summary = json.loads(summary_path.read_text())
    print(f"\n{'class':<12}{'Dice':>8}")
    for label, stats in summary["mean"].items():
        print(f"{label:<12}{stats['Dice']:>8.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrected", required=True, help="Directory of <case>_ct.nii.gz / <case>_mask.nii.gz pairs.")
    parser.add_argument("--base-dataset-id", required=True, help="nnU-Net dataset id of the already-trained base model, e.g. 503.")
    parser.add_argument("--new-dataset-id", required=True, help="New nnU-Net dataset id for the corrected data, e.g. 504.")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    parser.add_argument("--fold", type=int, default=0, help="Fold to fine-tune (fine-tuning trains one fold at a time).")
    parser.add_argument("--dry-run", action="store_true", help="Build the raw dataset and print commands without running them.")
    return parser


def run(args: argparse.Namespace) -> int:
    import os

    nnUNet_raw = Path(os.environ.get("nnUNet_raw", ""))
    nnUNet_preprocessed = Path(os.environ.get("nnUNet_preprocessed", ""))
    nnUNet_results = Path(os.environ.get("nnUNet_results", ""))
    if not (nnUNet_raw and nnUNet_preprocessed and nnUNet_results):
        raise RuntimeError(
            "nnUNet_raw, nnUNet_preprocessed, and nnUNet_results must be set "
            "(same env vars used by kaggle_train_abdomen_nnunet.ipynb)."
        )

    base_dataset_name = f"Dataset{args.base_dataset_id}_AbdomenCT"
    new_dataset_name = f"Dataset{args.new_dataset_id}_AbdomenCTCorrected"

    base_checkpoint = find_pretrained_checkpoint(
        nnUNet_results, base_dataset_name, args.configuration, args.trainer, args.fold
    )
    print(f"Base checkpoint: {base_checkpoint}")

    corrected_dir = Path(args.corrected).expanduser().resolve()
    raw_dataset_dir = nnUNet_raw / new_dataset_name
    count = build_raw_dataset(corrected_dir, raw_dataset_dir)
    print(f"Built {new_dataset_name} with {count} corrected case(s) at {raw_dataset_dir}")
    if count == 0:
        raise RuntimeError("No corrected case had both a CT and a mask; nothing to fine-tune on.")

    commands: List[List[str]] = [
        ["nnUNetv2_extract_fingerprint", "-d", args.new_dataset_id],
        [
            "nnUNetv2_move_plans_between_datasets",
            "-s", args.base_dataset_id, "-t", args.new_dataset_id,
            "-sp", "nnUNetPlans", "-tp", "nnUNetPlans",
        ],
        [
            "nnUNetv2_preprocess",
            "-d", args.new_dataset_id, "-plans_name", "nnUNetPlans", "-c", args.configuration,
        ],
        [
            "nnUNetv2_train", args.new_dataset_id, args.configuration, str(args.fold),
            "-tr", args.trainer,
            "-pretrained_weights", str(base_checkpoint),
        ],
    ]

    for command in commands:
        printable = " ".join(command)
        print(f"\n$ {printable}")
        if args.dry_run:
            continue
        # nnUNetv2_train prints per-epoch loss and pseudo-Dice to stdout as it
        # trains; not captured here so that progress streams live to the console.
        subprocess.run(command, check=True)

    if args.dry_run:
        print("\n--dry-run: no commands were executed.")
    else:
        fine_tuned_dir = (
            nnUNet_results / new_dataset_name
            / f"{args.trainer}__nnUNetPlans__{args.configuration}" / f"fold_{args.fold}"
        )
        print(f"\nFine-tuned checkpoint: {fine_tuned_dir / 'checkpoint_final.pth'}")
        print_validation_dice(nnUNet_results, new_dataset_name, args.trainer, args.configuration, args.fold)
        print(
            "\nRun abdomen_ct_annotation.py with --backend nnunet --nnunet-model-dir "
            f"{nnUNet_results / new_dataset_name / f'{args.trainer}__nnUNetPlans__{args.configuration}'}"
        )
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
