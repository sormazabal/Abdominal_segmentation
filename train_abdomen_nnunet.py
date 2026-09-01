#!/usr/bin/env python3
"""
Train a 6-class abdomen nnU-Net v2 model on this machine, end to end.

Runs the full pipeline in one script: pseudo-label with TotalSegmentator
(build_pseudo_labels_abdomen.py) -> nnU-Net preprocess -> nnU-Net train ->
print per-class Dice from nnU-Net's own validation split.

Equivalent to kaggle_train_abdomen_nnunet.ipynb, but for a machine you already
have CT data and a GPU on: point --input at your own downloaded dataset
instead of pasting notebook cells and fetching the public MSD tarball.

Sets nnUNet_raw / nnUNet_preprocessed / nnUNet_results under --work-dir if
those env vars aren't already set.

Example:
    uv run python train_abdomen_nnunet.py \\
        --input Dataset\\abdominal-ct \\
        --dataset-id 503 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from finetune_abdomen_nnunet import print_validation_dice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True, help="Directory of CT cases (NIfTI files or DICOM subfolders).")
    parser.add_argument("--dataset-id", default="503", help="nnU-Net dataset id, e.g. 503.")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fast", action="store_true", help="Use 3 mm TotalSegmentator inference for pseudo-labeling.")
    parser.add_argument("--work-dir", default="nnUNet_work", help="Where nnUNet_raw/preprocessed/results live if those env vars aren't already set.")
    return parser


def run(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir).expanduser().resolve()
    nnUNet_raw = Path(os.environ.get("nnUNet_raw") or work_dir / "nnUNet_raw")
    nnUNet_preprocessed = Path(os.environ.get("nnUNet_preprocessed") or work_dir / "nnUNet_preprocessed")
    nnUNet_results = Path(os.environ.get("nnUNet_results") or work_dir / "nnUNet_results")
    for path in (nnUNet_raw, nnUNet_preprocessed, nnUNet_results):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(nnUNet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnUNet_preprocessed)
    os.environ["nnUNet_results"] = str(nnUNet_results)

    dataset_name = f"Dataset{args.dataset_id}_AbdomenCT"
    raw_dataset_dir = nnUNet_raw / dataset_name

    print(f"[1/3] Building pseudo-labels into {raw_dataset_dir} ...")
    pseudo_label_cmd = [
        sys.executable, str(Path(__file__).parent / "build_pseudo_labels_abdomen.py"),
        "--input", args.input,
        "--output", str(raw_dataset_dir),
        "--device", args.device,
    ]
    if args.fast:
        pseudo_label_cmd.append("--fast")
    subprocess.run(pseudo_label_cmd, check=True)

    print(f"\n[2/3] Preprocessing dataset {args.dataset_id} ...")
    subprocess.run(
        [
            "nnUNetv2_plan_and_preprocess", "-d", args.dataset_id,
            "-c", args.configuration, "--verify_dataset_integrity",
        ],
        check=True,
    )

    print(f"\n[3/3] Training fold {args.fold} ...")
    # nnUNetv2_train prints per-epoch loss and pseudo-Dice to stdout as it
    # trains; not captured here so that progress streams live to the console.
    subprocess.run(
        ["nnUNetv2_train", args.dataset_id, args.configuration, str(args.fold), "-tr", args.trainer],
        check=True,
    )

    model_dir = nnUNet_results / dataset_name / f"{args.trainer}__nnUNetPlans__{args.configuration}"
    print(f"\nTrained checkpoint: {model_dir / f'fold_{args.fold}' / 'checkpoint_final.pth'}")
    print_validation_dice(nnUNet_results, dataset_name, args.trainer, args.configuration, args.fold)
    print(f"\nRun abdomen_ct_annotation.py with --backend nnunet --nnunet-model-dir {model_dir}")
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
