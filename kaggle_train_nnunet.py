"""
Train nnU-Net v2 on KiTS19 inside a Kaggle notebook (free GPU).

Paste each `# %%` block into its own Kaggle notebook cell, in order.
Turn on internet access for the notebook first (Settings > Internet > On) —
this pulls segmentation labels from GitHub and CT volumes from Hugging Face,
no Kaggle Dataset attachment needed.
"""

# %% [Cell 1] Install nnU-Net v2
!pip install -q nnunetv2

# %% [Cell 2] Paths, dataset id, and case budget
# /kaggle/input is read-only, so nnU-Net's raw/preprocessed/results dirs must
# live under /kaggle/working (the only writable, persisted-as-output location).
import os
from pathlib import Path

WORK = Path("/kaggle/working")
KITS19_REPO = WORK / "kits19"  # git clone target; labels live under kits19/data/case_*/segmentation.nii.gz
DATASET_ID = "137"
DATASET_NAME = f"Dataset{DATASET_ID}_KiTS19"

os.environ["nnUNet_raw"] = str(WORK / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(WORK / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(WORK / "nnUNet_results")

for path in (os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"], os.environ["nnUNet_results"]):
    Path(path).mkdir(parents=True, exist_ok=True)

# Kaggle GPU sessions cap out around 9-12h, and each case's CT volume is a
# multi-hundred-MB download. Cap case count + use a faster config/trainer.
MAX_CASES = 60  # ponytail: raise this (up to 300) if you have a multi-session budget
CONFIGURATION = "2d"  # faster than 3d_lowres/3d_fullres; switch if you need 3D context
TRAINER = "nnUNetTrainer_100epochs"  # built into nnunetv2; drop to _50epochs/_20epochs if still too slow

# %% [Cell 3] Fetch KiTS19: labels via git clone, imaging volumes via Hugging Face
# The official repo's own download script (starter_code/get_imaging.py) always
# fetches all 300 cases with no subset option, so this reimplements just enough
# of it to respect MAX_CASES. Source verified against neheller/kits19 (2026-08).
import requests
from tqdm import tqdm

if not KITS19_REPO.exists():
    os.system(f"git clone --depth 1 https://github.com/neheller/kits19.git {KITS19_REPO}")

HF_BASE_URL = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main"

for i in range(MAX_CASES):
    case_id = f"case_{i:05d}"
    case_dir = KITS19_REPO / "data" / case_id
    imaging_path = case_dir / "imaging.nii.gz"
    if not case_dir.exists() or not (case_dir / "segmentation.nii.gz").exists():
        print(f"Skipping {case_id}: no segmentation label in the cloned repo.")
        continue
    if imaging_path.exists():
        continue

    response = requests.get(f"{HF_BASE_URL}/images/{case_id}.nii.gz", stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    tmp_path = imaging_path.with_suffix(".tmp")
    with tmp_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=case_id) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp_path.rename(imaging_path)

# %% [Cell 4] Convert downloaded cases into nnU-Net's raw dataset layout
import json
import shutil

raw_dataset_dir = Path(os.environ["nnUNet_raw"]) / DATASET_NAME
images_dir = raw_dataset_dir / "imagesTr"
labels_dir = raw_dataset_dir / "labelsTr"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

case_dirs = sorted(
    p for p in (KITS19_REPO / "data").glob("case_*")
    if (p / "imaging.nii.gz").exists() and (p / "segmentation.nii.gz").exists()
)[:MAX_CASES]
if not case_dirs:
    raise FileNotFoundError("No downloaded case has both imaging.nii.gz and segmentation.nii.gz — check Cell 3's output.")

for case_dir in case_dirs:
    case_id = case_dir.name  # e.g. "case_00000"
    shutil.copy(case_dir / "imaging.nii.gz", images_dir / f"{case_id}_0000.nii.gz")
    shutil.copy(case_dir / "segmentation.nii.gz", labels_dir / f"{case_id}.nii.gz")

dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "kidney": 1, "tumor": 2},
    "numTraining": len(case_dirs),
    "file_ending": ".nii.gz",
}
(raw_dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
print(f"Converted {len(case_dirs)} cases into {raw_dataset_dir}")

# %% [Cell 5-pretrained] Skip training: run nnU-Net v1's official KiTS pretrained model
# nnU-Net v2 (nnunetv2) ships NO pretrained models at all — confirmed against its
# current docs (2026-08): https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/run_inference_with_pretrained_models.md
# Pretrained weights only exist on the older nnU-Net v1 (`nnunet` package), which
# includes Task135_KiTS2021 (kidney+tumor+cyst) — an exact match for this task.
# This installs `nnunet` alongside `nnunetv2`; only needs Cell 1-4's raw imagesTr/
# (v1 uses the same `<case>_0000.nii.gz` input convention) — skip Cell 5/6/7.
!pip install -q nnunet
import subprocess

# nnU-Net v1 uses its own env var names (different from v2's nnUNet_raw/etc set
# in Cell 2) — without these it exits silently before even starting the download.
V1_WORK = WORK / "nnunet_v1"
os.environ["nnUNet_raw_data_base"] = str(V1_WORK / "raw_data_base")
os.environ["nnUNet_preprocessed"] = str(V1_WORK / "preprocessed")
os.environ["RESULTS_FOLDER"] = str(V1_WORK / "results")
for path in os.environ["nnUNet_raw_data_base"], os.environ["nnUNet_preprocessed"], os.environ["RESULTS_FOLDER"]:
    Path(path).mkdir(parents=True, exist_ok=True)

predictions_dir = WORK / "predictions_pretrained"
predictions_dir.mkdir(exist_ok=True)

# `nnUNet_download_pretrained_model` is broken as of 2026-08 — it hardcodes an
# HTTP/1.0 downgrade that Zenodo now 404s on (MIC-DKFZ/nnUNet#2876), so this
# downloads+extracts the same zip directly instead of shelling out to that CLI.
# Swap the URL for any other nnU-Net v1 pretrained task — same input convention
# (<case>_0000.nii.gz), just a different zip/id. Other abdominal CT options:
# Task003_Liver (liver+tumor), Task007_Pancreas (pancreas+tumor),
# Task008_HepaticVessel, Task010_Colon — all at zenodo.org/record/4003545.
PRETRAINED_MODEL_URL = "https://zenodo.org/record/5126443/files/Task135_KiTS2021.zip?download=1"
PRETRAINED_TASK_ID = "135"

zip_path = WORK / "pretrained_model.zip"
if not zip_path.exists():
    response = requests.get(PRETRAINED_MODEL_URL, stream=True, timeout=100)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with zip_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="pretrained model") as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))

import zipfile
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(os.environ["RESULTS_FOLDER"])

subprocess.run(
    [
        "nnUNet_predict",
        "-i", str(images_dir),
        "-o", str(predictions_dir),
        "-t", PRETRAINED_TASK_ID, "-m", "3d_fullres",
        "-f", "0",  # single fold instead of the full 5-fold ensemble; drop for full ensemble accuracy
    ],
    check=True,
)
print(f"Pretrained predictions saved to {predictions_dir}")

# %% [Cell 4b] Find and drop any case with an image/label shape mismatch
# "RuntimeError: Error while setting the slice" during preprocessing means one
# case's imaging.nii.gz and segmentation.nii.gz have different array shapes —
# usually a partial/interrupted download. This finds it directly with nibabel
# instead of relying on nnU-Net's own crash message.
import nibabel as nib

bad_cases = []
for img_path in sorted(images_dir.glob("*_0000.nii.gz")):
    case_id = img_path.name.removesuffix("_0000.nii.gz")
    label_path = labels_dir / f"{case_id}.nii.gz"
    img_shape = nib.load(img_path).shape
    label_shape = nib.load(label_path).shape
    if img_shape != label_shape:
        print(f"MISMATCH {case_id}: image {img_shape} vs label {label_shape}")
        bad_cases.append(case_id)

if bad_cases:
    for case_id in bad_cases:
        (images_dir / f"{case_id}_0000.nii.gz").unlink()
        (labels_dir / f"{case_id}.nii.gz").unlink()
    dataset_json["numTraining"] -= len(bad_cases)
    (raw_dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Removed {len(bad_cases)} bad case(s): {bad_cases}")
else:
    print("No shape mismatches found — re-run Cell 5, the RuntimeError may be from something else.")

# %% [Cell 5] Preprocess
# Only preprocess the config we'll actually train (skips wasted 3d_lowres/3d_fullres
# work), and cap worker processes — nnU-Net's default (8 for 2d) OOM'd on Kaggle's
# RAM, each worker holding a full CT volume in memory. Drop -np further if it OOMs again.
# subprocess.run(check=True) is used instead of `!shell` so a failure here raises
# and stops the script, instead of silently falling through into training on
# whatever partial data happened to finish preprocessing.
import subprocess

preprocessed_dir = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
if preprocessed_dir.exists():
    shutil.rmtree(preprocessed_dir)  # drop any stale/partial output from a prior failed run

subprocess.run(
    [
        "nnUNetv2_plan_and_preprocess", "-d", DATASET_ID,
        "-c", CONFIGURATION, "-np", "2",
        "--verify_dataset_integrity",
    ],
    check=True,
)

# %% [Cell 6] Train folds 0 and 1 in parallel, one per T4
# nnU-Net v2 has no built-in DDP for a single fold, so the productive use of a
# 2-GPU session is training two folds concurrently instead of one fold alone.
procs = []
for fold, gpu_id in enumerate([0, 1]):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    procs.append(subprocess.Popen(
        ["nnUNetv2_train", DATASET_ID, CONFIGURATION, str(fold), "-tr", TRAINER],
        env=env,
    ))

for p in procs:
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"nnUNetv2_train exited with code {p.returncode}")

# %% [Cell 7] Copy the trained model into Kaggle's persisted notebook output
output_model_dir = WORK / "trained_model"
shutil.copytree(
    Path(os.environ["nnUNet_results"]) / DATASET_NAME,
    output_model_dir,
    dirs_exist_ok=True,
)
print(f"Model saved to {output_model_dir} — download it from the notebook's Output tab.")
