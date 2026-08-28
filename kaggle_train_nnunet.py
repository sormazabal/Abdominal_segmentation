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
