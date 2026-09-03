# Chest CT Annotation Script

## What This Script Does

`chest_ct_annotation.py` reads a 3D chest CT scan. It finds the lungs, heart, trachea, aorta, and spine with the TotalSegmentator model. It draws these structures on one CT slice and saves the result as an image.

You can also add your own nodule detector. The script never invents a nodule. It draws a nodule only when your detector finds one.

The script accepts two input types: a NIfTI file (`.nii` or `.nii.gz`) or a folder of DICOM files.

## Before You Start

This project uses `uv` to manage the Python environment. `uv` is a fast tool that creates the environment and installs the exact package versions for you.

### Step 1: Install uv

Open PowerShell. Run this command:

```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal. Then check that the install worked:

```
uv --version
```

### Step 2: Install the Project Environment

Open a terminal in the project folder. Run this command:

```
uv sync
```

This command creates a `.venv` folder and installs these packages at the versions listed in `pyproject.toml` and `uv.lock`:

- TotalSegmentator
- torch
- SimpleITK
- nibabel
- numpy
- matplotlib
- scikit-image
- pydicom

Note: The first segmentation run downloads model weights from the internet. Make sure that your network connection is active.

## How to Run the Script

Run the script with `uv run`. This command uses the project environment, so you do not need to activate it by hand.

### Example 1: Run on a NIfTI File

```
uv run python chest_ct_annotation.py --input Dataset\kits19\case_00000\imaging.nii.gz --output results
```

### Example 2: Run on a DICOM Folder

```
uv run python chest_ct_annotation.py --input Dataset\LIDC-IDRI-0021 --output results
```

### Example 3: Run Faster on a CPU

Add the `--fast` flag for a lower-resolution model. Add `--device cpu` when you have no GPU.

```
uv run python chest_ct_annotation.py --input Dataset\kits19\case_00000\imaging.nii.gz --output results --fast --device cpu
```

## What the Script Creates

The script writes three items to your output folder:

1. `annotated_slice.png` — one CT slice with colored outlines, labels, and leader lines.
2. `segmentation_mask.nii.gz` — the full 3D mask for all structures.
3. `annotations.json` — the structure list, contours, and metadata in JSON format.

The script also writes one overlay image per slice to an `overlays` subfolder.

## Common Options

| Flag | What It Does |
|---|---|
| `--input`, `-i` | Path to a NIfTI file or a DICOM folder. Required. |
| `--output`, `-o` | Path to the output folder. Required. |
| `--device` | Set to `auto`, `cuda`, or `cpu`. Default: `auto`. |
| `--fast` | Use the faster, lower-resolution model. |
| `--slice` | Set to a slice number, `middle`, or `auto`. Default: `auto`. |
| `--structures` | Add extra structure names from TotalSegmentator, separated by commas. |
| `--colors` | Set custom colors. Give a JSON object or a path to a JSON file. |
| `--enable-nodules` | Turn on nodule detection. Requires `--nodule-checkpoint`. |
| `--nodule-checkpoint` | Path to your TorchScript nodule detector file. |
| `--verbose`, `-v` | Show detailed log messages. |

For the full option list, run:

```
uv run python chest_ct_annotation.py --help
```

## How to Add Your Own Nodule Detector

You can supply a TorchScript model file with `--nodule-checkpoint`. The model must accept a tensor of shape `[1, 1, Z, Y, X]`, with CT values clipped to the range -1000 to 400 HU and scaled to -1 to 1.

The model must return rows in the format `[z, y, x, diameter_voxels, confidence]`. Turn on nodule detection with the `--enable-nodules` flag.

```
uv run python chest_ct_annotation.py --input Dataset\LIDC-IDRI-0021 --output results --enable-nodules --nodule-checkpoint path\to\model.pt
```

If you enable nodule detection without a checkpoint, the script skips nodule detection and writes a warning to the log and to `annotations.json`.

## Training Your Own 6-Class Model

By default the script segments with TotalSegmentator, a general-purpose 117-class model, and collapses its output down to the 6 chest classes shown above. You can instead train an owned nnU-Net v2 model that predicts exactly those 6 classes directly — smaller, faster at inference, and the only path that supports fine-tuning on your own corrected masks.

`chest_classes.py` holds the single definition of the 6 classes (`CHEST_LABELS`) shared by the annotation tool and every script below, so a trained model's label names always line up with the annotator's colors and display names.

### Step 1: Build pseudo-labels

There is no public dataset with ground-truth masks for exactly these 6 classes, so training data comes from running TotalSegmentator itself over chest CT volumes — the trained model is a *student* of TotalSegmentator, and its accuracy on scans like the ones it trained on approaches TotalSegmentator's own, but does not exceed it. That ceiling lifts once you fine-tune on hand-corrected masks (Step 4).

```
uv run python build_pseudo_labels.py --input <folder of NIfTI files or DICOM series> --output <nnUNet_raw>/Dataset501_ChestCT --device cpu --fast
```

Drop `--fast` and use `--device cuda` for full-resolution labels on a GPU. The script prints a per-case, per-class voxel count and drops any case missing more than one of the 6 classes (usually a cropped field of view) rather than silently training on it.

### Step 2: Train on Kaggle

`kaggle_train_chest_nnunet.ipynb` downloads 63 public chest CT volumes from the Medical Segmentation Decathlon (Task06_Lung, images only — its tumor labels are unused), runs Step 1 over them, and trains nnU-Net v2 on a Kaggle GPU session. Paste each `# Cell N` block into its own notebook cell, in order, with internet and a GPU accelerator turned on. It prints per-class Dice from nnU-Net's own validation split at the end.

Download the resulting `trained_model` folder from the notebook's Output tab.

### Step 3: Run your trained model

```
uv run python chest_ct_annotation.py --input Dataset\LIDC-IDRI-0021 --output results --backend nnunet --nnunet-model-dir <path>\nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
```

Output format (overlays, `segmentation_mask.nii.gz`, `annotations.json`) is identical either way; `annotations.json`'s `model`/`source` fields report whichever backend actually ran.

### Step 4: Fine-tune on corrections

Once you've reviewed some predictions and corrected the mask (e.g. by editing `segmentation_mask.nii.gz`), feed the corrections back in:

```
uv run python finetune_chest_nnunet.py --corrected <folder of case_ct.nii.gz/case_mask.nii.gz pairs> --base-dataset-id 501 --new-dataset-id 502
```

This supports the same 6 classes with new/corrected data only — nnU-Net's fine-tuning path cannot add a 7th class without surgically resizing the trained model's output head, which nnU-Net does not do for you. Adding new classes later would be the trigger to move to a MONAI-based model instead.

Note: `nnunetv2` is not pinned in this project's dependency file, so add it yourself before running any of the above (`uv add nnunetv2`).

## Troubleshooting

CAUTION: Do not run this script against production patient data without review by a qualified radiologist. The script output is not a medical diagnosis.

**"TotalSegmentator is unavailable."** Install the packages in the [Before You Start](#before-you-start) section.

**"CUDA ran out of memory."** Add the `--fast` flag, or run with `--device cpu`.

**"No readable DICOM series found."** Check that the input folder contains valid DICOM files. Each series needs at least two files.

**"Unknown TotalSegmentator structure name(s)."** Check the spelling of names given with `--structures`. Use only names from the TotalSegmentator `total` task.

# Abdominal CT Annotation Script

## What This Script Does

`abdomen_ct_annotation.py` reads a 3D CT scan and draws structures with the TotalSegmentator model, the same way `chest_ct_annotation.py` does. Its class set is the same six structures as the chest script — Right Lung, Left Lung, Heart, Trachea, Aorta, and Spine — by explicit project decision, not liver/kidney/spleen/pancreas.

The script accepts two input types: a NIfTI file (`.nii` or `.nii.gz`) or a folder of DICOM files.

## Before You Start

Same environment as the chest script — see [Before You Start](#before-you-start) above. No extra dependencies are needed.

## How to Run the Script

### Example 1: Run on a NIfTI File

```
uv run python abdomen_ct_annotation.py --input Dataset\kits19\case_00000\imaging.nii.gz --output results_abdomen
```

### Example 2: Run on a DICOM Folder

```
uv run python abdomen_ct_annotation.py --input Dataset\LIDC-IDRI-0021 --output results_abdomen
```

### Example 3: Run Faster on a CPU

```
uv run python abdomen_ct_annotation.py --input Dataset\kits19\case_00000\imaging.nii.gz --output results_abdomen --fast --device cpu
```

## What the Script Creates

Same three outputs as the chest script: `annotated_slice.png`, `segmentation_mask.nii.gz`, `annotations.json`, plus one overlay image per slice in an `overlays` subfolder.

## Common Options

| Flag | What It Does |
|---|---|
| `--input`, `-i` | Path to a NIfTI file or a DICOM folder. Required. |
| `--output`, `-o` | Path to the output folder. Required. |
| `--device` | Set to `auto`, `cuda`, or `cpu`. Default: `auto`. |
| `--fast` | Use the faster, lower-resolution model. |
| `--slice` | Set to a slice number, `middle`, or `auto`. Default: `auto`. |
| `--structures` | Add extra structure names from TotalSegmentator, separated by commas. |
| `--colors` | Set custom colors. Give a JSON object or a path to a JSON file. |
| `--verbose`, `-v` | Show detailed log messages. |

For the full option list, run:

```
uv run python abdomen_ct_annotation.py --help
```

## Training Your Own 6-Class Model

By default the script segments with TotalSegmentator and collapses its output down to the 6 classes shown above. You can instead train an owned nnU-Net v2 model that predicts exactly those 6 classes directly — smaller, faster at inference, and the only path that supports fine-tuning on your own corrected masks.

`abdomen_classes.py` holds the single definition of the 6 classes (`ABDOMEN_LABELS`) shared by the annotation tool and every script below, so a trained model's label names always line up with the annotator's colors and display names.

### Step 1: Build pseudo-labels

```
uv run python build_pseudo_labels_abdomen.py --input <folder of NIfTI files or DICOM series> --output <nnUNet_raw>/Dataset503_AbdomenCT --device cpu --fast
```

Drop `--fast` and use `--device cuda` for full-resolution labels on a GPU. The script prints a per-case, per-class voxel count and drops any case missing more than one of the 6 classes rather than silently training on it.

### Step 2: Train on Kaggle

`kaggle_train_abdomen_nnunet.ipynb` follows the same steps as `kaggle_train_chest_nnunet.ipynb` — downloads public CT volumes from the Medical Segmentation Decathlon, runs Step 1 over them, and trains nnU-Net v2 on a Kaggle GPU session. Paste each `# Cell N` block into its own notebook cell, in order, with internet and a GPU accelerator turned on.

Download the resulting `trained_model` folder from the notebook's Output tab.

### Step 2, alternative: Train locally with train_abdomen_nnunet.py

If you have your own CT data and a GPU on the machine you're already working on, `train_abdomen_nnunet.py` runs the same pipeline as Step 1 + Step 2 end to end — pseudo-label, preprocess, train, print per-class Dice — without going through Kaggle:

```
uv run python train_abdomen_nnunet.py --input <folder of NIfTI files or DICOM series> --dataset-id 503 --device cuda
```

It sets `nnUNet_raw`/`nnUNet_preprocessed`/`nnUNet_results` under `--work-dir` (default `nnUNet_work`) if those env vars aren't already set in your shell, and writes the trained checkpoint to `<work-dir>\nnUNet_results\Dataset503_AbdomenCT\nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth`.

#### Example: training on KITS19 volumes

`Dataset\kits19` holds raw KITS19 CT volumes. KITS19 is a kidney/tumor dataset, and this project's 6 classes are Right Lung, Left Lung, Heart, Trachea, Aorta, and Spine (no kidney) — so KITS19 only helps here as extra abdominal CT *volumes* for pseudo-labeling whichever of those 6 structures fall inside each scan's field of view, not as kidney training data. Download the full imaging data first (see the [official kits19 instructions](https://github.com/neheller/kits19)) — it ships as `data\case_00000\imaging.nii.gz` (plus a `segmentation.nii.gz` you don't need here) through `case_00299`, 300 cases in total.

`build_pseudo_labels_abdomen.py`'s case discovery expects one `.nii.gz` file (or one DICOM folder) per top-level entry under `--input`, so KITS19's nested `case_XXXXX\imaging.nii.gz` layout needs flattening first — hard links avoid duplicating the ~60GB of imaging data (same-volume hard links need no admin rights):

```powershell
$src = "Dataset\kits19\data"
$dest = "Dataset\kits19_flat"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

Get-ChildItem $src -Directory | ForEach-Object {
    $imaging = Join-Path $_.FullName "imaging.nii.gz"
    if (Test-Path $imaging) {
        $link = Join-Path $dest "$($_.Name).nii.gz"
        cmd /c mklink /H "`"$link`"" "`"$imaging`"" | Out-Null
    }
}

(Get-ChildItem $dest -Filter *.nii.gz).Count   # should print 300
```

Then train against the flattened folder:

```powershell
uv run python train_abdomen_nnunet.py --input Dataset\kits19_flat --dataset-id 503 --device cuda
```

(`--work-dir` defaults to `nnUNet_work`, matching the folder this pipeline already uses.)

Note: pseudo-labeling runs TotalSegmentator once per case, so 300 cases is real GPU time — hours, not minutes. To sanity-check the pipeline first, point `--input` at a subfolder holding a smaller slice of `kits19_flat` (e.g. its first 20-30 cases) before committing to the full run.

### Step 3: Run your trained model

```
uv run python abdomen_ct_annotation.py --input Dataset\LIDC-IDRI-0021 --output results_abdomen --backend nnunet --nnunet-model-dir <path>\nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
```

### Step 4: Fine-tune on corrections

Once Step 2 (or its local alternative) has produced a `checkpoint_final.pth` for `--base-dataset-id`, you can fine-tune it further on hand-corrected masks:

```
uv run python finetune_abdomen_nnunet.py --corrected <folder of case_ct.nii.gz/case_mask.nii.gz pairs> --base-dataset-id 503 --new-dataset-id 504
```

This supports the same 6 classes with new/corrected data only — same "can't add a 7th class" limit as the chest pipeline's fine-tuning script. `--base-dataset-id` must already have a checkpoint under `nnUNet_results` (from Step 2 or its local alternative above) or this fails with "No checkpoint at ...".

Both `train_abdomen_nnunet.py` and `finetune_abdomen_nnunet.py` need `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` set to the *same* paths across runs — if you trained locally with the default `--work-dir nnUNet_work`, set these in your shell before fine-tuning, or fine-tuning will look in the wrong (or an empty) place:

```powershell
$env:nnUNet_raw = "<repo path>\nnUNet_work\nnUNet_raw"
$env:nnUNet_preprocessed = "<repo path>\nnUNet_work\nnUNet_preprocessed"
$env:nnUNet_results = "<repo path>\nnUNet_work\nnUNet_results"
```

Note: `nnunetv2` is not pinned in this project's dependency file, so add it yourself before running any of the above (`uv add nnunetv2`).

# Running on a Remote GPU Machine

These steps set up and run either script on a shared Windows GPU machine reached over VPN + RDP.

## Step 1: Connect

1. Connect to the VPN.
2. Open Remote Desktop Connection and log in to the GPU machine.

## Step 2: Install uv

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal, then confirm:

```powershell
uv --version
```

## Step 3: Get the code onto the machine

If your repo is private and the machine isn't logged in to GitHub, use a personal access token:

```powershell
git clone https://<your-github-username>:<personal-access-token>@github.com/<owner>/<repo>.git
```

## Step 4: Install the environment

```powershell
uv sync
```

For the nnU-Net training/fine-tuning scripts, also run:

```powershell
uv add nnunetv2
```

## Step 5: Download your dataset from Kaggle

The GPU machine's login (e.g. `MLuser4`) may be shared with other people, so do not save your Kaggle credentials permanently — set them per-session instead:

```powershell
$env:KAGGLE_USERNAME = "<your-kaggle-username>"
$env:KAGGLE_KEY = "<your-kaggle-key>"
uv run kaggle datasets download -d <owner>/<dataset-name> -p .\Dataset --unzip
```

Get `<your-kaggle-key>` from kaggle.com → Settings → API → "Create New Token". If someone else's `kaggle.json` already exists at `C:\Users\<login>\.kaggle\kaggle.json`, do not delete it — that account is shared, so back it up first (`Rename-Item kaggle.json kaggle.json.bak`) before writing your own, and swap yours back out when you are done.

## Step 6: Run with GPU acceleration

Drop `--fast` and `--device cpu` — on this machine, use the GPU directly:

```powershell
uv run python chest_ct_annotation.py --input Dataset\kits19\case_00000\imaging.nii.gz --output results --device cuda
```

```powershell
uv run python abdomen_ct_annotation.py --input Dataset\kits19\case_00000\imaging.nii.gz --output results_abdomen --device cuda
```

Confirm the GPU is visible before a long run:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Step 7: Fine-tune on corrected masks

Once you have reviewed some predictions and corrected the masks, fine-tune directly on this machine's GPU instead of going back to Kaggle. Each corrected case needs a `case_ct.nii.gz`/`case_mask.nii.gz` pair in one folder.

For the chest model:

```powershell
uv run python finetune_chest_nnunet.py --corrected <folder of case_ct.nii.gz/case_mask.nii.gz pairs> --base-dataset-id 501 --new-dataset-id 502
```

For the abdomen model:

```powershell
uv run python finetune_abdomen_nnunet.py --corrected <folder of case_ct.nii.gz/case_mask.nii.gz pairs> --base-dataset-id 503 --new-dataset-id 504
```

Both require `nnunetv2` (Step 4) and the base dataset ID's training run to already exist under `nnUNet_raw`/`nnUNet_preprocessed` on this machine. Run the resulting model the same way as [Step 3](#step-3-run-your-trained-model) in the training sections above, pointing `--nnunet-model-dir` at the new dataset ID's output folder.

While training runs, `nnUNetv2_train` prints per-epoch loss and pseudo-Dice straight to the console — nothing extra to do to watch progress live. When it finishes, the script prints per-class Dice from nnU-Net's own held-out validation split, so you get a metrics summary without a separate evaluation step.
