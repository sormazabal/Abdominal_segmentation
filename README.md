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

## Troubleshooting

CAUTION: Do not run this script against production patient data without review by a qualified radiologist. The script output is not a medical diagnosis.

**"TotalSegmentator is unavailable."** Install the packages in the [Before You Start](#before-you-start) section.

**"CUDA ran out of memory."** Add the `--fast` flag, or run with `--device cpu`.

**"No readable DICOM series found."** Check that the input folder contains valid DICOM files. Each series needs at least two files.

**"Unknown TotalSegmentator structure name(s)."** Check the spelling of names given with `--structures`. Use only names from the TotalSegmentator `total` task.
