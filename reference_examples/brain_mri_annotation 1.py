#!/usr/bin/env python3
"""
Generate a professionally annotated axial Brain Tumor MRI image.

Standalone Lambda-style worker (same shape as chest_ct_annotation.py).
Does not import the Inscribe website. Pack this file plus model weights.

Tumor  — MONAI brats_mri_segmentation (SegResNet). Never invents a mass.
Organs — SynthSeg v1 (lobes, ventricles, cerebellum). Skipped if weights/TF missing.

JSON contours use displayed radiological pixels: origin top-left, x right, y down.
Lambda can upload the output folder and fill annotated_image.uri / s3_path.

Container layout (example):
  /opt/brain-mri/brats_mri_segmentation/   # bundle: configs/ + models/model.pt
  /opt/brain-mri/SynthSeg/                 # BBillot/SynthSeg: models/ + data/

  python brain_mri_annotation.py \\
    --input /tmp/scan.nii.gz --output /tmp/out --device cpu \\
    --weights-dir /opt/brain-mri
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger("brain_mri_annotation")

# Set before TensorFlow/MONAI import on CPU Lambdas.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ORGAN_KEYS = (
    "left_frontal_lobe",
    "right_frontal_lobe",
    "left_parietal_lobe",
    "right_parietal_lobe",
    "ventricles",
    "cerebellum",
)
TUMOR_SUBKEYS = ("enhancing_tumor", "necrotic_core", "peritumoral_edema")
BRATS_CHANNEL_NAMES = ("t1ce", "t1", "t2", "flair")

DEFAULT_COLORS: Dict[str, str] = {
    "left_frontal_lobe": "#f1c40f",
    "right_frontal_lobe": "#e74c3c",
    "left_parietal_lobe": "#e67e22",
    "right_parietal_lobe": "#8e44ad",
    "ventricles": "#1e8449",
    "cerebellum": "#27ae60",
    "enhancing_tumor": "#af7ac5",
    "necrotic_core": "#f1c40f",
    "peritumoral_edema": "#00bcd4",
    "tumor": "#1abc9c",
}
DISPLAY_NAMES = {
    "left_frontal_lobe": "Left frontal lobe",
    "right_frontal_lobe": "Right frontal lobe",
    "left_parietal_lobe": "Left parietal lobe",
    "right_parietal_lobe": "Right parietal lobe",
    "ventricles": "Ventricles",
    "cerebellum": "Cerebellum",
    "enhancing_tumor": "Enhancing tumor",
    "necrotic_core": "Necrotic core",
    "peritumoral_edema": "Peritumoral oedema",
    "tumor": "Tumor",
}
PNG_DEFAULT_KEYS = (
    "left_frontal_lobe",
    "right_frontal_lobe",
    "left_parietal_lobe",
    "right_parietal_lobe",
    "ventricles",
    "tumor",
)
MASK_LABELS: Dict[str, int] = {
    "left_frontal_lobe": 1,
    "right_frontal_lobe": 2,
    "left_parietal_lobe": 3,
    "right_parietal_lobe": 4,
    "ventricles": 5,
    "cerebellum": 6,
    "enhancing_tumor": 7,
    "necrotic_core": 8,
    "peritumoral_edema": 9,
}

_HEMISPHERE_LABELS = (
    2, 3, 10, 11, 12, 13, 17, 18, 26, 28,
    41, 42, 49, 50, 51, 52, 53, 54, 58, 60,
)
_VENTRICLE_LABELS = (4, 5, 14, 15, 43, 44)
_CEREBELLUM_LABELS = (7, 8, 46, 47)
_LOBE_LEFT_FRONTAL = 1
_LOBE_RIGHT_FRONTAL = 2
_LOBE_LEFT_PARIETAL = 3
_LOBE_RIGHT_PARIETAL = 4


@dataclass
class LoadedMRI:
    image: Any
    display_3d: np.ndarray
    spacing: Tuple[float, float, float]
    input_type: str
    n_channels: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class WeightPaths:
    brats_bundle: Optional[Path]
    synthseg_home: Optional[Path]


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def is_nifti(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def volume_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.lower().endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for MONAI BraTS inference.") from exc

    value = requested.lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
            return "cpu"
        return "cuda"
    if value == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu"
    raise ValueError("--device must be auto, cuda, or cpu.")


def _looks_like_brats_bundle(path: Path) -> bool:
    return (path / "configs" / "inference.json").exists() and (
        (path / "models" / "model.pt").exists() or (path / "models" / "model.ts").exists()
    )


def _looks_like_synthseg(path: Path) -> bool:
    model = path / "models" / "synthseg_1.0.h5"
    labels = path / "data" / "labels_classes_priors" / "synthseg_segmentation_labels.npy"
    return model.exists() and model.stat().st_size > 1_000_000 and labels.exists()


def _find_brats_in(root: Path) -> Optional[Path]:
    for candidate in (
        root,
        root / "brats_mri_segmentation",
        root / "bundles" / "brats_mri_segmentation",
    ):
        if _looks_like_brats_bundle(candidate):
            return candidate.resolve()
    return None


def _find_synthseg_in(root: Path) -> Optional[Path]:
    for candidate in (
        root,
        root / "SynthSeg",
        root / "third_party" / "SynthSeg",
    ):
        if _looks_like_synthseg(candidate):
            return candidate.resolve()
    return None


def resolve_weights(
    weights_dir: Optional[str],
    brats_bundle: Optional[str],
    synthseg_home: Optional[str],
) -> WeightPaths:
    """Locate MONAI bundle + SynthSeg home for local runs and Lambda /opt packing."""
    search: List[Path] = []
    if weights_dir:
        search.append(Path(weights_dir).expanduser())
    if os.environ.get("BRAIN_MRI_WEIGHTS"):
        search.append(Path(os.environ["BRAIN_MRI_WEIGHTS"]))
    if os.environ.get("MONAI_BUNDLE_DIR"):
        search.append(Path(os.environ["MONAI_BUNDLE_DIR"]))
    if os.environ.get("SYNTHSEG_HOME"):
        search.append(Path(os.environ["SYNTHSEG_HOME"]))
    search.extend(
        [
            Path("/opt/brain-mri"),
            Path("/opt/ml/model"),
            Path("/opt/ml/weights"),
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent / "weights",
            Path.home() / "tao-projects" / "medical_image_annotation",
            Path("/Users/atul.bharadwaj/tao-projects/medical_image_annotation"),
        ]
    )

    brats = Path(brats_bundle).expanduser() if brats_bundle else None
    synth = Path(synthseg_home).expanduser() if synthseg_home else None
    if brats and not _looks_like_brats_bundle(brats):
        raise FileNotFoundError(f"--brats-bundle is not a MONAI BraTS bundle: {brats}")
    if synth and not _looks_like_synthseg(synth):
        raise FileNotFoundError(f"--synthseg-home is not a SynthSeg install: {synth}")

    if brats is None:
        for root in search:
            found = _find_brats_in(root)
            if found:
                brats = found
                break
    if synth is None:
        for root in search:
            found = _find_synthseg_in(root)
            if found:
                synth = found
                break
    return WeightPaths(brats_bundle=brats, synthseg_home=synth)


def load_mri(input_path: str) -> LoadedMRI:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to load MRI volumes.") from exc

    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if not path.is_file() or not is_nifti(path):
        raise ValueError(
            "Unsupported input. Supply a .nii/.nii.gz Brain Tumor MRI "
            "(4D T1ce/T1/T2/FLAIR, or a 3D volume for organs only)."
        )

    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not read NIfTI file '{path}': {exc}") from exc

    warnings: List[str] = []
    n_channels = 1
    if data.ndim == 4:
        if data.shape[-1] <= 8 and data.shape[-1] < min(data.shape[:3]):
            n_channels = int(data.shape[-1])
            display = data[..., 0]
        elif data.shape[0] <= 8 and data.shape[0] < min(data.shape[1:]):
            n_channels = int(data.shape[0])
            display = data[0]
        else:
            raise ValueError(f"Could not interpret MRI channel axis for shape {data.shape}.")
        if n_channels < 4:
            warnings.append(
                f"BraTS tumor model needs 4 modalities (T1ce, T1, T2, FLAIR); "
                f"found {n_channels}. Tumor segmentation will be skipped."
            )
    elif data.ndim == 3:
        display = data
        warnings.append(
            "Single-channel 3D MRI: SynthSeg organs may run; MONAI BraTS tumor "
            "requires a 4-channel T1ce/T1/T2/FLAIR stack."
        )
    else:
        raise ValueError(f"Expected a 3D or 4D NIfTI MRI, got shape {data.shape}.")

    display = np.nan_to_num(display, nan=0.0, posinf=0.0, neginf=0.0)
    if display.ndim != 3 or min(display.shape) < 2:
        raise ValueError(f"Invalid MRI dimensions: {display.shape}; expected a 3D volume.")
    if not np.any(np.isfinite(display)):
        raise ValueError("The MRI contains no finite voxel values.")

    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return LoadedMRI(
        image=image,
        display_3d=display.astype(np.float32),
        spacing=spacing,
        input_type="nifti",
        n_channels=n_channels,
        warnings=warnings,
    )


def write_canonical_nifti(source: Path, destination: Path) -> Path:
    import nibabel as nib

    destination.parent.mkdir(parents=True, exist_ok=True)
    image = nib.as_closest_canonical(nib.load(str(source)))
    nib.save(image, str(destination))
    return destination


def window_mri(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    pos = finite[finite > 0]
    sample = pos if pos.size >= 16 else finite
    if sample.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    low, high = np.percentile(sample, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32)


def to_radiological(array_xy: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(np.transpose(array_xy, (1, 0))))


def find_contours(mask_yx: np.ndarray) -> List[np.ndarray]:
    try:
        from skimage.measure import find_contours as sk_find_contours
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for contour extraction.") from exc
    contours: List[np.ndarray] = []
    for contour_rc in sk_find_contours(mask_yx.astype(np.uint8), 0.5):
        if len(contour_rc) >= 6:
            contours.append(np.column_stack((contour_rc[:, 1], contour_rc[:, 0])))
    return contours


def decimate_contour(contour: np.ndarray, maximum: int = 300) -> List[List[float]]:
    if len(contour) > maximum:
        indices = np.linspace(0, len(contour) - 1, maximum, dtype=int)
        contour = contour[indices]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in contour]


def bbox_for_mask(mask_yx: np.ndarray) -> Optional[List[int]]:
    rows, columns = np.where(mask_yx)
    if not len(rows):
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]


def resample_bool(mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise RuntimeError("scipy is required to resample masks.") from exc
    factors = [t / max(s, 1) for t, s in zip(target_shape, mask.shape)]
    return zoom(mask.astype(np.float32), factors, order=0) >= 0.5


def nested_brats_regions(lab: np.ndarray) -> Dict[str, np.ndarray]:
    u = {int(v) for v in np.unique(lab) if int(v) != 0}
    ncr = lab == 1
    ed = lab == 2
    if 4 in u or (3 not in u and 1 in u):
        et = lab == 4
    else:
        et = lab == 3
    return {
        "enhancing_tumor": et,
        "necrotic_core": ncr,
        "peritumoral_edema": ed,
    }


def cerebrum_lobe_codes(shape_xyz: Tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    import nibabel as nib

    sx, sy, sz = (max(int(s), 1) for s in shape_xyz)
    shape = (sx, sy, sz)
    axcodes = nib.aff2axcodes(np.asarray(affine))
    axis_fracs = (
        np.linspace(0.0, 1.0, sx, dtype=np.float32)[:, None, None],
        np.linspace(0.0, 1.0, sy, dtype=np.float32)[None, :, None],
        np.linspace(0.0, 1.0, sz, dtype=np.float32)[None, None, :],
    )
    lr = np.full(shape, 0.5, dtype=np.float32)
    ap = np.full(shape, 0.5, dtype=np.float32)
    for axis, code in enumerate(axcodes):
        g = axis_fracs[axis]
        if code == "R":
            lr[...] = g
        elif code == "L":
            lr[...] = 1.0 - g
        elif code == "A":
            ap[...] = g
        elif code == "P":
            ap[...] = 1.0 - g
    right = lr >= 0.5
    frontal = ap > 0.55
    codes = np.zeros(shape, dtype=np.uint8)
    codes[~right & frontal] = _LOBE_LEFT_FRONTAL
    codes[right & frontal] = _LOBE_RIGHT_FRONTAL
    codes[~right & ~frontal] = _LOBE_LEFT_PARIETAL
    codes[right & ~frontal] = _LOBE_RIGHT_PARIETAL
    return codes


def organs_from_synthseg_labels(arr: np.ndarray, affine: np.ndarray) -> Dict[str, np.ndarray]:
    codes = cerebrum_lobe_codes((int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])), affine)
    hemisphere = np.isin(arr, _HEMISPHERE_LABELS)
    ventricles = np.isin(arr, _VENTRICLE_LABELS)
    cerebellum = np.isin(arr, _CEREBELLUM_LABELS)
    claimed = ventricles | cerebellum
    return {
        "left_frontal_lobe": hemisphere & (codes == _LOBE_LEFT_FRONTAL) & ~claimed,
        "right_frontal_lobe": hemisphere & (codes == _LOBE_RIGHT_FRONTAL) & ~claimed,
        "left_parietal_lobe": hemisphere & (codes == _LOBE_LEFT_PARIETAL) & ~claimed,
        "right_parietal_lobe": hemisphere & (codes == _LOBE_RIGHT_PARIETAL) & ~claimed,
        "ventricles": ventricles,
        "cerebellum": cerebellum,
    }


def stage_brats_modalities(nifti_path: Path, staging_dir: Path) -> Tuple[Optional[List[Path]], Optional[str]]:
    import nibabel as nib

    staging_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj)
    affine = img.affine
    if data.ndim == 3:
        return None, (
            "BraTS needs 4 MRI modalities (T1ce, T1, T2, FLAIR). "
            f"Got a single-channel 3D volume {tuple(int(s) for s in data.shape)}."
        )
    if data.ndim != 4:
        return None, f"Unsupported NIfTI shape {tuple(int(s) for s in data.shape)} for BraTS."
    if data.shape[-1] <= 8 and data.shape[-1] < min(data.shape[:3]):
        channels = [data[..., i] for i in range(data.shape[-1])]
    elif data.shape[0] <= 8 and data.shape[0] < min(data.shape[1:]):
        channels = [data[i] for i in range(data.shape[0])]
    else:
        return None, f"Could not interpret channel axis for shape {tuple(int(s) for s in data.shape)}."
    if len(channels) < 4:
        return None, f"BraTS needs 4 modalities; found {len(channels)} channel(s)."
    channels = channels[:4]
    stem = volume_stem(nifti_path)
    out_paths: List[Path] = []
    for name, vol in zip(BRATS_CHANNEL_NAMES, channels):
        out = staging_dir / f"{stem}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.asanyarray(vol), affine), str(out))
        out_paths.append(out)
    return out_paths, None


def run_brats(
    nifti_path: Path,
    output_dir: Path,
    bundle_root: Path,
    device: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Return (label_volume, affine) or (None, None) plus warnings."""
    import nibabel as nib

    warnings: List[str] = []
    pred_dir = output_dir / "pred"
    cached = []
    if pred_dir.exists():
        cached = [
            p
            for p in sorted(pred_dir.rglob("*.nii*"))
            if "seg" in p.name.lower() and p.stat().st_size > 0
        ]
    if not cached:
        try:
            from monai.bundle import run as bundle_run
            import torch
        except ImportError as exc:
            return None, None, [f"MONAI/PyTorch unavailable ({exc}). Tumor skipped."]

        staging = output_dir / "_bundle_input"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        modality_paths, err = stage_brats_modalities(nifti_path, staging)
        if err or not modality_paths:
            return None, None, [err or "Failed to stage BraTS modalities. Tumor skipped."]

        datalist_path = staging / "datalist.json"
        datalist_path.write_text(
            json.dumps(
                {
                    "testing": [{"image": [str(p.resolve()) for p in modality_paths]}],
                    "training": [],
                    "validation": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        pred_dir.mkdir(parents=True, exist_ok=True)
        map_location = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"
        common: Dict[str, Any] = {
            "bundle_root": str(bundle_root),
            "meta_file": str(bundle_root / "configs" / "metadata.json"),
            "config_file": str(bundle_root / "configs" / "inference.json"),
            "dataset_dir": str(staging),
            "data_list_file_path": str(datalist_path),
            "output_dir": str(pred_dir),
            "dataloader#num_workers": 0,
            "dataloader#shuffle": False,
            "checkpointloader#map_location": map_location,
        }
        logging_file = bundle_root / "configs" / "logging.conf"
        if logging_file.exists():
            common["logging_file"] = str(logging_file)
        print("Running MONAI BraTS tumor segmentation...")
        try:
            bundle_run(**common)
        except TypeError:
            common.pop("dataloader#num_workers", None)
            common.pop("dataloader#shuffle", None)
            try:
                bundle_run(**common)
            except TypeError:
                common.pop("checkpointloader#map_location", None)
                bundle_run(**common)
        cached = [
            p
            for p in sorted(pred_dir.rglob("*.nii*"))
            if "seg" in p.name.lower() and p.stat().st_size > 0
        ]

    if not cached:
        return None, None, ["MONAI BraTS produced no tumor mask. Tumor was not invented."]
    img = nib.load(str(cached[0]))
    lab = np.asanyarray(img.dataobj)
    if lab.ndim > 3:
        lab = lab[..., 0]
    return lab.astype(np.int16), np.asarray(img.affine), warnings


def prepare_synthseg_input(image_path: Path, work_dir: Path, channel: int = 1) -> Path:
    import nibabel as nib

    work_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(image_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        if data.shape[-1] <= 8 and data.shape[-1] < min(data.shape[:3]):
            ch = min(channel, data.shape[-1] - 1)
            data3 = data[..., ch]
        else:
            data3 = data[0]
    elif data.ndim == 3:
        data3 = data
    else:
        raise ValueError(f"Unsupported MRI shape for SynthSeg: {data.shape}")
    out = work_dir / f"{volume_stem(image_path)}_synthseg_input.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ascontiguousarray(data3.astype(np.float32)), img.affine, img.header),
        str(out),
    )
    return out


def run_synthseg(
    nifti_path: Path,
    output_dir: Path,
    synthseg_home: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    warnings: List[str] = []
    seg_path = output_dir / "synthseg_labels.nii.gz"
    if not (seg_path.exists() and seg_path.stat().st_size > 0):
        try:
            import tensorflow as tf
        except Exception as exc:
            return None, None, [
                f"SynthSeg TensorFlow import failed ({type(exc).__name__}: {exc}). "
                "Use numpy 1.26.x with tensorflow-macos 2.15. Organs skipped."
            ]
        if str(synthseg_home) not in sys.path:
            sys.path.insert(0, str(synthseg_home))
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        try:
            from SynthSeg.predict_synthseg import predict
        except Exception as exc:
            return None, None, [
                f"SynthSeg Python package import failed ({type(exc).__name__}: {exc}). "
                "Clone BBillot/SynthSeg into --synthseg-home. Organs skipped."
            ]

        input_nii = prepare_synthseg_input(nifti_path, output_dir / "_input")
        labels_dir = synthseg_home / "data" / "labels_classes_priors"
        model = synthseg_home / "models" / "synthseg_1.0.h5"
        print("Running SynthSeg organ segmentation...")
        predict(
            path_images=str(input_nii),
            path_segmentations=str(seg_path),
            path_model_segmentation=str(model),
            labels_segmentation=str(labels_dir / "synthseg_segmentation_labels.npy"),
            robust=False,
            fast=True,
            v1=True,
            do_parcellation=False,
            n_neutral_labels=18,
            names_segmentation=str(labels_dir / "synthseg_segmentation_names.npy"),
            labels_denoiser=str(labels_dir / "synthseg_denoiser_labels_2.0.npy"),
            path_posteriors=None,
            path_resampled=None,
            path_volumes=str(output_dir / "synthseg_volumes.csv"),
            path_model_parcellation=str(synthseg_home / "models" / "synthseg_parc_2.0.h5"),
            labels_parcellation=str(labels_dir / "synthseg_parcellation_labels.npy"),
            names_parcellation=str(labels_dir / "synthseg_parcellation_names.npy"),
            path_model_qc=str(synthseg_home / "models" / "synthseg_qc_2.0.h5"),
            labels_qc=str(labels_dir / "synthseg_qc_labels.npy"),
            path_qc_scores=None,
            names_qc=str(labels_dir / "synthseg_qc_names.npy"),
            cropping=None,
            topology_classes=str(labels_dir / "synthseg_topological_classes.npy"),
            ct=False,
        )
    if not seg_path.exists():
        return None, None, ["SynthSeg finished without a label map. Organs were not invented."]
    import nibabel as nib

    img = nib.load(str(seg_path))
    arr = np.asanyarray(img.dataobj).astype(np.int32)
    if arr.ndim > 3:
        arr = arr[..., 0]
    return arr, np.asarray(img.affine), warnings


def combine_masks(
    target_shape: Tuple[int, ...],
    organ_masks: Mapping[str, np.ndarray],
    tumor_masks: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, int]]:
    combined = np.zeros(target_shape, dtype=np.uint16)
    used: Dict[str, int] = {}
    order = list(ORGAN_KEYS) + list(TUMOR_SUBKEYS)
    source = {**dict(organ_masks), **dict(tumor_masks)}
    for key in order:
        region = source.get(key)
        if region is None:
            continue
        region = resample_bool(region, target_shape)
        if not np.any(region):
            continue
        label = MASK_LABELS[key]
        combined[region] = label
        used[key] = label
    return combined, used


def region_on_plane(mask_plane: np.ndarray, key: str, labels: Mapping[str, int]) -> np.ndarray:
    if key == "tumor":
        region = np.zeros(mask_plane.shape, dtype=bool)
        for sub in ("enhancing_tumor", "necrotic_core"):
            lab = labels.get(sub)
            if lab is not None:
                region |= mask_plane == lab
        return region
    lab = labels.get(key)
    if lab is None:
        return np.zeros(mask_plane.shape, dtype=bool)
    return mask_plane == lab


def volume_region(mask: np.ndarray, key: str, labels: Mapping[str, int]) -> np.ndarray:
    if key == "tumor":
        region = np.zeros(mask.shape, dtype=bool)
        for sub in ("enhancing_tumor", "necrotic_core"):
            lab = labels.get(sub)
            if lab is not None:
                region |= mask == lab
        return region
    lab = labels.get(key)
    if lab is None:
        return np.zeros(mask.shape, dtype=bool)
    return mask == lab


def informative_slice(mask: np.ndarray, labels: Mapping[str, int]) -> int:
    best_slice, best_score = mask.shape[2] // 2, -1.0
    for z in range(mask.shape[2]):
        plane = mask[:, :, z]
        areas = [int(np.count_nonzero(region_on_plane(plane, key, labels))) for key in PNG_DEFAULT_KEYS]
        present = sum(area > 0 for area in areas)
        tumor = areas[-1] if areas else 0
        score = present * 1_000_000.0 + math.sqrt(tumor) * 1_000.0 + math.sqrt(sum(areas))
        if score > best_score:
            best_slice, best_score = z, score
    return best_slice


def choose_slice(requested: Optional[str], mask: np.ndarray, labels: Mapping[str, int]) -> int:
    if requested is None or requested.lower() == "auto":
        return informative_slice(mask, labels)
    if requested.lower() == "middle":
        return mask.shape[2] // 2
    try:
        index = int(requested)
    except ValueError as exc:
        raise ValueError("--slice must be an integer, 'middle', or 'auto'.") from exc
    if not 0 <= index < mask.shape[2]:
        raise ValueError(f"--slice {index} is outside the valid range 0..{mask.shape[2] - 1}.")
    return index


def hex_to_rgb(color: str) -> np.ndarray:
    value = color.lstrip("#")
    return np.asarray([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def label_position(
    key: str,
    centroid: Tuple[float, float],
    width: int,
    height: int,
    order: int,
) -> Tuple[float, float, str]:
    x, y = centroid
    if key == "ventricles":
        return width * 0.50, -height * 0.10, "center"
    if key == "cerebellum":
        return width * 0.50, height * 1.12, "center"
    if key in {"tumor", "enhancing_tumor", "necrotic_core", "peritumoral_edema"}:
        if x < width / 2:
            return -width * 0.10, min(height * 0.88, y), "right"
        return width * 1.10, min(height * 0.88, y), "left"
    if x < width / 2:
        return -width * 0.10, max(height * 0.12, min(height * 0.88, y + order * 4)), "right"
    return width * 1.10, max(height * 0.12, min(height * 0.88, y + order * 4)), "left"


def model_name_for_key(key: str) -> str:
    return "SynthSeg:v1" if key in ORGAN_KEYS else "MONAI:brats_mri_segmentation"


def build_slice_annotations(
    mask: np.ndarray,
    labels: Mapping[str, int],
    z: int,
    spacing: Sequence[float],
    colors: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    annotations: List[Dict[str, Any]] = []
    structures: Dict[str, Dict[str, Any]] = {}
    voxel_volume = float(np.prod(spacing))
    keys = list(labels.keys())
    if any(k in labels for k in ("enhancing_tumor", "necrotic_core")) and "tumor" not in keys:
        keys.append("tumor")
    for key in keys:
        volume_bin = volume_region(mask, key, labels)
        slice_region = to_radiological(volume_bin[:, :, z])
        contours = find_contours(slice_region) if np.any(slice_region) else []
        bbox = bbox_for_mask(slice_region)
        voxel_count = int(np.count_nonzero(volume_bin))
        display_name = DISPLAY_NAMES.get(key, key.replace("_", " ").title())
        source = model_name_for_key(key)
        structures[key] = {
            "name": display_name,
            "label": key,
            "model": source,
            "detected": voxel_count > 0,
            "visible_on_slice": bool(contours),
            "mask_available": voxel_count > 0,
            "contour_available": bool(contours),
            "slice_index": z,
            "bbox": bbox,
            "contour_points": [decimate_contour(item) for item in contours],
            "voxel_count": voxel_count,
            "volume_mm3": round(voxel_count * voxel_volume, 3),
            "confidence": None,
            "color": colors[key],
        }
        if contours:
            annotations.append(
                {
                    "id": f"annotation_{uuid.uuid4().hex[:12]}",
                    "class": display_name,
                    "label": key,
                    "type": "segmentation",
                    "slice_index": z,
                    "contours": [decimate_contour(item) for item in contours],
                    "bbox": bbox,
                    "mask_label_value": None if key == "tumor" else int(labels[key]),
                    "mask_file": "segmentation_mask.nii.gz",
                    "voxel_count_3d": voxel_count,
                    "confidence": None,
                    "source": source,
                    "status": "pending_review",
                    "editable": True,
                    "color": colors[key],
                }
            )
    return annotations, structures


def save_annotated_slice(
    destination: Path,
    mri_data: np.ndarray,
    mask: np.ndarray,
    labels: Mapping[str, int],
    z: int,
    colors: Mapping[str, str],
    alpha: float,
    visible_keys: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    mri_yx = to_radiological(window_mri(mri_data[:, :, z]))
    mask_yx = to_radiological(mask[:, :, z])
    height, image_width = mri_yx.shape
    fig, axis = plt.subplots(figsize=(12, 10), dpi=180, facecolor="#090d12")
    axis.set_facecolor("#090d12")
    axis.imshow(mri_yx, cmap="gray", vmin=0, vmax=1, origin="upper")
    visible: List[Tuple[str, Tuple[float, float], str]] = []
    for key in visible_keys:
        region = region_on_plane(mask_yx, key, labels)
        if not np.any(region):
            continue
        color = colors[key]
        rgba = np.zeros((height, image_width, 4), dtype=float)
        rgba[region, :3] = hex_to_rgb(color)
        rgba[region, 3] = alpha
        axis.imshow(rgba, origin="upper")
        contours = find_contours(region)
        for contour in contours:
            axis.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.7)
        largest = max(contours, key=len)
        centroid = (float(np.mean(largest[:, 0])), float(np.mean(largest[:, 1])))
        visible.append((key, centroid, color))
    for order, (key, centroid, color) in enumerate(visible):
        label_x, label_y, alignment = label_position(key, centroid, image_width, height, order)
        annotation = axis.annotate(
            DISPLAY_NAMES.get(key, key.replace("_", " ").title()),
            xy=centroid,
            xytext=(label_x, label_y),
            color="white",
            fontsize=11,
            fontweight="bold",
            ha=alignment,
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#111a24", edgecolor=color, alpha=0.96),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=1.6,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=20,
        )
        annotation.set_path_effects([path_effects.withStroke(linewidth=2, foreground="#000000")])
    axis.text(0.02, 1.02, "R", transform=axis.transAxes, color="white", fontsize=15, fontweight="bold", ha="left", va="bottom")
    axis.text(0.98, 1.02, "L", transform=axis.transAxes, color="white", fontsize=15, fontweight="bold", ha="right", va="bottom")
    axis.text(
        0.5, 1.02, f"AI-annotated brain MRI · axial slice {z}",
        transform=axis.transAxes, color="#d9e2ec", fontsize=11, ha="center", va="bottom",
    )
    axis.set_xlim(-image_width * 0.18, image_width * 1.18)
    axis.set_ylim(height * 1.16, -height * 0.16)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.savefig(destination, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def save_mask(path: Path, mask: np.ndarray, reference: Any) -> None:
    import nibabel as nib

    header = reference.header.copy()
    header.set_data_dtype(np.uint16)
    nib.save(nib.Nifti1Image(mask.astype(np.uint16), reference.affine, header=header), str(path))


def parse_visible_keys(raw: str) -> List[str]:
    if not raw.strip():
        return list(PNG_DEFAULT_KEYS)
    values = [v.strip().lower().replace(" ", "_") for v in raw.replace(";", ",").split(",") if v.strip()]
    aliases = {
        "left_frontal": "left_frontal_lobe",
        "right_frontal": "right_frontal_lobe",
        "left_parietal": "left_parietal_lobe",
        "right_parietal": "right_parietal_lobe",
        "oedema": "peritumoral_edema",
        "edema": "peritumoral_edema",
    }
    allowed = set(MASK_LABELS) | {"tumor"}
    out: List[str] = []
    for value in values:
        key = aliases.get(value, value)
        if key not in allowed:
            raise ValueError(f"Unknown --draw class '{value}'.")
        if key not in out:
            out.append(key)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment a Brain Tumor MRI and write a radiological annotated PNG, "
            "3D mask, and JSON contours (Lambda-style worker)."
        ),
        epilog=(
            "Pack: this .py + MONAI brats_mri_segmentation bundle + SynthSeg v1. "
            "Install: torch monai nibabel numpy==1.26.4 matplotlib scikit-image scipy "
            "tensorflow (or tensorflow-macos==2.15.0). "
            "Pass --weights-dir /opt/brain-mri on Lambda."
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="NIfTI file (3D or 4D BraTS stack).")
    parser.add_argument("--output", "-o", required=True, help="Output directory.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--weights-dir",
        help="Folder containing brats_mri_segmentation/ and SynthSeg/ (Lambda: /opt/brain-mri).",
    )
    parser.add_argument("--brats-bundle", help="Path to the MONAI brats_mri_segmentation bundle.")
    parser.add_argument("--synthseg-home", help="Path to the BBillot/SynthSeg checkout.")
    parser.add_argument(
        "--slice",
        default="auto",
        help="Axial slice index, 'middle', or 'auto' (default: auto).",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.18)
    parser.add_argument("--draw", default="", help="Comma-separated PNG classes.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1.")

    device = resolve_device(args.device)
    weights = resolve_weights(args.weights_dir, args.brats_bundle, args.synthseg_home)
    if weights.brats_bundle:
        print(f"BraTS bundle: {weights.brats_bundle}")
    else:
        LOGGER.warning("MONAI BraTS bundle not found. Tumor will be skipped.")
    if weights.synthseg_home:
        print(f"SynthSeg home: {weights.synthseg_home}")
    else:
        LOGGER.warning("SynthSeg install not found. Organs will be skipped.")

    print("Loading MRI...")
    original = Path(args.input).expanduser().resolve()
    load_mri(str(original))
    canonical_path = output_dir / "_work" / f"{volume_stem(original)}_ras.nii.gz"
    print("Writing RAS-canonical volume...")
    write_canonical_nifti(original, canonical_path)
    loaded = load_mri(str(canonical_path))
    mri_data = loaded.display_3d
    print(f"MRI display volume: {tuple(int(v) for v in mri_data.shape)}")
    print(f"MRI spacing: {tuple(round(v, 4) for v in loaded.spacing)} mm")
    print(f"Channels: {loaded.n_channels}")
    for warning in loaded.warnings:
        LOGGER.warning("Warning: %s", warning)

    visible_keys = parse_visible_keys(args.draw)
    warnings = list(loaded.warnings)
    organ_masks: Dict[str, np.ndarray] = {}
    tumor_masks: Dict[str, np.ndarray] = {}

    if int(mri_data.shape[2]) < 8:
        warnings.append(
            f"Series has only {mri_data.shape[2]} slice(s) — too thin for reliable "
            "BraTS/SynthSeg. No mask was invented."
        )
    else:
        work = output_dir / "_models"
        work.mkdir(parents=True, exist_ok=True)
        if weights.brats_bundle is not None:
            lab, _aff, brats_warn = run_brats(
                canonical_path, work / "brats", weights.brats_bundle, device
            )
            warnings.extend(brats_warn)
            if lab is not None:
                tumor_masks = nested_brats_regions(lab)
            else:
                warnings.append(
                    "No MONAI BraTS tumor mask. Need a 4-channel T1ce/T1/T2/FLAIR NIfTI "
                    "and models/model.pt. Tumor was not invented."
                )
        else:
            warnings.append("BraTS weights missing. Tumor was not invented.")

        if weights.synthseg_home is not None:
            ss_lab, ss_aff, ss_warn = run_synthseg(
                canonical_path, work / "synthseg", weights.synthseg_home
            )
            warnings.extend(ss_warn)
            if ss_lab is not None and ss_aff is not None:
                organ_masks = organs_from_synthseg_labels(ss_lab, ss_aff)
            else:
                warnings.append("No SynthSeg organ masks. Organs were not invented.")
        else:
            warnings.append("SynthSeg weights missing. Organs were not invented.")

    print("Combining organ + tumor masks...")
    mask, labels = combine_masks(mri_data.shape, organ_masks, tumor_masks)
    if not np.any(mask):
        raise RuntimeError(
            "No organ or tumor mask was produced. Check --weights-dir "
            "(brats_mri_segmentation + SynthSeg) and that the input is a full MRI."
        )

    selected_slice = choose_slice(args.slice, mask, labels)
    colors = dict(DEFAULT_COLORS)
    print("Generating contours...")
    annotations, structures_by_key = build_slice_annotations(
        mask, labels, selected_slice, loaded.spacing, colors
    )
    print("Generating visualization...")
    mask_path = output_dir / "segmentation_mask.nii.gz"
    save_mask(mask_path, mask, loaded.image)
    image_path = output_dir / "annotated_slice.png"
    save_annotated_slice(
        destination=image_path,
        mri_data=mri_data,
        mask=mask,
        labels=labels,
        z=selected_slice,
        colors=colors,
        alpha=args.overlay_alpha,
        visible_keys=visible_keys,
    )

    payload = {
        "schema_version": "1.0",
        "status": "completed",
        "image": image_path.name,
        "annotated_image": {
            "file": image_path.name,
            "path": str(image_path),
            "s3_path": None,
            "uri": None,
        },
        "slice": selected_slice,
        "slice_index": selected_slice,
        "input": str(original),
        "input_type": loaded.input_type,
        "orientation": {
            "display_convention": "radiological",
            "image_left": "patient_right",
            "image_right": "patient_left",
            "markers": {"left_edge": "R", "right_edge": "L"},
            "source_affine_ras": loaded.image.affine.tolist(),
        },
        "coordinate_system": {
            "origin": "top_left_of_displayed_MRI",
            "x_direction": "right",
            "y_direction": "down",
            "units": "pixels",
            "image_shape_yx": [int(mri_data.shape[1]), int(mri_data.shape[0])],
        },
        "model": {
            "anatomy": "SynthSeg:v1",
            "tumor": "MONAI:brats_mri_segmentation",
            "device": device,
            "hard_labels": True,
            "confidence_note": (
                "MONAI/SynthSeg hard-label output does not provide calibrated "
                "per-structure confidence; confidence is null rather than fabricated."
            ),
            "brats_bundle": str(weights.brats_bundle) if weights.brats_bundle else None,
            "synthseg_home": str(weights.synthseg_home) if weights.synthseg_home else None,
        },
        "volume": {
            "shape_xyz": [int(v) for v in mri_data.shape],
            "spacing_xyz_mm": [float(v) for v in loaded.spacing],
            "n_channels": loaded.n_channels,
            "mask_file": "segmentation_mask.nii.gz",
            "mask_labels": {key: int(value) for key, value in labels.items()},
        },
        "structures": list(structures_by_key.values()),
        "annotations": annotations,
        "overlays": [],
        "frontend_capabilities": {
            "review_status": "pending_review",
            "operations": ["accept", "reject", "edit_contour", "move", "change_class", "delete"],
        },
        "warnings": warnings,
    }
    json_path = output_dir / "annotations.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved:")
    print(image_path)
    print(mask_path)
    print(json_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        return run(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130
    except Exception as exc:
        LOGGER.error("Error: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
