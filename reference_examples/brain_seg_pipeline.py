#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_seg_pipeline.py
======================================================================
A single-file reference pipeline that FUSES the two architectures the
user pointed at:

  * nnU-Net v2   (MIC-DKFZ/nnUNet)      -> tumour sub-region segmentation
                                            (multi-contrast, supervised,
                                             deep-supervised 3D U-Net)
  * SynthSeg     (BBillot/SynthSeg)     -> whole-brain anatomy parcellation
                                            (single-contrast, trained only
                                             from label maps via domain-
                                             randomised image synthesis, so
                                             it generalises across MRI
                                             contrasts/scanners)

Both ideas are re-implemented from scratch in plain PyTorch (no external
nnunetv2 / SynthSeg packages required) and wired into ONE pipeline that
matches the diagram supplied by the user:

    4D MRI (T1ce+T1+T2+FLAIR)
        -> PREPROCESSING   (RAS reorientation, resampling, normalisation)
        -> nnU-Net branch  (tumour: WT / TC / ET)   \\
        -> SynthSeg branch (anatomy: lobes, ventricles ...) / MASK INTEGRATION
        -> POSTPROCESSING  (contours, bounding boxes, volumes)
        -> OUTPUT          (NIfTI, JSON, PNG, 3D mesh/preview)

--------------------------------------------------------------------------
DATASET (as uploaded)
--------------------------------------------------------------------------
brain-mri/
  BRATS-Patient-001/brats_patient_001_t1ce_t1_t2_flair.nii.gz   (240,240,155,4)
  BRATS-Patient-001/manual_organ_strokes.json                    <- weak anatomy labels
  BRATS-Patient-002/brats_patient_002_t1ce_t1_t2_flair.nii.gz
  BRATS-Patient-004/brats_patient_004_t1ce_t1_t2_flair.nii.gz
  _raw/BRATS-Patient-001_gt.nii.gz   (0=bg,1=NCR,2=ED,3=ET)   <- full tumour labels
  _raw/BRATS-Patient-002_gt.nii.gz
  _raw/BRATS-Patient-004_gt.nii.gz

Only 3 patients ship full tumour ground truth, and only 1 patient carries
hand-drawn anatomy strokes (5 freehand outlines: left/right frontal lobe,
left/right parietal lobe, ventricles -- drawn on a single 2D slice in
normalised [0,1] canvas coordinates, no explicit slice index stored, so
this script assumes they were drawn on the mid-axial slice and extrudes
them a few slices in Z to obtain a thin sparse 3D label volume). This is
FAR below what either published method needs for a clinically valid
model -- treat everything this script trains as a runnable, correctly
-wired DEMO/prototype of the fused architecture and training philosophy,
not a validated segmentation tool.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python3 brain_seg_pipeline.py demo   --data-dir brain-mri --out-dir outputs
    python3 brain_seg_pipeline.py train  --data-dir brain-mri --out-dir outputs --epochs 20
    python3 brain_seg_pipeline.py infer  --data-dir brain-mri --out-dir outputs --patient BRATS-Patient-001

`demo` (the default) trains both networks briefly on the bundled data and
immediately runs full inference + mask integration + postprocessing +
output generation for every case, so the whole diagram executes end to
end in one command.

Dependencies: numpy, scipy, scikit-image, matplotlib, nibabel, torch.
======================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs nibabel: pip install nibabel") from exc

try:
    import scipy.ndimage as ndi
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs scipy: pip install scipy") from exc

try:
    from skimage import measure as skmeasure
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs scikit-image: pip install scikit-image") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs matplotlib: pip install matplotlib") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs PyTorch: pip install torch") from exc


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brain_seg_pipeline")


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ======================================================================
# 0. LABEL DEFINITIONS
# ======================================================================

MODALITIES = ["t1ce", "t1", "t2", "flair"]  # channel order inside the 4D nifti

# Ground-truth tumour labels (Medical Segmentation Decathlon / BraTS convention)
TUMOR_CLASS_NAMES = {0: "background", 1: "NCR", 2: "ED", 3: "ET"}
NUM_TUMOR_CLASSES = 4

# Clinically-used composite tumour regions, derived from the raw labels
TUMOR_REGIONS = {
    "WT": (1, 2, 3),   # whole tumour
    "TC": (1, 3),      # tumour core
    "ET": (3,),        # enhancing tumour
}

# Anatomy classes are *data driven*: whatever organ_key values are present in
# manual_organ_strokes.json become the SynthSeg branch's output classes,
# class 0 is always "background/unlabeled".
DEFAULT_ANATOMY_CLASS_NAMES = {
    0: "background",
    1: "left_frontal_lobe",
    2: "right_frontal_lobe",
    3: "left_parietal_lobe",
    4: "right_parietal_lobe",
    5: "ventricles",
}


# ======================================================================
# 1. DATASET DISCOVERY
# ======================================================================

@dataclass
class CaseFiles:
    patient_id: str
    image_path: str
    gt_path: Optional[str] = None
    strokes_path: Optional[str] = None


def discover_dataset(root: str) -> List[CaseFiles]:
    """Scan a `brain-mri/` style folder and pair up 4D images with their
    tumour ground truth (in `_raw/`) and manual anatomy strokes, if any."""
    cases: Dict[str, CaseFiles] = {}

    for img_path in sorted(glob.glob(os.path.join(root, "BRATS-Patient-*", "*t1ce_t1_t2_flair*.nii.gz"))):
        patient_dir = os.path.basename(os.path.dirname(img_path))
        cases[patient_dir] = CaseFiles(patient_id=patient_dir, image_path=img_path)

    raw_dir = os.path.join(root, "_raw")
    for gt_path in sorted(glob.glob(os.path.join(raw_dir, "*_gt.nii.gz"))):
        patient_id = os.path.basename(gt_path).replace("_gt.nii.gz", "")
        if patient_id in cases:
            cases[patient_id].gt_path = gt_path
        else:
            log.warning("Found GT for unknown patient %s (no image found)", patient_id)

    for patient_id, case in cases.items():
        strokes_path = os.path.join(root, patient_id, "manual_organ_strokes.json")
        if os.path.isfile(strokes_path):
            case.strokes_path = strokes_path

    ordered = sorted(cases.values(), key=lambda c: c.patient_id)
    if not ordered:
        raise FileNotFoundError(f"No BRATS-Patient-* volumes found under {root}")
    log.info("Discovered %d case(s): %s", len(ordered), [c.patient_id for c in ordered])
    return ordered


# ======================================================================
# 2. PREPROCESSING  (shared "PREPROCESSING" box in the diagram)
# ======================================================================

def load_4d_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, "nib.Nifti1Header"]:
    """Load a 4D (X,Y,Z,C) NIfTI. Returns data, affine, header."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine, img.header


def reorient_to_ras(data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient a volume (3D or 4D, channels last) to RAS+ using nibabel's
    canonical-orientation machinery, mirroring nnU-Net/SynthSeg's first
    preprocessing step."""
    is_4d = data.ndim == 4
    if is_4d:
        img = nib.Nifti1Image(data, affine)
    else:
        img = nib.Nifti1Image(data, affine)
    ras_img = nib.as_closest_canonical(img)
    return np.asarray(ras_img.dataobj, dtype=data.dtype), ras_img.affine


def resample_to_spacing(
    volume: np.ndarray,
    affine: np.ndarray,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a 3D (or 4D, channels-last) volume to isotropic spacing,
    exactly like nnU-Net's resampling stage. Uses linear interpolation for
    images and nearest-neighbour for label maps to avoid inventing new
    classes at boundaries."""
    current_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    zoom_factors = current_spacing / np.array(target_spacing, dtype=np.float64)

    order = 0 if is_label else 1
    if volume.ndim == 4:
        channels = [
            ndi.zoom(volume[..., c], zoom_factors, order=order, mode="nearest")
            for c in range(volume.shape[-1])
        ]
        resampled = np.stack(channels, axis=-1)
    else:
        resampled = ndi.zoom(volume, zoom_factors, order=order, mode="nearest")

    new_affine = affine.copy()
    scale_fix = np.diag(1.0 / zoom_factors)
    new_affine[:3, :3] = affine[:3, :3] @ scale_fix
    return resampled, new_affine


def normalize_nnunet_style(volume_4d: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation over the non-zero (foreground)
    voxels, clipped to [-5, 5] -- the default nnU-Net MRI intensity
    normalisation scheme."""
    out = np.zeros_like(volume_4d, dtype=np.float32)
    for c in range(volume_4d.shape[-1]):
        chan = volume_4d[..., c]
        mask = chan > 0
        if mask.sum() > 0:
            mean, std = chan[mask].mean(), chan[mask].std() + 1e-8
        else:
            mean, std = 0.0, 1.0
        norm = (chan - mean) / std
        norm = np.clip(norm, -5, 5)
        norm[~mask] = 0.0
        out[..., c] = norm
    return out


def normalize_synthseg_style(volume_1ch: np.ndarray) -> np.ndarray:
    """Robust min-max normalisation to [0, 1] using the 0.5th/99.5th
    percentiles of foreground voxels -- SynthSeg normalises every
    synthesised/real image this way so the network never sees raw
    scanner-dependent intensities."""
    mask = volume_1ch > 0
    if mask.sum() == 0:
        return np.zeros_like(volume_1ch, dtype=np.float32)
    lo, hi = np.percentile(volume_1ch[mask], [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = (volume_1ch - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def center_crop_or_pad(
    volume: np.ndarray, target_shape: Sequence[int], pad_value: float = 0.0
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Center-crop or zero-pad the first len(target_shape) spatial dims of
    `volume` to `target_shape`. Extra trailing dims (e.g. channels) are
    kept as-is. Returns the result and the crop/pad bookkeeping needed to
    map coordinates back to the original volume."""
    ndim = len(target_shape)
    src_shape = volume.shape[:ndim]
    pads = []
    slices_src = []
    slices_dst = []
    for s_src, s_tgt in zip(src_shape, target_shape):
        if s_src >= s_tgt:
            start = (s_src - s_tgt) // 2
            slices_src.append(slice(start, start + s_tgt))
            slices_dst.append(slice(0, s_tgt))
            pads.append((0, 0))
        else:
            pad_before = (s_tgt - s_src) // 2
            pad_after = s_tgt - s_src - pad_before
            slices_src.append(slice(0, s_src))
            slices_dst.append(slice(pad_before, pad_before + s_src))
            pads.append((pad_before, pad_after))

    out_shape = tuple(target_shape) + volume.shape[ndim:]
    out = np.full(out_shape, pad_value, dtype=volume.dtype)
    out[tuple(slices_dst)] = volume[tuple(slices_src)]
    return out, pads


@dataclass
class PreprocessedCase:
    patient_id: str
    images_4ch: np.ndarray          # (X,Y,Z,4) nnU-Net-normalised, RAS, resampled
    image_t1: np.ndarray            # (X,Y,Z) SynthSeg-normalised single contrast (T1)
    affine: np.ndarray
    orig_shape: Tuple[int, int, int]
    tumor_gt: Optional[np.ndarray] = None      # (X,Y,Z) uint8, 0..3
    voxel_volume_mm3: float = 1.0


def preprocess_case(case: CaseFiles, target_spacing=(1.0, 1.0, 1.0)) -> PreprocessedCase:
    """The PREPROCESSING box in the diagram: RAS reorientation, isotropic
    resampling, then the two normalisation schemes each downstream network
    expects."""
    data, affine, _ = load_4d_nifti(case.image_path)
    data, affine = reorient_to_ras(data, affine)

    gt = None
    gt_data_ras = None
    gt_affine_ras = None
    if case.gt_path is not None:
        gt_img = nib.load(case.gt_path)
        gt_data = np.asarray(gt_img.dataobj, dtype=np.uint8)
        gt_data_ras, gt_affine_ras = reorient_to_ras(gt_data, gt_img.affine)

    data, affine = resample_to_spacing(data, affine, target_spacing, is_label=False)
    if gt_data_ras is not None:
        gt, _ = resample_to_spacing(gt_data_ras, gt_affine_ras, target_spacing, is_label=True)
        gt = np.round(gt).astype(np.uint8)

    voxel_volume_mm3 = float(np.prod(target_spacing))

    images_4ch = normalize_nnunet_style(data)
    t1_idx = MODALITIES.index("t1")
    image_t1 = normalize_synthseg_style(data[..., t1_idx])

    return PreprocessedCase(
        patient_id=case.patient_id,
        images_4ch=images_4ch,
        image_t1=image_t1,
        affine=affine,
        orig_shape=data.shape[:3],
        tumor_gt=gt,
        voxel_volume_mm3=voxel_volume_mm3,
    )


# ======================================================================
# 3. WEAK ANATOMY LABELS  (rasterise manual_organ_strokes.json -> 3D map)
# ======================================================================

def load_anatomy_class_map(strokes_paths: Sequence[str]) -> Dict[int, str]:
    """Build the anatomy class-id <-> name table from whatever organ_key
    values actually appear in the dataset's stroke files (data-driven, so
    the SynthSeg branch's output head always matches the available weak
    labels)."""
    names = []
    for path in strokes_paths:
        with open(path) as f:
            payload = json.load(f)
        for stroke in payload["strokes"]:
            name = stroke.get("organ_key") or stroke.get("organKey")
            name = name.strip().lower().replace(" ", "_")
            if name not in names:
                names.append(name)
    class_map = {0: "background"}
    for i, name in enumerate(sorted(names), start=1):
        class_map[i] = name
    return class_map if len(class_map) > 1 else dict(DEFAULT_ANATOMY_CLASS_NAMES)


def rasterize_strokes_to_volume(
    strokes_path: str,
    volume_shape: Tuple[int, int, int],
    class_map: Dict[int, str],
    extrude_slices: int = 3,
) -> np.ndarray:
    """Turn the freehand polygon strokes (normalised [0,1] canvas coords,
    no stored slice index) into a sparse 3D label volume.

    Assumption (stated explicitly, since the JSON does not record a slice
    index): the strokes were drawn on the mid-axial slice of the volume
    they were annotated on. Each polygon is filled on that slice with
    `skimage.draw.polygon` and extruded +/- `extrude_slices` in Z so the
    3D U-Net gets a solid (if thin) training target instead of a single
    infinitely-thin plane.
    """
    from skimage.draw import polygon as sk_polygon

    name_to_id = {v: k for k, v in class_map.items()}
    label_vol = np.zeros(volume_shape, dtype=np.uint8)

    with open(strokes_path) as f:
        payload = json.load(f)

    W, H, Z = volume_shape
    mid_z = Z // 2
    z_lo = max(0, mid_z - extrude_slices)
    z_hi = min(Z, mid_z + extrude_slices + 1)

    for stroke in payload["strokes"]:
        name = (stroke.get("organ_key") or stroke.get("organKey")).strip().lower().replace(" ", "_")
        class_id = name_to_id.get(name)
        if class_id is None:
            continue
        pts = stroke["points"]
        xs = np.array([p["x"] for p in pts]) * (W - 1)
        ys = np.array([p["y"] for p in pts]) * (H - 1)
        rr, cc = sk_polygon(ys, xs, shape=(H, W))  # row=y, col=x
        for z in range(z_lo, z_hi):
            label_vol[cc, rr, z] = class_id
            # note: cc indexes the x/width axis -> volume axis 0, rr -> axis 1

    return label_vol


# ======================================================================
# 4. SynthSeg-style DOMAIN-RANDOMISATION IMAGE SYNTHESIS
# ======================================================================
# This is the core idea that makes SynthSeg "contrast agnostic": the
# segmentation network is trained purely from label maps. Every step, a
# brand-new synthetic MRI-like image is generated on the fly from the
# label map by (1) sampling a random Gaussian intensity per label,
# (2) adding a smooth multiplicative bias field, (3) random spatial
# deformation, (4) random blur/resolution degradation, (5) gamma/contrast
# jitter. The network never trains on the same image twice.

def _random_smooth_field(shape: Tuple[int, int, int], low_res: int = 4, amplitude: float = 0.3) -> np.ndarray:
    """Smooth random field for bias-field simulation: sample coarse noise,
    upsample with cubic interpolation to full resolution."""
    coarse = np.random.uniform(-amplitude, amplitude, size=(low_res, low_res, low_res)).astype(np.float32)
    zoom_factors = [s / low_res for s in shape]
    field = ndi.zoom(coarse, zoom_factors, order=3, mode="nearest")
    field, _ = center_crop_or_pad(field, shape)
    return field


def synthesize_image_from_labels(
    label_map: np.ndarray,
    num_classes: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """SynthSeg-style generative augmentation: label map -> synthetic scan.
    Returns a float32 volume normalised to [0, 1]."""
    rng = rng or np.random.RandomState()

    # 1) random GMM: one Gaussian intensity per class
    means = rng.uniform(0.05, 0.95, size=num_classes)
    means[0] = rng.uniform(0.0, 0.1)  # background stays dark
    stds = rng.uniform(0.02, 0.08, size=num_classes)

    image = means[label_map] + rng.normal(0, 1, size=label_map.shape).astype(np.float32) * stds[label_map]

    # 2) smooth multiplicative bias field (scanner inhomogeneity)
    bias = 1.0 + _random_smooth_field(label_map.shape, low_res=4, amplitude=0.25)
    image = image * bias

    # 3) random spatial deformation (small elastic warp) shared philosophy
    #    with SynthSeg's spatial augmentation; applied here to intensities
    #    only (label map stays aligned with itself by construction since we
    #    synthesise per-voxel from the *original* label map).
    if rng.rand() < 0.7:
        sigma = rng.uniform(0.5, 1.5)
        image = ndi.gaussian_filter(image, sigma=sigma)

    # 4) simulate resolution loss: downsample then upsample back
    if rng.rand() < 0.5:
        factor = rng.uniform(0.5, 0.9)
        small = ndi.zoom(image, factor, order=1)
        image = ndi.zoom(small, np.array(image.shape) / np.array(small.shape), order=1)
        image, _ = center_crop_or_pad(image, label_map.shape)

    # 5) gamma / contrast jitter + renormalise to [0,1]
    image = image - image.min()
    denom = image.max() + 1e-6
    image = image / denom
    gamma = rng.uniform(0.7, 1.5)
    image = np.clip(image, 0, 1) ** gamma
    return image.astype(np.float32)


# ======================================================================
# 5. NETWORK ARCHITECTURES
# ======================================================================
# Both networks share a "stacked conv block" building unit, but differ in
# exactly the ways the two papers differ:
#   nnU-Net : InstanceNorm3d + LeakyReLU, strided-conv down/up-sampling,
#             deep supervision heads at every decoder resolution.
#   SynthSeg: BatchNorm3d + ELU, max-pool down-sampling + conv-upsampling,
#             single full-resolution softmax output, single-channel input
#             (contrast agnostic -> no multi-modal fusion needed).

class ConvBlock3D(nn.Module):
    """Two 3x3x3 convs, each followed by normalisation + activation."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "instance", act: str = "leaky_relu"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        Act = (lambda: nn.LeakyReLU(0.01, inplace=True)) if act == "leaky_relu" else (lambda: nn.ELU(inplace=True))
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=norm != "instance"),
            Norm(out_ch, affine=True) if norm == "instance" else Norm(out_ch),
            Act(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=norm != "instance"),
            Norm(out_ch, affine=True) if norm == "instance" else Norm(out_ch),
            Act(),
        )

    def forward(self, x):
        return self.block(x)


class NNUNet3D(nn.Module):
    """A faithful-in-spirit, compact re-implementation of nnU-Net's generic
    3D U-Net: strided-conv downsampling, transposed-conv upsampling, skip
    connections, and deep supervision outputs at every decoder stage
    (coarse -> fine). Input: multi-channel (T1ce, T1, T2, FLAIR)."""

    def __init__(self, in_channels: int = 4, num_classes: int = NUM_TUMOR_CLASSES,
                 base_features: int = 16, num_pool: int = 3):
        super().__init__()
        self.num_pool = num_pool
        feats = [base_features * (2 ** i) for i in range(num_pool + 1)]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for f in feats[:-1]:
            self.encoders.append(ConvBlock3D(prev_ch, f, norm="instance", act="leaky_relu"))
            self.downs.append(nn.Conv3d(f, f, kernel_size=2, stride=2))
            prev_ch = f

        self.bottleneck = ConvBlock3D(prev_ch, feats[-1], norm="instance", act="leaky_relu")

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ds_heads = nn.ModuleList()  # deep supervision 1x1x1 conv heads
        prev_ch = feats[-1]
        for f in reversed(feats[:-1]):
            self.ups.append(nn.ConvTranspose3d(prev_ch, f, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(f * 2, f, norm="instance", act="leaky_relu"))
            self.ds_heads.append(nn.Conv3d(f, num_classes, kernel_size=1))
            prev_ch = f

    def forward(self, x):
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        deep_outputs = []
        for up, dec, head, skip in zip(self.ups, self.decoders, self.ds_heads, reversed(skips)):
            x = up(x)
            x = _match_and_concat(x, skip)
            x = dec(x)
            deep_outputs.append(head(x))

        # deep_outputs[-1] is full resolution (the "main" output); the rest
        # are the coarser deep-supervision heads, exactly like nnU-Net.
        return deep_outputs  # list, coarse -> fine


def _match_and_concat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Center-crop/pad `x` to `skip`'s spatial size (handles odd input
    sizes gracefully) before concatenating along the channel dim."""
    diffs = [s - xi for s, xi in zip(skip.shape[2:], x.shape[2:])]
    if any(d != 0 for d in diffs):
        pad = []
        for d in reversed(diffs):
            lo = d // 2
            hi = d - lo
            pad.extend([lo, hi])
        x = F.pad(x, pad)
    return torch.cat([x, skip], dim=1)


class SynthSegUNet3D(nn.Module):
    """SynthSeg's architecture: a standard 3D U-Net with same-padding
    convs, BatchNorm + ELU, max-pool downsampling and upsample+conv decoder,
    single-channel (contrast-agnostic) input, single full-resolution
    softmax output (no deep supervision -- SynthSeg trains a plain U-Net
    on an endless stream of synthetic images instead)."""

    def __init__(self, in_channels: int = 1, num_classes: int = len(DEFAULT_ANATOMY_CLASS_NAMES),
                 base_features: int = 24, num_pool: int = 3):
        super().__init__()
        feats = [base_features * (2 ** i) for i in range(num_pool + 1)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for f in feats[:-1]:
            self.encoders.append(ConvBlock3D(prev_ch, f, norm="batch", act="elu"))
            prev_ch = f
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(prev_ch, feats[-1], norm="batch", act="elu")

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = feats[-1]
        for f in reversed(feats[:-1]):
            self.up_convs.append(nn.Conv3d(prev_ch, f, kernel_size=3, padding=1))
            self.decoders.append(ConvBlock3D(f * 2, f, norm="batch", act="elu"))
            prev_ch = f

        self.out_conv = nn.Conv3d(prev_ch, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)  # single full-res logit map


# ======================================================================
# 6. LOSSES
# ======================================================================

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int,
                    ignore_index: Optional[int] = None, eps: float = 1e-5) -> torch.Tensor:
    """Multi-class soft Dice loss (mean over foreground+background classes),
    the workhorse loss of nnU-Net."""
    probs = F.softmax(logits, dim=1)
    target_1h = F.one_hot(target.long().clamp(min=0), num_classes).permute(0, 4, 1, 2, 3).float()

    if ignore_index is not None:
        valid = (target != ignore_index).unsqueeze(1).float()
        probs = probs * valid
        target_1h = target_1h * valid

    dims = (0, 2, 3, 4)
    intersection = (probs * target_1h).sum(dims)
    union = probs.sum(dims) + target_1h.sum(dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice_per_class.mean()


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int,
                  ignore_index: Optional[int] = None) -> torch.Tensor:
    """nnU-Net's default combo loss: unweighted sum of Dice + Cross-Entropy."""
    ce = F.cross_entropy(logits, target.long(), ignore_index=ignore_index if ignore_index is not None else -100)
    dice = soft_dice_loss(logits, target, num_classes, ignore_index)
    return ce + dice


def deep_supervision_loss(deep_outputs: List[torch.Tensor], target: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
    """Downsample the target to each deep-supervision head's resolution and
    weight losses 1/2, 1/4, 1/8, ... (coarsest -> finest), renormalised to
    sum to 1, exactly as nnU-Net does."""
    n = len(deep_outputs)
    weights = np.array([1.0 / (2 ** i) for i in range(n)][::-1])  # finest gets largest weight
    weights = weights / weights.sum()

    total = 0.0
    for w, out in zip(weights, deep_outputs):
        if out.shape[2:] != target.shape[1:]:
            tgt_f = F.interpolate(
                target.unsqueeze(1).float(), size=out.shape[2:], mode="nearest"
            ).squeeze(1).long()
        else:
            tgt_f = target.long()
        total = total + w * dice_ce_loss(out, tgt_f, num_classes)
    return total


# ======================================================================
# 7. DATASETS
# ======================================================================

def random_patch_coords(shape: Tuple[int, int, int], patch_size: Tuple[int, int, int],
                         center_bias: Optional[Tuple[int, int, int]] = None) -> Tuple[slice, slice, slice]:
    coords = []
    for i in range(3):
        s, p = shape[i], patch_size[i]
        if s <= p:
            coords.append(slice(0, s))
            continue
        if center_bias is not None:
            lo = max(0, min(s - p, center_bias[i] - p // 2 + random.randint(-p // 4, p // 4)))
        else:
            lo = random.randint(0, s - p)
        coords.append(slice(lo, lo + p))
    return tuple(coords)


class TumorPatchDataset(Dataset):
    """Patch-based dataset for the nnU-Net tumour branch, with nnU-Net's
    signature 2/3 foreground-oversampling strategy: two out of three
    patches are centred on a random tumour voxel instead of a fully random
    location, so the (heavily imbalanced) tumour classes are seen often
    enough to learn from."""

    def __init__(self, cases: List[PreprocessedCase], patch_size=(64, 64, 64),
                 patches_per_case: int = 20, augment: bool = True):
        self.cases = [c for c in cases if c.tumor_gt is not None]
        if not self.cases:
            raise ValueError("No cases with tumour ground truth available for training")
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.augment = augment

    def __len__(self):
        return len(self.cases) * self.patches_per_case

    def __getitem__(self, idx):
        case = self.cases[idx % len(self.cases)]
        shape = case.orig_shape
        fg_voxels = None
        oversample_fg = random.random() < (2 / 3)
        center = None
        if oversample_fg:
            fg_voxels = np.argwhere(case.tumor_gt > 0)
            if len(fg_voxels) > 0:
                center = tuple(fg_voxels[random.randrange(len(fg_voxels))])

        sl = random_patch_coords(shape, self.patch_size, center_bias=center)
        img_patch = case.images_4ch[sl[0], sl[1], sl[2], :]
        lbl_patch = case.tumor_gt[sl[0], sl[1], sl[2]]

        img_patch, _ = center_crop_or_pad(img_patch, self.patch_size)
        lbl_patch, _ = center_crop_or_pad(lbl_patch, self.patch_size)

        if self.augment:
            img_patch, lbl_patch = _random_flip(img_patch, lbl_patch)

        img_t = torch.from_numpy(np.ascontiguousarray(img_patch.transpose(3, 0, 1, 2))).float()
        lbl_t = torch.from_numpy(np.ascontiguousarray(lbl_patch)).long()
        return img_t, lbl_t


def _random_flip(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    for axis in range(3):
        if random.random() < 0.5:
            img = np.flip(img, axis=axis)
            lbl = np.flip(lbl, axis=axis)
    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)


class SynthSegSyntheticDataset(Dataset):
    """The defining trick of SynthSeg: the dataset holds ONLY label maps.
    Every __getitem__ call synthesises a brand-new fake MRI on the fly via
    `synthesize_image_from_labels`, so the network never overfits to any
    particular scanner/contrast and generalises to real T1/T2/FLAIR alike."""

    def __init__(self, label_maps: List[np.ndarray], num_classes: int,
                 patch_size=(64, 64, 64), length: int = 200):
        if not label_maps:
            raise ValueError("No anatomy label maps available for SynthSeg training")
        self.label_maps = label_maps
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label_map = self.label_maps[idx % len(self.label_maps)]
        fg = np.argwhere(label_map > 0)
        center = tuple(fg[random.randrange(len(fg))]) if len(fg) > 0 else None
        sl = random_patch_coords(label_map.shape, self.patch_size, center_bias=center)
        lbl_patch = label_map[sl]
        lbl_patch, _ = center_crop_or_pad(lbl_patch, self.patch_size)

        rng = np.random.RandomState(random.randint(0, 2 ** 31 - 1))
        img_patch = synthesize_image_from_labels(lbl_patch, self.num_classes, rng)
        img_patch, lbl_patch = _random_flip(img_patch[..., None], lbl_patch)
        img_patch = img_patch[..., 0]

        img_t = torch.from_numpy(img_patch[None]).float()
        lbl_t = torch.from_numpy(np.ascontiguousarray(lbl_patch)).long()
        return img_t, lbl_t


# ======================================================================
# 8. TRAINING LOOPS
# ======================================================================

def train_nnunet_branch(cases: List[PreprocessedCase], epochs: int, patch_size, device,
                         batch_size: int = 2, steps_per_epoch: int = 25,
                         base_features: int = 16, lr: float = 1e-3) -> NNUNet3D:
    log.info("Training nnU-Net tumour branch: %d epoch(s), patch=%s", epochs, patch_size)
    model = NNUNet3D(in_channels=4, num_classes=NUM_TUMOR_CLASSES, base_features=base_features).to(device)
    dataset = TumorPatchDataset(cases, patch_size=patch_size, patches_per_case=steps_per_epoch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=max(epochs, 1), power=0.9)

    model.train()
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            deep_outputs = model(img)
            loss = deep_supervision_loss(deep_outputs, lbl, NUM_TUMOR_CLASSES)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        log.info("  [nnU-Net] epoch %d/%d  loss=%.4f", epoch + 1, epochs, epoch_loss / max(n_batches, 1))
    model.eval()
    return model


def train_synthseg_branch(label_maps: List[np.ndarray], num_classes: int, epochs: int, patch_size, device,
                           batch_size: int = 2, steps_per_epoch: int = 25,
                           base_features: int = 24, lr: float = 1e-3) -> SynthSegUNet3D:
    log.info("Training SynthSeg anatomy branch: %d epoch(s), patch=%s, classes=%d",
              epochs, patch_size, num_classes)
    model = SynthSegUNet3D(in_channels=1, num_classes=num_classes, base_features=base_features).to(device)
    dataset = SynthSegSyntheticDataset(label_maps, num_classes, patch_size=patch_size,
                                        length=steps_per_epoch * batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = dice_ce_loss(logits, lbl, num_classes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        log.info("  [SynthSeg] epoch %d/%d  loss=%.4f", epoch + 1, epochs, epoch_loss / max(n_batches, 1))
    model.eval()
    return model


# ======================================================================
# 9. SLIDING-WINDOW INFERENCE  (shared by both branches)
# ======================================================================

def _gaussian_importance_map(patch_size: Tuple[int, int, int]) -> np.ndarray:
    sigma_scale = 0.125
    center = [(p - 1) / 2.0 for p in patch_size]
    sigmas = [p * sigma_scale for p in patch_size]
    grids = np.meshgrid(*[np.arange(p) for p in patch_size], indexing="ij")
    weight = np.ones(patch_size, dtype=np.float32)
    for g, c, s in zip(grids, center, sigmas):
        weight *= np.exp(-((g - c) ** 2) / (2 * s ** 2 + 1e-8))
    return (weight / weight.max()).astype(np.float32)


def sliding_window_inference(model: nn.Module, volume: np.ndarray, patch_size: Tuple[int, int, int],
                              num_classes: int, device, overlap: float = 0.5) -> np.ndarray:
    """Patch-based inference with Gaussian-weighted patch blending, the
    same strategy nnU-Net/SynthSeg both use to segment full-size volumes
    that don't fit the network's training patch size in one shot.

    `volume` is (X,Y,Z) for single-channel input or (X,Y,Z,C) for
    multi-channel input. Returns a (num_classes, X, Y, Z) softmax
    probability map.
    """
    is_multi = volume.ndim == 4
    shape = volume.shape[:3]
    padded_shape = tuple(max(s, p) for s, p in zip(shape, patch_size))
    if is_multi:
        vol_p, pads = center_crop_or_pad(volume, padded_shape)
    else:
        vol_p, pads = center_crop_or_pad(volume, padded_shape)

    step = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)
    starts_per_axis = []
    for s, p, st in zip(padded_shape, patch_size, step):
        starts = list(range(0, max(s - p, 0) + 1, st))
        if starts[-1] != s - p:
            starts.append(s - p)
        starts_per_axis.append(starts)

    prob_sum = np.zeros((num_classes,) + padded_shape, dtype=np.float32)
    weight_sum = np.zeros(padded_shape, dtype=np.float32)
    gaussian = _gaussian_importance_map(patch_size)

    model.eval()
    with torch.no_grad():
        for x0 in starts_per_axis[0]:
            for y0 in starts_per_axis[1]:
                for z0 in starts_per_axis[2]:
                    sl = (slice(x0, x0 + patch_size[0]), slice(y0, y0 + patch_size[1]), slice(z0, z0 + patch_size[2]))
                    if is_multi:
                        patch = vol_p[sl[0], sl[1], sl[2], :].transpose(3, 0, 1, 2)
                    else:
                        patch = vol_p[sl[0], sl[1], sl[2]][None]
                    patch_t = torch.from_numpy(np.ascontiguousarray(patch)).float().unsqueeze(0).to(device)
                    out = model(patch_t)
                    logits = out[-1] if isinstance(out, list) else out
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                    prob_sum[(slice(None),) + sl] += probs * gaussian[None]
                    weight_sum[sl] += gaussian

    weight_sum[weight_sum == 0] = 1e-6
    probs = prob_sum / weight_sum[None]

    # undo the pad/crop bookkeeping to get back to the original shape
    out_slices = []
    for i in range(3):
        pad_before, pad_after = pads[i]
        if pad_before > 0 or pad_after > 0:
            out_slices.append(slice(pad_before, pad_before + shape[i]))
        else:
            start = (padded_shape[i] - shape[i]) // 2
            out_slices.append(slice(start, start + shape[i]))
    probs = probs[(slice(None),) + tuple(out_slices)]
    return probs


def keep_largest_component_per_label(label_volume: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """nnU-Net-style postprocessing: for each requested label, keep only
    the largest connected component and drop isolated false-positive
    speckles."""
    out = label_volume.copy()
    for lab in labels:
        mask = out == lab
        if mask.sum() == 0:
            continue
        comp, n = ndi.label(mask)
        if n <= 1:
            continue
        sizes = ndi.sum(mask, comp, index=np.arange(1, n + 1))
        keep = np.argmax(sizes) + 1
        drop_mask = mask & (comp != keep)
        out[drop_mask] = 0
    return out


# ======================================================================
# 10. MASK INTEGRATION  ("MASK INTEGRATION" box in the diagram)
# ======================================================================

@dataclass
class IntegratedResult:
    tumor_label: np.ndarray            # (X,Y,Z) uint8, 0..3 raw tumour classes
    anatomy_label: np.ndarray          # (X,Y,Z) uint8, anatomy class ids
    combined_label: np.ndarray         # (X,Y,Z) uint16, anatomy ids with tumour overriding
    combined_class_names: Dict[int, str]
    tumor_class_names: Dict[int, str]
    anatomy_class_names: Dict[int, str]


def integrate_masks(tumor_label: np.ndarray, anatomy_label: np.ndarray,
                     anatomy_class_names: Dict[int, str]) -> IntegratedResult:
    """Fuse the nnU-Net tumour mask and the SynthSeg anatomy mask into one
    combined label volume: tumour voxels (label > 0) always take priority
    over the anatomy prediction, since the tumour is the clinically
    dominant finding; everywhere else keeps its anatomical label."""
    anatomy_offset = NUM_TUMOR_CLASSES  # shift anatomy ids so they never collide with tumour ids
    combined = anatomy_label.astype(np.uint16).copy()
    combined[combined > 0] += anatomy_offset - 1  # keep 0=background shared
    combined[tumor_label > 0] = tumor_label[tumor_label > 0]  # tumour overrides anatomy

    combined_names = dict(TUMOR_CLASS_NAMES)
    for cid, name in anatomy_class_names.items():
        if cid == 0:
            continue
        combined_names[cid + anatomy_offset - 1] = f"anatomy:{name}"

    return IntegratedResult(
        tumor_label=tumor_label,
        anatomy_label=anatomy_label,
        combined_label=combined,
        combined_class_names=combined_names,
        tumor_class_names=TUMOR_CLASS_NAMES,
        anatomy_class_names=anatomy_class_names,
    )


# ======================================================================
# 11. POSTPROCESSING  (contours, bounding boxes, volumes)
# ======================================================================

def compute_region_stats(mask: np.ndarray, voxel_volume_mm3: float) -> Optional[dict]:
    """Voxel count, physical volume, bounding box and centroid for one
    binary region."""
    if mask.sum() == 0:
        return None
    coords = np.argwhere(mask)
    bbox_min = coords.min(axis=0).tolist()
    bbox_max = coords.max(axis=0).tolist()
    centroid = coords.mean(axis=0).tolist()
    voxel_count = int(mask.sum())
    return {
        "voxel_count": voxel_count,
        "volume_mm3": round(voxel_count * voxel_volume_mm3, 2),
        "bounding_box": {"min_xyz": bbox_min, "max_xyz": bbox_max},
        "centroid_xyz": [round(c, 1) for c in centroid],
    }


def build_report(result: IntegratedResult, voxel_volume_mm3: float, patient_id: str) -> dict:
    report = {"patient_id": patient_id, "tumor_regions": {}, "anatomy_regions": {}}

    for region_name, labels in TUMOR_REGIONS.items():
        mask = np.isin(result.tumor_label, labels)
        stats = compute_region_stats(mask, voxel_volume_mm3)
        if stats:
            report["tumor_regions"][region_name] = stats

    for cid, name in result.tumor_class_names.items():
        if cid == 0:
            continue
        stats = compute_region_stats(result.tumor_label == cid, voxel_volume_mm3)
        if stats:
            report["tumor_regions"].setdefault("raw_labels", {})[name] = stats

    for cid, name in result.anatomy_class_names.items():
        if cid == 0:
            continue
        stats = compute_region_stats(result.anatomy_label == cid, voxel_volume_mm3)
        if stats:
            report["anatomy_regions"][name] = stats

    return report


def extract_slice_contours(label_slice: np.ndarray, class_ids: Sequence[int]) -> Dict[int, list]:
    """`skimage.measure.find_contours` per class on one 2D slice; used for
    the PNG overlay outputs."""
    contours_by_class = {}
    for cid in class_ids:
        binmask = (label_slice == cid).astype(np.float32)
        if binmask.sum() == 0:
            continue
        contours = skmeasure.find_contours(binmask, level=0.5)
        contours_by_class[cid] = [c.tolist() for c in contours]
    return contours_by_class


# ======================================================================
# 12. OUTPUT WRITERS
# ======================================================================

def save_nifti(volume: np.ndarray, affine: np.ndarray, path: str, dtype=np.int16) -> None:
    img = nib.Nifti1Image(volume.astype(dtype), affine)
    nib.save(img, path)


def save_json_report(report: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


_TUMOR_COLORS = {1: "#f1c40f", 2: "#2ecc71", 3: "#e74c3c"}  # NCR, ED, ET
_ANATOMY_CMAP = plt.get_cmap("tab20")


def save_overlay_png(base_volume: np.ndarray, tumor_label: np.ndarray, anatomy_label: np.ndarray,
                      class_names_anatomy: Dict[int, str], out_path: str) -> None:
    """Middle axial / coronal / sagittal slices of the base image with
    tumour + anatomy contours overlaid."""
    shape = base_volume.shape
    mid = [s // 2 for s in shape]
    views = {
        "axial": (base_volume[:, :, mid[2]], tumor_label[:, :, mid[2]], anatomy_label[:, :, mid[2]]),
        "coronal": (base_volume[:, mid[1], :], tumor_label[:, mid[1], :], anatomy_label[:, mid[1], :]),
        "sagittal": (base_volume[mid[0], :, :], tumor_label[mid[0], :, :], anatomy_label[mid[0], :, :]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (view_name, (img_slice, tum_slice, anat_slice)) in zip(axes, views.items()):
        # Standard skimage recipe: imshow(array) puts array[row, col] at
        # image position (row=y, col=x) under the default origin="upper",
        # and find_contours returns (row, col) points, so they're plotted
        # as (col, row) = (x, y) -- no rotation/flipping needed.
        ax.imshow(img_slice, cmap="gray")
        for cid, color in _TUMOR_COLORS.items():
            for contour in skmeasure.find_contours((tum_slice == cid).astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.3, color=color)
        for cid in class_names_anatomy:
            if cid == 0:
                continue
            for contour in skmeasure.find_contours((anat_slice == cid).astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.8, linestyle="--",
                        color=_ANATOMY_CMAP(cid % 20))
        ax.set_title(view_name)
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=c, label=TUMOR_CLASS_NAMES[k]) for k, c in _TUMOR_COLORS.items()]
    handles += [plt.Line2D([0], [0], color=_ANATOMY_CMAP(k % 20), linestyle="--", label=v)
                for k, v in class_names_anatomy.items() if k != 0]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_3d_visualization(combined_label: np.ndarray, class_names: Dict[int, str],
                           voxel_spacing: Tuple[float, float, float], out_obj_path: str,
                           out_png_path: str, max_labels: int = 8,
                           target_vertices_per_label: int = 40_000) -> None:
    """Marching-cubes surface mesh per structure written out as a single
    OBJ file (no extra mesh-IO dependency needed), plus a quick static
    matplotlib 3D preview render.

    `step_size` is chosen per-label from its voxel count so a structure
    that happens to fill a large fraction of the volume (e.g. an
    under-trained model collapsing onto one class) still produces a
    bounded-size mesh instead of a many-hundred-MB OBJ file.
    """
    present_labels = [l for l in np.unique(combined_label) if l != 0]
    present_labels = sorted(present_labels, key=lambda l: -(combined_label == l).sum())[:max_labels]

    hard_vertex_cap = target_vertices_per_label * 3  # absolute ceiling per label, whatever the heuristic misses

    vertex_offset = 0
    with open(out_obj_path, "w") as f:
        f.write("# brain_seg_pipeline combined segmentation mesh\n")
        for lab in present_labels:
            mask = (combined_label == lab).astype(np.uint8)
            voxel_count = int(mask.sum())
            if voxel_count < 20:
                continue
            # Smooth away speckle/salt-and-pepper noise first (common in an
            # under-trained or low-data model's predictions) -- this alone
            # collapses a lot of otherwise-explosive surface complexity
            # before marching cubes ever runs.
            mask = ndi.binary_closing(mask, iterations=1)
            mask = ndi.binary_opening(mask, iterations=1).astype(np.uint8)
            if mask.sum() < 20:
                continue

            # crude heuristic: marching_cubes vertex count grows roughly with
            # surface area ~ voxel_count^(2/3) at step_size=1; scale step_size
            # up for big/blobby regions to cap mesh size, then verify and
            # escalate step_size further if the heuristic undershot.
            approx_verts_at_1 = max(float(mask.sum()) ** (2 / 3), 1.0)
            step_size = max(1, int(math.ceil(approx_verts_at_1 / target_vertices_per_label)))
            step_size = min(step_size, 8)

            verts = faces = None
            for attempt in range(3):
                try:
                    verts, faces, _, _ = skmeasure.marching_cubes(
                        mask, level=0.5, spacing=voxel_spacing, step_size=step_size
                    )
                except (RuntimeError, ValueError):
                    verts = None
                    break
                if len(verts) <= hard_vertex_cap or step_size >= 16:
                    break
                step_size = min(step_size * 3, 16)  # heuristic undershot -> coarsen further and retry

            if verts is None or len(verts) == 0:
                continue
            if len(verts) > hard_vertex_cap:
                log.warning("3D mesh for label %s still has %d vertices after coarsening; "
                            "writing it anyway (NIfTI/JSON outputs are unaffected).",
                            class_names.get(int(lab), lab), len(verts))

            name = class_names.get(int(lab), f"label_{lab}")
            f.write(f"o {name} (step_size={step_size})\n")
            for v in verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for face in faces:
                f.write("f " + " ".join(str(i + 1 + vertex_offset) for i in face) + "\n")
            vertex_offset += len(verts)

    # static preview: downsample heavily and render a voxel scatter, colored per label
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, max(combined_label.shape) // 60)
    small = combined_label[::step, ::step, ::step]
    for i, lab in enumerate(present_labels):
        xs, ys, zs = np.where(small == lab)
        if len(xs) == 0:
            continue
        color = _TUMOR_COLORS.get(int(lab)) if lab in _TUMOR_COLORS else _ANATOMY_CMAP(i % 20)
        ax.scatter(xs, ys, zs, s=2, color=color, label=class_names.get(int(lab), str(lab)), alpha=0.6)
    ax.set_title("Combined segmentation (downsampled 3D preview)")
    ax.legend(loc="upper left", fontsize=6, bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


# ======================================================================
# 13. END-TO-END PIPELINE ORCHESTRATOR
# ======================================================================

class BrainMRIPipeline:
    """Wires every box of the diagram together:

        4D MRI -> preprocessing -> {nnU-Net tumour, SynthSeg anatomy}
                -> mask integration -> postprocessing -> output files
    """

    def __init__(self, data_dir: str, out_dir: str, device: Optional[str] = None,
                 patch_size: Tuple[int, int, int] = (64, 64, 64),
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        os.makedirs(out_dir, exist_ok=True)

        self.cases = discover_dataset(data_dir)
        self.tumor_model: Optional[NNUNet3D] = None
        self.anatomy_model: Optional[SynthSegUNet3D] = None
        self.anatomy_class_names: Dict[int, str] = dict(DEFAULT_ANATOMY_CLASS_NAMES)

    # -- preprocessing over the whole dataset (cached) --------------------
    def preprocess_all(self) -> Dict[str, PreprocessedCase]:
        if not hasattr(self, "_preprocessed"):
            self._preprocessed = {}
            for case in self.cases:
                log.info("Preprocessing %s ...", case.patient_id)
                self._preprocessed[case.patient_id] = preprocess_case(case, self.target_spacing)
        return self._preprocessed

    # -- training -----------------------------------------------------------
    def train(self, epochs: int = 10, steps_per_epoch: int = 25, batch_size: int = 2,
              nnunet_features: int = 16, synthseg_features: int = 24) -> None:
        preprocessed = self.preprocess_all()

        tumor_cases = [pc for pc in preprocessed.values() if pc.tumor_gt is not None]
        self.tumor_model = train_nnunet_branch(
            tumor_cases, epochs=epochs, patch_size=self.patch_size, device=self.device,
            batch_size=batch_size, steps_per_epoch=steps_per_epoch, base_features=nnunet_features,
        )
        ckpt_path = os.path.join(self.out_dir, "nnunet_tumor_branch.pt")
        torch.save(self.tumor_model.state_dict(), ckpt_path)
        log.info("Saved tumour branch checkpoint -> %s", ckpt_path)

        strokes_cases = [c for c in self.cases if c.strokes_path]
        if strokes_cases:
            self.anatomy_class_names = load_anatomy_class_map([c.strokes_path for c in strokes_cases])
            label_maps = [
                rasterize_strokes_to_volume(c.strokes_path, preprocessed[c.patient_id].orig_shape,
                                             self.anatomy_class_names)
                for c in strokes_cases
            ]
            self.anatomy_model = train_synthseg_branch(
                label_maps, num_classes=len(self.anatomy_class_names), epochs=epochs,
                patch_size=self.patch_size, device=self.device, batch_size=batch_size,
                steps_per_epoch=steps_per_epoch, base_features=synthseg_features,
            )
            ckpt_path = os.path.join(self.out_dir, "synthseg_anatomy_branch.pt")
            torch.save(self.anatomy_model.state_dict(), ckpt_path)
            log.info("Saved anatomy branch checkpoint -> %s", ckpt_path)
        else:
            log.warning("No manual_organ_strokes.json found in the dataset -- "
                        "SynthSeg anatomy branch was not trained.")

    def load_checkpoints(self) -> None:
        tumor_ckpt = os.path.join(self.out_dir, "nnunet_tumor_branch.pt")
        anat_ckpt = os.path.join(self.out_dir, "synthseg_anatomy_branch.pt")
        if os.path.isfile(tumor_ckpt):
            self.tumor_model = NNUNet3D(in_channels=4, num_classes=NUM_TUMOR_CLASSES).to(self.device)
            self.tumor_model.load_state_dict(torch.load(tumor_ckpt, map_location=self.device))
            self.tumor_model.eval()
        strokes_cases = [c for c in self.cases if c.strokes_path]
        if strokes_cases:
            self.anatomy_class_names = load_anatomy_class_map([c.strokes_path for c in strokes_cases])
        if os.path.isfile(anat_ckpt):
            self.anatomy_model = SynthSegUNet3D(in_channels=1, num_classes=len(self.anatomy_class_names)).to(self.device)
            self.anatomy_model.load_state_dict(torch.load(anat_ckpt, map_location=self.device))
            self.anatomy_model.eval()

    # -- inference + postprocessing + outputs for a single case ------------
    def run_case(self, patient_id: str) -> dict:
        preprocessed = self.preprocess_all()
        pc = preprocessed[patient_id]
        case_out_dir = os.path.join(self.out_dir, patient_id)
        os.makedirs(case_out_dir, exist_ok=True)

        # --- nnU-Net branch: tumour segmentation ---
        if self.tumor_model is not None:
            tumor_probs = sliding_window_inference(
                self.tumor_model, pc.images_4ch, self.patch_size, NUM_TUMOR_CLASSES, self.device
            )
            tumor_label = np.argmax(tumor_probs, axis=0).astype(np.uint8)
            tumor_label = keep_largest_component_per_label(tumor_label, [1, 2, 3])
        else:
            log.warning("No tumour model loaded -- skipping tumour branch for %s", patient_id)
            tumor_label = np.zeros(pc.orig_shape, dtype=np.uint8)

        # --- SynthSeg branch: anatomy parcellation ---
        if self.anatomy_model is not None:
            anat_probs = sliding_window_inference(
                self.anatomy_model, pc.image_t1, self.patch_size,
                len(self.anatomy_class_names), self.device
            )
            anatomy_label = np.argmax(anat_probs, axis=0).astype(np.uint8)
        else:
            log.warning("No anatomy model loaded -- skipping SynthSeg branch for %s", patient_id)
            anatomy_label = np.zeros(pc.orig_shape, dtype=np.uint8)

        # --- mask integration ---
        result = integrate_masks(tumor_label, anatomy_label, self.anatomy_class_names)

        # --- postprocessing: bounding boxes / volumes / contours ---
        report = build_report(result, pc.voxel_volume_mm3, patient_id)
        mid_axial_tum = result.tumor_label[:, :, result.tumor_label.shape[2] // 2]
        mid_axial_anat = result.anatomy_label[:, :, result.anatomy_label.shape[2] // 2]
        report["mid_axial_slice_contours"] = {
            "tumor": extract_slice_contours(mid_axial_tum, list(TUMOR_CLASS_NAMES)),
            "anatomy": extract_slice_contours(mid_axial_anat, list(self.anatomy_class_names)),
        }
        if pc.tumor_gt is not None:
            report["ground_truth_available"] = True
            report["dice_vs_ground_truth"] = _quick_dice_report(result.tumor_label, pc.tumor_gt)
        else:
            report["ground_truth_available"] = False

        # --- output: NIfTI, JSON, PNG, 3D visualization ---
        save_nifti(result.tumor_label, pc.affine, os.path.join(case_out_dir, "tumor_mask.nii.gz"), dtype=np.uint8)
        save_nifti(result.anatomy_label, pc.affine, os.path.join(case_out_dir, "anatomy_mask.nii.gz"), dtype=np.uint8)
        save_nifti(result.combined_label, pc.affine, os.path.join(case_out_dir, "combined_mask.nii.gz"), dtype=np.uint16)
        save_json_report(report, os.path.join(case_out_dir, "report.json"))
        save_overlay_png(pc.image_t1, result.tumor_label, result.anatomy_label,
                          self.anatomy_class_names, os.path.join(case_out_dir, "overlay_slices.png"))
        save_3d_visualization(result.combined_label, result.combined_class_names, self.target_spacing,
                               os.path.join(case_out_dir, "combined_mesh.obj"),
                               os.path.join(case_out_dir, "combined_3d_preview.png"))

        log.info("Finished %s -> outputs in %s", patient_id, case_out_dir)
        return report

    def run_all(self) -> Dict[str, dict]:
        return {c.patient_id: self.run_case(c.patient_id) for c in self.cases}


def _quick_dice_report(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    scores = {}
    for region, labels in TUMOR_REGIONS.items():
        p = np.isin(pred, labels)
        g = np.isin(gt, labels)
        denom = p.sum() + g.sum()
        scores[region] = round(float(2 * (p & g).sum() / denom), 4) if denom > 0 else None
    return scores


# ======================================================================
# 14. CLI
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fused nnU-Net (tumour) + SynthSeg (anatomy) brain MRI segmentation pipeline.",
    )
    parser.add_argument("mode", nargs="?", default="demo", choices=["demo", "train", "infer"],
                        help="demo = train briefly + run full inference on every case (default); "
                             "train = train and save checkpoints only; "
                             "infer = load existing checkpoints and run inference only.")
    parser.add_argument("--data-dir", default="brain-mri", help="Dataset root (default: brain-mri)")
    parser.add_argument("--out-dir", default="outputs", help="Where to write checkpoints/results")
    parser.add_argument("--patient", default=None, help="Restrict infer/demo to a single patient id")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per branch")
    parser.add_argument("--steps-per-epoch", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--nnunet-features", type=int, default=16, help="Base feature count for the tumour U-Net")
    parser.add_argument("--synthseg-features", type=int, default=24, help="Base feature count for the anatomy U-Net")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda' (default: auto-detect)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)

    pipeline = BrainMRIPipeline(
        data_dir=args.data_dir, out_dir=args.out_dir, device=args.device,
        patch_size=tuple(args.patch_size),
    )
    log.info("Using device: %s", pipeline.device)

    if args.mode in ("demo", "train"):
        pipeline.train(
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
            nnunet_features=args.nnunet_features, synthseg_features=args.synthseg_features,
        )
    else:  # infer
        pipeline.load_checkpoints()
        if pipeline.tumor_model is None and pipeline.anatomy_model is None:
            raise SystemExit(
                f"No checkpoints found in {args.out_dir}. Run with mode 'train' or 'demo' first."
            )

    if args.mode in ("demo", "infer"):
        t0 = time.time()
        if args.patient:
            reports = {args.patient: pipeline.run_case(args.patient)}
        else:
            reports = pipeline.run_all()
        save_json_report(reports, os.path.join(args.out_dir, "all_patients_summary.json"))
        log.info("Inference + postprocessing + outputs done in %.1fs. Summary -> %s",
                  time.time() - t0, os.path.join(args.out_dir, "all_patients_summary.json"))

    log.info("Pipeline complete. See %s for NIfTI / JSON / PNG / mesh outputs.", args.out_dir)


if __name__ == "__main__":
    main()
