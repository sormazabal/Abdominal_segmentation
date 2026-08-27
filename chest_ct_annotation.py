#!/usr/bin/env python3
"""
Generate a professionally annotated axial chest CT image.

Anatomical masks come from the pretrained TotalSegmentator "total" model.
Optional nodule detections come only from a user-supplied TorchScript detector;
this script never invents nodule locations.

The JSON contour coordinates use the displayed radiological CT pixel coordinate
system: origin at the top-left, x increasing rightward, y increasing downward.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger("chest_ct_annotation")
WINDOW_CENTER = -600.0
WINDOW_WIDTH = 1500.0

LEFT_LUNG_PARTS = ("lung_upper_lobe_left", "lung_lower_lobe_left")
RIGHT_LUNG_PARTS = (
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)
THORACIC_VERTEBRAE = tuple(f"vertebrae_T{i}" for i in range(1, 13))

BASE_MODEL_STRUCTURES = (
    *LEFT_LUNG_PARTS,
    *RIGHT_LUNG_PARTS,
    "heart",
    "trachea",
    "aorta",
    *THORACIC_VERTEBRAE,
)

DEFAULT_COLORS: Dict[str, str] = {
    "lung_right": "#ff9f1c",
    "lung_left": "#2d9cdb",
    "heart": "#ff5c8a",
    "trachea": "#ffd166",
    "aorta": "#ef233c",
    "vertebrae": "#f2f2f2",
    "nodule": "#2ee66b",
}

DISPLAY_NAMES = {
    "lung_right": "Right lung",
    "lung_left": "Left lung",
    "heart": "Heart",
    "trachea": "Trachea",
    "aorta": "Aorta",
    "vertebrae": "Spine",
    "nodule": "Nodule",
}


@dataclass
class LoadedCT:
    image: Any
    input_type: str
    warnings: List[str]


@dataclass
class NoduleDetection:
    center_xyz: Tuple[float, float, float]
    diameter_voxels: float
    confidence: float


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def is_nifti(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def looks_like_dicom(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required to inspect DICOM input.") from exc

    files = [item for item in path.rglob("*") if item.is_file() and not item.name.startswith(".")]
    if not files:
        return False
    for candidate in files[: min(16, len(files))]:
        try:
            pydicom.dcmread(str(candidate), stop_before_pixels=True)
            return True
        except Exception:
            continue
    return False


def load_dicom_series(path: Path) -> LoadedCT:
    """Read the largest DICOM series and convert its LPS geometry to NIfTI RAS."""
    try:
        import nibabel as nib
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK and nibabel are required for DICOM input.") from exc

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(path))
    if not series_ids:
        # A supplied directory may contain one nested study/series directory.
        candidate_dirs = sorted({item.parent for item in path.rglob("*") if item.is_file()})
        choices: List[Tuple[int, Path, str, Sequence[str]]] = []
        for directory in candidate_dirs:
            for series_id in reader.GetGDCMSeriesIDs(str(directory)) or []:
                names = reader.GetGDCMSeriesFileNames(str(directory), series_id)
                choices.append((len(names), directory, series_id, names))
        if not choices:
            raise ValueError(f"No readable DICOM series found under: {path}")
        _, series_dir, series_id, file_names = max(choices, key=lambda item: item[0])
        LOGGER.info("Using DICOM series %s in %s", series_id, series_dir)
    else:
        series_id = max(
            series_ids,
            key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(path), sid)),
        )
        file_names = reader.GetGDCMSeriesFileNames(str(path), series_id)

    if len(file_names) < 2:
        raise ValueError(
            "The selected DICOM series has fewer than two files; a 3D chest CT is required."
        )

    warnings = inspect_dicom_spacing(file_names)
    reader.SetFileNames(file_names)
    try:
        sitk_image = reader.Execute()
    except RuntimeError as exc:
        raise ValueError(f"SimpleITK could not decode the DICOM series: {exc}") from exc

    voxels_xyz = np.transpose(sitk.GetArrayFromImage(sitk_image), (2, 1, 0))
    spacing = np.asarray(sitk_image.GetSpacing(), dtype=float)
    direction_lps = np.asarray(sitk_image.GetDirection(), dtype=float).reshape(3, 3)
    origin_lps = np.asarray(sitk_image.GetOrigin(), dtype=float)

    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, :3] = direction_lps @ np.diag(spacing)
    affine_lps[:3, 3] = origin_lps
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    image = nib.Nifti1Image(voxels_xyz.astype(np.float32), affine_ras)
    image = nib.as_closest_canonical(image)
    return LoadedCT(image=image, input_type="dicom", warnings=warnings)


def inspect_dicom_spacing(file_names: Sequence[str]) -> List[str]:
    """Return useful warnings for obvious missing/nonuniform DICOM slices."""
    warnings: List[str] = []
    try:
        import pydicom

        positions: List[float] = []
        for name in file_names:
            ds = pydicom.dcmread(
                str(name),
                stop_before_pixels=True,
                specific_tags=["ImagePositionPatient", "ImageOrientationPatient"],
            )
            if hasattr(ds, "ImagePositionPatient"):
                position = np.asarray(ds.ImagePositionPatient, dtype=float)
                if hasattr(ds, "ImageOrientationPatient"):
                    orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
                    normal = np.cross(orientation[:3], orientation[3:])
                    positions.append(float(np.dot(position, normal)))
                else:
                    positions.append(float(position[2]))
        if len(positions) >= 3:
            gaps = np.diff(np.sort(np.asarray(positions)))
            positive = gaps[gaps > 1e-4]
            if positive.size:
                median = float(np.median(positive))
                if np.any(positive > median * 1.8):
                    warnings.append(
                        "Possible missing DICOM slice(s): a spacing gap is substantially larger "
                        "than the median slice spacing."
                    )
                if np.max(np.abs(positive - median)) > max(0.2, median * 0.2):
                    warnings.append("The DICOM series has nonuniform slice spacing.")
    except Exception as exc:
        LOGGER.debug("DICOM spacing validation was skipped: %s", exc)
    return warnings


def load_ct(input_path: str) -> LoadedCT:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to load CT volumes.") from exc

    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")

    if path.is_file() and is_nifti(path):
        try:
            image = nib.load(str(path))
            _ = image.get_fdata(dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Could not read NIfTI file '{path}': {exc}") from exc
        if len(image.shape) != 3:
            raise ValueError(f"Expected a 3D NIfTI CT, got shape {image.shape}.")
        return LoadedCT(
            image=nib.as_closest_canonical(image),
            input_type="nifti",
            warnings=[],
        )

    if path.is_dir() and looks_like_dicom(path):
        return load_dicom_series(path)

    raise ValueError(
        "Unsupported input. Supply a .nii/.nii.gz file or a directory containing a DICOM series."
    )


def validate_ct(image: Any) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3 or min(data.shape) < 2:
        raise ValueError(f"Invalid CT dimensions: {data.shape}; expected a 3D volume.")
    if not np.any(np.isfinite(data)):
        raise ValueError("The CT contains no finite voxel values.")
    finite = data[np.isfinite(data)]
    if float(np.max(finite) - np.min(finite)) < 1.0:
        raise ValueError("The CT has no meaningful intensity variation.")
    data = np.nan_to_num(data, nan=-1024.0, posinf=3071.0, neginf=-1024.0)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data, spacing


def resolve_device(requested: str) -> Tuple[str, str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model inference.") from exc

    value = requested.lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
            return "cpu", "cpu"
        return "cuda", "gpu"
    if value == "cpu":
        return "cpu", "cpu"
    raise ValueError("--device must be auto, cuda, or cpu.")


def parse_extra_structures(raw: str) -> List[str]:
    if not raw.strip():
        return []
    values = [value.strip() for value in raw.replace(";", ",").split(",") if value.strip()]
    return list(dict.fromkeys(values))


def run_anatomical_segmentation(
    image: Any,
    model_structures: Sequence[str],
    ts_device: str,
    fast: bool,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[int, str]]:
    try:
        import torch
        from totalsegmentator.map_to_binary import class_map
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise RuntimeError(
            "TotalSegmentator is unavailable. Install the dependencies shown in --help."
        ) from exc

    label_map = {int(index): str(name) for index, name in class_map["total"].items()}
    available = set(label_map.values())
    unknown = sorted(set(model_structures) - available)
    if unknown:
        raise ValueError(
            "Unknown TotalSegmentator structure name(s): "
            + ", ".join(unknown)
            + ". Use names from the TotalSegmentator 'total' task."
        )

    try:
        with torch.inference_mode():
            result = totalsegmentator(
                input=image,
                output=None,
                ml=True,
                task="total",
                roi_subset=list(model_structures),
                device=ts_device,
                fast=fast,
                quiet=not verbose,
                verbose=verbose,
                skip_saving=True,
            )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory. Retry with --device cpu or add --fast."
        ) from exc
    except Exception as exc:
        message = str(exc)
        if "download" in message.lower() or "weight" in message.lower():
            raise RuntimeError(
                "TotalSegmentator model weights are unavailable. Check the network/model "
                f"installation and retry. Details: {message}"
            ) from exc
        raise RuntimeError(f"TotalSegmentator inference failed: {message}") from exc

    data = np.asarray(result.dataobj, dtype=np.uint16)
    if data.shape != image.shape:
        raise RuntimeError(
            f"Segmentation geometry mismatch: CT {image.shape}, mask {data.shape}."
        )
    if not np.any(data):
        raise RuntimeError("TotalSegmentator returned an empty segmentation.")
    return data, label_map


def combine_anatomical_masks(
    raw_mask: np.ndarray,
    label_map: Mapping[int, str],
    extra_structures: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    name_to_source = {name: label for label, name in label_map.items()}
    output = np.zeros(raw_mask.shape, dtype=np.uint16)
    output_labels: Dict[str, int] = {
        "lung_right": 1,
        "lung_left": 2,
        "heart": 3,
        "trachea": 4,
        "aorta": 5,
        "vertebrae": 6,
    }

    groups: Dict[str, Iterable[str]] = {
        "lung_right": RIGHT_LUNG_PARTS,
        "lung_left": LEFT_LUNG_PARTS,
        "heart": ("heart",),
        "trachea": ("trachea",),
        "aorta": ("aorta",),
        "vertebrae": THORACIC_VERTEBRAE,
    }
    for key, source_names in groups.items():
        region = np.zeros(raw_mask.shape, dtype=bool)
        for source_name in source_names:
            source_label = name_to_source.get(source_name)
            if source_label is not None:
                region |= raw_mask == source_label
        output[region] = output_labels[key]

    base_components = set(BASE_MODEL_STRUCTURES)
    next_label = 7
    for name in extra_structures:
        if name in base_components:
            continue
        source_label = name_to_source[name]
        output_labels[name] = next_label
        output[raw_mask == source_label] = next_label
        next_label += 1
    return output, output_labels


def load_colors(raw: Optional[str], structure_keys: Sequence[str]) -> Dict[str, str]:
    colors = dict(DEFAULT_COLORS)
    if raw:
        source = Path(raw).expanduser()
        try:
            custom = json.loads(source.read_text(encoding="utf-8") if source.exists() else raw)
        except Exception as exc:
            raise ValueError(
                "--colors must be a JSON object or the path to a JSON file."
            ) from exc
        if not isinstance(custom, dict):
            raise ValueError("--colors must resolve to a JSON object.")
        colors.update({str(key): str(value) for key, value in custom.items()})

    palette = ("#8ac926", "#6a4c93", "#00b4d8", "#f9844a", "#90be6d", "#f72585")
    for index, key in enumerate(structure_keys):
        colors.setdefault(key, palette[index % len(palette)])
    return colors


def run_nodule_detector(
    enabled: bool,
    checkpoint: Optional[str],
    ct_data: np.ndarray,
    device: str,
) -> Tuple[List[NoduleDetection], Optional[str]]:
    if not enabled:
        return [], "Nodule detection was not requested."
    if not checkpoint:
        message = "Nodule detection is disabled because no nodule model/checkpoint was supplied."
        LOGGER.warning(message)
        return [], message

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        message = f"Nodule detection is disabled because the checkpoint was not found: {checkpoint_path}"
        LOGGER.warning(message)
        return [], message

    try:
        import torch

        model = torch.jit.load(str(checkpoint_path), map_location=device)
        model.eval()
        # Detector contract: input [1,1,Z,Y,X], clipped HU scaled to [-1,1].
        normalized = np.clip(ct_data, -1000.0, 400.0)
        normalized = ((normalized + 300.0) / 700.0).astype(np.float32)
        tensor = torch.from_numpy(np.transpose(normalized, (2, 1, 0))[None, None]).to(device)
        with torch.inference_mode():
            prediction = model(tensor)
        if isinstance(prediction, dict):
            prediction = prediction.get("detections", prediction.get("boxes"))
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        rows = prediction.detach().cpu().numpy()
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[1] < 5:
            raise ValueError(
                "expected Nx5 output rows [z, y, x, diameter_voxels, confidence]"
            )
        detections: List[NoduleDetection] = []
        for z, y, x, diameter, confidence, *_ in rows:
            if not np.all(np.isfinite([z, y, x, diameter, confidence])):
                continue
            if confidence < 0.5 or diameter <= 0:
                continue
            if not (0 <= x < ct_data.shape[0] and 0 <= y < ct_data.shape[1] and 0 <= z < ct_data.shape[2]):
                continue
            detections.append(
                NoduleDetection(
                    center_xyz=(float(x), float(y), float(z)),
                    diameter_voxels=float(diameter),
                    confidence=float(confidence),
                )
            )
        return detections, None
    except Exception as exc:
        message = f"Nodule detection is disabled because detector inference failed: {exc}"
        LOGGER.warning(message)
        return [], message


def window_ct(data: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("--window-width must be greater than zero.")
    low, high = center - width / 2.0, center + width / 2.0
    return np.clip((np.clip(data, low, high) - low) / width, 0.0, 1.0)


def to_radiological(array_xy: np.ndarray) -> np.ndarray:
    """
    Canonical RAS axial array -> standard radiological display.

    The resulting image has patient right on image left and patient anterior
    at the top. Both decisions follow from the canonicalized affine rather
    than from the input array's original row/column order.
    """
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
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    ]


def informative_slice(mask: np.ndarray, labels: Mapping[str, int]) -> int:
    best_slice, best_score = mask.shape[2] // 2, -1.0
    preferred = ("lung_right", "lung_left", "heart", "trachea", "aorta", "vertebrae")
    for z in range(mask.shape[2]):
        plane = mask[:, :, z]
        areas = [int(np.count_nonzero(plane == labels[key])) for key in preferred]
        present = sum(area > 0 for area in areas)
        bilateral = int(areas[0] > 0 and areas[1] > 0)
        score = present * 1_000_000.0 + bilateral * 1_000_000.0 + math.sqrt(sum(areas))
        if score > best_score:
            best_slice, best_score = z, score
    return best_slice


def choose_slice(
    requested: Optional[str],
    mask: np.ndarray,
    labels: Mapping[str, int],
    nodules: Sequence[NoduleDetection],
) -> int:
    if requested is None or requested.lower() == "auto":
        if nodules:
            # Prefer a real nodule-containing slice while retaining anatomical context.
            candidates = sorted({int(round(item.center_xyz[2])) for item in nodules})
            candidates = [z for z in candidates if 0 <= z < mask.shape[2]]
            if candidates:
                return max(
                    candidates,
                    key=lambda z: np.count_nonzero(mask[:, :, z]),
                )
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
    if len(value) != 6:
        raise ValueError(f"Invalid color '{color}'; expected #RRGGBB.")
    return np.asarray([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def compose_overlay(
    ct_yx: np.ndarray,
    mask_yx: np.ndarray,
    labels: Mapping[str, int],
    colors: Mapping[str, str],
    alpha: float,
) -> np.ndarray:
    rgb = np.repeat(ct_yx[:, :, None], 3, axis=2)
    for key, label in labels.items():
        region = mask_yx == label
        if not np.any(region):
            continue
        color = hex_to_rgb(colors[key])
        rgb[region] = (1.0 - alpha) * rgb[region] + alpha * color
        for contour in find_contours(region):
            points = np.rint(contour).astype(int)
            x = np.clip(points[:, 0], 0, rgb.shape[1] - 1)
            y = np.clip(points[:, 1], 0, rgb.shape[0] - 1)
            rgb[y, x] = color
            rgb[np.clip(y + 1, 0, rgb.shape[0] - 1), x] = color
    return np.clip(rgb, 0.0, 1.0)


def save_overlays(
    output_dir: Path,
    ct_data: np.ndarray,
    mask: np.ndarray,
    labels: Mapping[str, int],
    colors: Mapping[str, str],
    center: float,
    width: float,
    alpha: float,
    stride: int,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for z in range(0, ct_data.shape[2], stride):
        ct_yx = to_radiological(window_ct(ct_data[:, :, z], center, width))
        mask_yx = to_radiological(mask[:, :, z])
        overlay = compose_overlay(ct_yx, mask_yx, labels, colors, alpha)
        name = f"slice_{z:03d}.png"
        plt.imsave(overlay_dir / name, overlay)
        saved.append(f"overlays/{name}")
    return saved


def label_position(
    key: str,
    centroid: Tuple[float, float],
    width: int,
    height: int,
    order: int,
) -> Tuple[float, float, str]:
    x, y = centroid
    if key == "trachea":
        return width * 0.5, -height * 0.10, "center"
    if key == "vertebrae":
        return width * 0.72, height * 1.10, "center"
    if key == "heart":
        return -width * 0.10, height * 0.82, "right"
    if x < width / 2:
        return -width * 0.10, max(height * 0.12, min(height * 0.88, y + order * 4)), "right"
    return width * 1.10, max(height * 0.12, min(height * 0.88, y + order * 4)), "left"


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

    for key, label in labels.items():
        volume_region = mask == label
        slice_region = to_radiological(mask[:, :, z] == label)
        contours = find_contours(slice_region) if np.any(slice_region) else []
        bbox = bbox_for_mask(slice_region)
        voxel_count = int(np.count_nonzero(volume_region))
        display_name = DISPLAY_NAMES.get(key, key.replace("_", " ").title())
        structures[key] = {
            "name": display_name,
            "label": key,
            "model": "TotalSegmentator",
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
                    "mask_label_value": int(label),
                    "mask_file": "segmentation_mask.nii.gz",
                    "voxel_count_3d": voxel_count,
                    "confidence": None,
                    "source": "TotalSegmentator",
                    "status": "pending_review",
                    "editable": True,
                    "color": colors[key],
                }
            )
    return annotations, structures


def nodule_annotations(
    nodules: Sequence[NoduleDetection],
    z: int,
    shape_xyz: Sequence[int],
    spacing: Sequence[float],
    color: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    width = int(shape_xyz[0])
    height = int(shape_xyz[1])
    annotations: List[Dict[str, Any]] = []
    all_detections: List[Dict[str, Any]] = []
    for index, item in enumerate(nodules, start=1):
        x, y, center_z = item.center_xyz
        display_x = float(width - 1 - x)
        display_y = float(height - 1 - y)
        radius = max(2.0, item.diameter_voxels / 2.0)
        visible = abs(center_z - z) <= max(0.5, radius)
        detection = {
            "id": f"nodule_{index:03d}",
            "center_xyz_voxel": [round(x, 3), round(y, 3), round(center_z, 3)],
            "diameter_voxels": round(item.diameter_voxels, 3),
            "confidence": round(item.confidence, 5),
            "visible_on_slice": visible,
        }
        all_detections.append(detection)
        if not visible:
            continue
        bbox = [
            int(round(display_x - radius)),
            int(round(display_y - radius)),
            int(round(display_x + radius)),
            int(round(display_y + radius)),
        ]
        theta = np.linspace(0, 2 * np.pi, 72)
        contour = np.column_stack(
            (display_x + radius * np.cos(theta), display_y + radius * np.sin(theta))
        )
        annotations.append(
            {
                "id": detection["id"],
                "class": "Nodule",
                "label": "nodule",
                "type": "detection",
                "slice_index": z,
                "contours": [decimate_contour(contour)],
                "bbox": bbox,
                "center": [round(display_x, 2), round(display_y, 2)],
                "diameter_voxels": round(item.diameter_voxels, 3),
                "confidence": round(item.confidence, 5),
                "source": "user_supplied_torchscript_nodule_detector",
                "status": "pending_review",
                "editable": True,
                "color": color,
            }
        )
    structure = {
        "name": "Nodule",
        "label": "nodule",
        "model": "user_supplied_torchscript_nodule_detector",
        "detected": bool(nodules),
        "visible_on_slice": bool(annotations),
        "mask_available": False,
        "contour_available": bool(annotations),
        "slice_index": z,
        "detections": all_detections,
        "confidence": max((item.confidence for item in nodules), default=None),
        "color": color,
    }
    return annotations, structure


def save_annotated_slice(
    destination: Path,
    ct_data: np.ndarray,
    mask: np.ndarray,
    labels: Mapping[str, int],
    z: int,
    colors: Mapping[str, str],
    center: float,
    width: float,
    alpha: float,
    nodule_items: Sequence[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    ct_yx = to_radiological(window_ct(ct_data[:, :, z], center, width))
    mask_yx = to_radiological(mask[:, :, z])
    height, image_width = ct_yx.shape

    fig, axis = plt.subplots(figsize=(12, 10), dpi=180, facecolor="#090d12")
    axis.set_facecolor("#090d12")
    axis.imshow(ct_yx, cmap="gray", vmin=0, vmax=1, origin="upper")

    visible: List[Tuple[str, Tuple[float, float], str]] = []
    for key, label in labels.items():
        region = mask_yx == label
        if not np.any(region):
            continue
        color = colors[key]
        color_rgb = hex_to_rgb(color)
        rgba = np.zeros((height, image_width, 4), dtype=float)
        rgba[region, :3] = color_rgb
        rgba[region, 3] = alpha
        axis.imshow(rgba, origin="upper")
        contours = find_contours(region)
        for contour in contours:
            axis.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.7)
        largest = max(contours, key=len)
        centroid = (float(np.mean(largest[:, 0])), float(np.mean(largest[:, 1])))
        visible.append((key, centroid, color))

    for order, (key, centroid, color) in enumerate(visible):
        label_x, label_y, alignment = label_position(
            key, centroid, image_width, height, order
        )
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

    for item in nodule_items:
        center_x, center_y = item["center"]
        radius = item["diameter_voxels"] / 2.0
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius=radius,
                fill=False,
                edgecolor=colors["nodule"],
                linewidth=2.2,
                zorder=25,
            )
        )
        axis.annotate(
            f"Nodule ({item['confidence']:.2f})",
            xy=(center_x, center_y),
            xytext=(image_width * 1.10, min(height * 0.90, center_y)),
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#111a24",
                edgecolor=colors["nodule"],
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color=colors["nodule"], linewidth=1.6),
            zorder=26,
        )

    axis.text(
        0.02,
        1.02,
        "R",
        transform=axis.transAxes,
        color="white",
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axis.text(
        0.98,
        1.02,
        "L",
        transform=axis.transAxes,
        color="white",
        fontsize=15,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    axis.text(
        0.5,
        1.02,
        f"AI-annotated chest CT · axial slice {z}",
        transform=axis.transAxes,
        color="#d9e2ec",
        fontsize=11,
        ha="center",
        va="bottom",
    )
    axis.set_xlim(-image_width * 0.18, image_width * 1.18)
    axis.set_ylim(height * 1.16, -height * 0.16)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.savefig(
        destination,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)


def save_mask(path: Path, mask: np.ndarray, reference: Any) -> None:
    import nibabel as nib

    header = reference.header.copy()
    header.set_data_dtype(np.uint16)
    image = nib.Nifti1Image(mask.astype(np.uint16), reference.affine, header=header)
    nib.save(image, str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment a 3D chest CT and create a radiological annotated PNG with "
            "transparent masks, contours, labels, and leader lines."
        ),
        epilog=(
            "Install: python -m pip install TotalSegmentator torch SimpleITK "
            "nibabel numpy matplotlib scikit-image pydicom"
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="NIfTI file or DICOM directory.")
    parser.add_argument("--output", "-o", required=True, help="Output directory.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fast", action="store_true", help="Use 3 mm TotalSegmentator inference.")
    parser.add_argument(
        "--slice",
        default="auto",
        help="Axial slice index, 'middle', or 'auto' (default: auto).",
    )
    parser.add_argument("--window-center", type=float, default=WINDOW_CENTER)
    parser.add_argument("--window-width", type=float, default=WINDOW_WIDTH)
    parser.add_argument("--overlay-alpha", type=float, default=0.18)
    parser.add_argument(
        "--overlay-stride",
        type=int,
        default=1,
        help="Save every Nth overlay slice (default: 1).",
    )
    parser.add_argument(
        "--structures",
        default="",
        help="Additional comma-separated class names from TotalSegmentator's total task.",
    )
    parser.add_argument(
        "--colors",
        help='JSON object or JSON file, e.g. \'{"heart":"#ff00aa"}\'.',
    )
    parser.add_argument(
        "--enable-nodules",
        action="store_true",
        help="Enable optional inference using --nodule-checkpoint.",
    )
    parser.add_argument(
        "--nodule-checkpoint",
        help=(
            "TorchScript checkpoint returning Nx5 [z,y,x,diameter_voxels,confidence]. "
            "No nodule is drawn unless this detector returns one."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1.")
    if args.overlay_stride < 1:
        raise ValueError("--overlay-stride must be at least 1.")

    print("Loading CT...")
    loaded = load_ct(args.input)
    ct_data, spacing = validate_ct(loaded.image)
    for warning in loaded.warnings:
        LOGGER.warning("Warning: %s", warning)
    print(f"CT dimensions: {tuple(int(value) for value in ct_data.shape)}")
    print(f"CT spacing: {tuple(round(value, 4) for value in spacing)} mm")

    torch_device, ts_device = resolve_device(args.device)
    extras = parse_extra_structures(args.structures)
    model_structures = list(dict.fromkeys((*BASE_MODEL_STRUCTURES, *extras)))

    print("Loading TotalSegmentator...")
    print("Running anatomical segmentation...")
    raw_mask, label_map = run_anatomical_segmentation(
        image=loaded.image,
        model_structures=model_structures,
        ts_device=ts_device,
        fast=args.fast,
        verbose=args.verbose,
    )
    mask, labels = combine_anatomical_masks(raw_mask, label_map, extras)
    colors = load_colors(args.colors, list(labels) + ["nodule"])

    print("Detecting nodules...")
    nodules, nodule_message = run_nodule_detector(
        enabled=args.enable_nodules,
        checkpoint=args.nodule_checkpoint,
        ct_data=ct_data,
        device=torch_device,
    )
    selected_slice = choose_slice(args.slice, mask, labels, nodules)

    print("Generating contours...")
    annotations, structures_by_key = build_slice_annotations(
        mask, labels, selected_slice, spacing, colors
    )
    nodule_items, nodule_structure = nodule_annotations(
        nodules, selected_slice, ct_data.shape, spacing, colors["nodule"]
    )
    annotations.extend(nodule_items)
    structures_by_key["nodule"] = nodule_structure

    print("Generating annotations...")
    mask_path = output_dir / "segmentation_mask.nii.gz"
    save_mask(mask_path, mask, loaded.image)

    print("Generating visualization...")
    overlay_files = save_overlays(
        output_dir=output_dir,
        ct_data=ct_data,
        mask=mask,
        labels=labels,
        colors=colors,
        center=args.window_center,
        width=args.window_width,
        alpha=args.overlay_alpha,
        stride=args.overlay_stride,
    )
    image_path = output_dir / "annotated_slice.png"
    save_annotated_slice(
        destination=image_path,
        ct_data=ct_data,
        mask=mask,
        labels=labels,
        z=selected_slice,
        colors=colors,
        center=args.window_center,
        width=args.window_width,
        alpha=args.overlay_alpha,
        nodule_items=nodule_items,
    )

    payload = {
        "schema_version": "1.0",
        "status": "completed",
        "image": image_path.name,
        # The image these annotations were drawn on. "uri" is filled in by
        # deployments that copy the outputs elsewhere (e.g. Lambda uploads to S3),
        # because the local path is meaningless once the run finishes.
        "annotated_image": {
            "file": image_path.name,
            "path": str(image_path),
            "s3_path": None,
            "uri": None,
        },
        "slice": selected_slice,
        "slice_index": selected_slice,
        "input": str(Path(args.input).expanduser().resolve()),
        "input_type": loaded.input_type,
        "orientation": {
            "display_convention": "radiological",
            "image_left": "patient_right",
            "image_right": "patient_left",
            "markers": {"left_edge": "R", "right_edge": "L"},
            "source_affine_ras": loaded.image.affine.tolist(),
        },
        "coordinate_system": {
            "origin": "top_left_of_displayed_CT",
            "x_direction": "right",
            "y_direction": "down",
            "units": "pixels",
            "image_shape_yx": [int(ct_data.shape[1]), int(ct_data.shape[0])],
        },
        "window": {
            "center": args.window_center,
            "width": args.window_width,
        },
        "model": {
            "anatomy": "TotalSegmentator/total",
            "fast": bool(args.fast),
            "device": torch_device,
            "hard_labels": True,
            "confidence_note": (
                "TotalSegmentator hard-label output does not provide calibrated per-structure "
                "confidence; anatomy confidence is null rather than fabricated."
            ),
            "nodule_detector": (
                "user_supplied_torchscript" if args.enable_nodules and not nodule_message else None
            ),
            "nodule_status": nodule_message or "enabled",
        },
        "volume": {
            "shape_xyz": [int(value) for value in ct_data.shape],
            "spacing_xyz_mm": [float(value) for value in spacing],
            "mask_file": "segmentation_mask.nii.gz",
            "mask_labels": {key: int(value) for key, value in labels.items()},
        },
        "structures": list(structures_by_key.values()),
        "annotations": annotations,
        "overlays": overlay_files,
        "frontend_capabilities": {
            "review_status": "pending_review",
            "operations": ["accept", "reject", "edit_contour", "move", "change_class", "delete"],
        },
        "warnings": loaded.warnings + ([nodule_message] if nodule_message else []),
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
