"""
Shared 6-class chest anatomy label spec.

Both `chest_ct_annotation.py` (TotalSegmentator-backed annotation tool) and the
nnU-Net training pipeline (`build_pseudo_labels.py`, the Kaggle training
notebook, `finetune_chest_nnunet.py`) import from here, so the class set never
drifts between the two.

The label *names* below (the keys of `CHEST_LABELS`) are the contract: a
trained nnU-Net model's `dataset.json["labels"]` must use exactly these keys,
because `chest_ct_annotation.py`'s nnunet backend reads label names straight
from that file and looks them up in `DEFAULT_COLORS`/`DISPLAY_NAMES` by name.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

# TotalSegmentator "total" task structure names that make up each merged class.
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

# The 6-class training target: nnU-Net dataset.json["labels"]. Class indices
# match the merged-mask output of combine_anatomical_masks() below.
CHEST_LABELS: Dict[str, int] = {
    "background": 0,
    "lung_right": 1,
    "lung_left": 2,
    "heart": 3,
    "trachea": 4,
    "aorta": 5,
    "vertebrae": 6,
}


def combine_anatomical_masks(
    raw_mask: np.ndarray,
    label_map: Mapping[int, str],
    extra_structures: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Collapse a raw TotalSegmentator "total" mask into the 6 chest classes.

    Returns the merged mask plus a name -> label dict covering the 6 base
    classes (matching CHEST_LABELS minus "background") and any requested
    extras, appended from label 7 upward.
    """
    name_to_source = {name: label for label, name in label_map.items()}
    output = np.zeros(raw_mask.shape, dtype=np.uint16)
    output_labels: Dict[str, int] = {
        name: index for name, index in CHEST_LABELS.items() if name != "background"
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
