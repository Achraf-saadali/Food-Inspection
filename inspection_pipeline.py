"""Detection, per-object quality analysis, and image-level aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Optional

import numpy as np

from schemas import (
    Detection,
    ImageSize,
    InspectionItem,
    InspectionResult,
    InspectionStatus,
    QualityAssessment,
    RequiredAction,
)
from vlm_reasoning import VLMBackend

MAX_CROPS_PER_VLM_REQUEST = 6
MAX_VLM_CALLS_PER_INSPECTION = 3


def crop_detection(image: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def _image_commentary(items: List[InspectionItem], analyzed_count: int) -> str:
    """Create the sole human-readable summary from structured results only."""
    if not items:
        return "No objects were detected in the image."

    counts = Counter(item.detection.label for item in items)
    inventory = ", ".join(
        f"{count} {label}{'' if count == 1 else 's'}" for label, count in counts.items()
    )
    quality_counts = Counter(item.quality.status for item in items)
    good = quality_counts[InspectionStatus.OK]
    defects = quality_counts[InspectionStatus.DEFECT]
    uncertain = quality_counts[InspectionStatus.UNCERTAIN]
    skipped = quality_counts[InspectionStatus.SKIPPED]

    sentences = [f"The image contains {inventory}."]
    if good and not defects and not uncertain and not skipped:
        sentences.append("All analyzed items show acceptable visible quality.")
    elif good:
        sentences.append(f"{good} analyzed item{'s' if good != 1 else ''} show acceptable visible quality.")
    if defects:
        defect_names = sorted({d for item in items for d in item.quality.defects if d})
        detail = f" Visible issues include {', '.join(defect_names)}." if defect_names else " Some items show visible quality defects."
        sentences.append(f"{defects} item{'s' if defects != 1 else ''} require attention.{detail}")
    if uncertain:
        sentences.append(f"{uncertain} analyzed item{'s' if uncertain != 1 else ''} could not be assessed confidently and require review.")
    if skipped or analyzed_count < len(items):
        skipped_total = max(skipped, len(items) - analyzed_count)
        sentences.append(f"{skipped_total} detected item{'s' if skipped_total != 1 else ''} were not fully evaluated by the VLM and require review; they must not be treated as confirmed good.")
    return "".join(sentences)


def run_inspection(
    image: np.ndarray,
    yolo_model,
    frame_id: int,
    source: str,
    vlm_backend: Optional[VLMBackend] = None,
    vlm_confidence_gate: float = 0.0,
) -> InspectionResult:
    """Run YOLO and preserve detailed object results plus one image summary.

    Crop IDs are stable within an inspection. VLM request limits are enforced;
    skipped and failed objects remain explicit uncertainty rather than becoming
    implicit successful assessments.
    """
    import time

    h, w = image.shape[:2]
    yolo_start = time.perf_counter()
    yolo_results = yolo_model(image)[0]
    print(f"[Pipeline] YOLO inference finished: {time.perf_counter() - yolo_start:.3f} seconds")

    items: List[InspectionItem] = []
    candidates = []
    for index, box in enumerate(yolo_results.boxes, start=1):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox_xyxy = box.xyxy[0].tolist()
        label = yolo_model.names[class_id]
        crop_id = f"CROP_{index:03d}"
        detection = Detection(
            crop_id=crop_id,
            label=label,
            class_id=class_id,
            confidence=confidence,
            bbox_xyxy=bbox_xyxy,
            bbox_normalized=[bbox_xyxy[0] / w, bbox_xyxy[1] / h, bbox_xyxy[2] / w, bbox_xyxy[3] / h],
        )
        item = InspectionItem(
            detection=detection,
            quality=QualityAssessment(
                status=InspectionStatus.SKIPPED,
                overall_quality_score=None,
                quality_metrics={},
                defects=[],
                explanation="VLM reasoning was not run for this detection.",
                required_action=RequiredAction.FLAG_FOR_REVIEW,
                vlm_backend=vlm_backend.name if vlm_backend else "none",
            ),
        )
        items.append(item)
        if vlm_backend is not None and confidence >= vlm_confidence_gate:
            crop = crop_detection(image, bbox_xyxy)
            if crop.size > 0:
                candidates.append((index - 1, crop, label, confidence))

    analyzed_count = 0
    if vlm_backend is not None:
        max_analyzable = MAX_CROPS_PER_VLM_REQUEST * MAX_VLM_CALLS_PER_INSPECTION
        for item_index, crop, label, confidence in candidates[:max_analyzable]:
            items[item_index].quality = vlm_backend.analyze(crop, label, confidence)
            analyzed_count += 1

    return InspectionResult(
        frame_id=frame_id,
        timestamp=datetime.utcnow(),
        source=source,
        image_size=ImageSize(width=w, height=h),
        num_detections=len(items),
        items=items,
        commentary=_image_commentary(items, analyzed_count),
    )
