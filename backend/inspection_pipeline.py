"""
Orchestrates the full detection + reasoning pipeline:

    image -> YOLO -> detections -> crop each -> VLM (optional) -> InspectionResult

Kept separate from backend/live_inference.py so the same function can be called from
the webcam/video loop, the FastAPI endpoints, and the benchmark script.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from backend.schemas import (
    Detection,
    ImageSize,
    InspectionItem,
    InspectionResult,
    InspectionStatus,
    QualityAssessment,
    RequiredAction,
)
if TYPE_CHECKING:
    from backend.vlm_reasoning import VLMBackend


def crop_detection(image: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def build_quality_commentary(label: str, quality: QualityAssessment) -> str:
    """Translate structured assessment values into an operator-ready paragraph."""
    item_name = label.replace("_", " ").strip().capitalize()
    if quality.status == InspectionStatus.SKIPPED:
        return f"{item_name} was detected, but quality was not assessed. Recommended action: no action required."
    if quality.status == InspectionStatus.UNCERTAIN:
        evidence = quality.explanation.rstrip(".") or "the visual evidence was inconclusive"
        return f"{item_name} needs manual review because {evidence}. Recommended action: flag for review."

    score = (
        f"Quality score: {quality.overall_quality_score:.2f}/1.00. "
        if quality.overall_quality_score is not None
        else "Quality score is unavailable. "
    )
    if quality.defects:
        defects = ", ".join(defect.replace("_", " ") for defect in quality.defects[:3])
        evidence = f"Observed issue{'s' if len(quality.defects) > 1 else ''}: {defects}. "
    elif quality.status == InspectionStatus.OK:
        evidence = "No visible quality defects were identified. "
    elif quality.quality_metrics:
        weakest_metric, weakest_score = min(quality.quality_metrics.items(), key=lambda metric: metric[1])
        evidence = f"The lowest assessed metric is {weakest_metric.replace('_', ' ')} at {weakest_score:.2f}. "
    else:
        evidence = quality.explanation.rstrip(".") + ". " if quality.explanation else "Visible quality concerns were identified. "

    actions = {
        RequiredAction.NONE: "no action required",
        RequiredAction.FLAG_FOR_REVIEW: "flag for review",
        RequiredAction.REMOVE: "remove from the saleable batch",
    }
    return f"{item_name}. {score}{evidence}Recommended action: {actions[quality.required_action]}."


def run_inspection(
    image: np.ndarray,
    yolo_model,
    frame_id: int,
    source: str,
    vlm_backend: Optional[VLMBackend] = None,
    vlm_confidence_gate: float = 0.0,
) -> InspectionResult:
    """
    Run YOLO detection, optionally followed by VLM reasoning on each crop.

    vlm_confidence_gate: only send crops to the VLM if the YOLO confidence is
    >= this value. Set to 0.0 to run the VLM on every detection; raise it to
    save VLM calls on low-confidence detections that are likely noise anyway.
    """
    import time
    h, w = image.shape[:2]
    
    yolo_start = time.perf_counter()
    yolo_results = yolo_model(image)[0]  # Ultralytics-style single-image result
    yolo_time = time.perf_counter() - yolo_start
    print(f"[Pipeline] YOLO inference finished: {yolo_time:.3f} seconds")

    items: List[InspectionItem] = []
    for box in yolo_results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox_xyxy = box.xyxy[0].tolist()
        # Normalize coordinates so clients can render boxes at any image size.
        bbox_normalized = [
            bbox_xyxy[0] / w,
            bbox_xyxy[1] / h,
            bbox_xyxy[2] / w,
            bbox_xyxy[3] / h,
        ]
        label = yolo_model.names[class_id]

        detection = Detection(
            label=label,
            class_id=class_id,
            confidence=confidence,
            bbox_xyxy=bbox_xyxy,
            bbox_normalized=bbox_normalized,
        )

        quality = None
        # Gate VLM calls to control latency and external inference cost.
        if vlm_backend is not None and confidence >= vlm_confidence_gate:
            crop = crop_detection(image, bbox_xyxy)
            if crop.size > 0:
                quality = vlm_backend.analyze(crop, label, confidence)
        
        # Preserve a typed result when reasoning is disabled or gated out.
        if quality is None:
            quality = QualityAssessment(
                status=InspectionStatus.SKIPPED,
                overall_quality_score=None,
                quality_metrics={},
                defects=[],
                explanation="VLM reasoning skipped (low confidence or VLM disabled).",
                required_action=RequiredAction.NONE,
                vlm_backend=vlm_backend.name if vlm_backend else "none",
            )

        # Keep a concise explanation beside the raw metrics for operators.
        quality.commentary = build_quality_commentary(label, quality)
        items.append(InspectionItem(detection=detection, quality=quality))

    return InspectionResult(
        frame_id=frame_id,
        timestamp=datetime.utcnow(),
        source=source,
        image_size=ImageSize(width=w, height=h),
        num_detections=len(items),
        items=items,
    )
