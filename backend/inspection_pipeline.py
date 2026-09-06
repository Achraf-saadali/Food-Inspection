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
from backend.quality_profiles import get_quality_metrics
from backend.reporting import translate_defect, translate_object_label

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

MAX_CROPS_PER_VLM_REQUEST = 4


def build_crop_collage(crops: list[dict[str, object]]) -> np.ndarray:
    """Build a labeled 2-column collage so the VLM can map results to crops."""
    import cv2

    panel_size = 512
    label_height = 32
    columns = 2
    rows = (len(crops) + columns - 1) // columns
    collage = np.full((rows * (panel_size + label_height), columns * panel_size, 3), 255, dtype=np.uint8)
    for index, crop_info in enumerate(crops):
        crop = crop_info["crop"]
        resized = cv2.resize(crop, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
        x = (index % columns) * panel_size
        y = (index // columns) * (panel_size + label_height)
        collage[y:y + panel_size, x:x + panel_size] = resized
        cv2.putText(collage, str(crop_info["crop_id"]), (x + 8, y + panel_size + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return collage


def crop_detection(image: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def build_quality_commentary(label: str, quality: QualityAssessment) -> str:
    """Translate structured assessment values into an operator-ready paragraph."""
    item_name = translate_object_label(label)
    if quality.status == InspectionStatus.SKIPPED:
        return f"{item_name} a été détecté, mais sa qualité n’a pas été évaluée. Action recommandée : aucune action nécessaire."
    if quality.status == InspectionStatus.UNCERTAIN:
        evidence = quality.explanation.rstrip(".") or "les éléments visuels sont insuffisants"
        return f"{item_name} doit être vérifié manuellement, car {evidence}. Action recommandée : vérification manuelle."

    score = (
        f"Score qualité : {quality.overall_quality_score:.2f}/1,00. "
        if quality.overall_quality_score is not None
        else "Score qualité indisponible. "
    )
    if quality.defects:
        defects = ", ".join(translate_defect(defect) for defect in quality.defects)
        evidence = f"Problème{'s' if len(quality.defects) > 1 else ''} observé{'s' if len(quality.defects) > 1 else ''} : {defects}. "
    elif quality.status == InspectionStatus.OK:
        evidence = "Aucun défaut visuel n’a été identifié. "
    elif quality.status == InspectionStatus.DEFECT:
        evidence = "Des problèmes de qualité visibles ont été identifiés. "
    elif quality.quality_metrics:
        weakest_metric, weakest_score = min(quality.quality_metrics.items(), key=lambda metric: metric[1])
        evidence = f"La métrique la plus faible est {weakest_metric.replace('_', ' ')} avec {weakest_score:.2f}. "
    else:
        evidence = quality.explanation.rstrip(".") + ". " if quality.explanation else "Des problèmes de qualité visibles ont été identifiés. "

    actions = {
        RequiredAction.NONE: "aucune action nécessaire",
        RequiredAction.FLAG_FOR_REVIEW: "vérifier manuellement",
        RequiredAction.REMOVE: "isoler du lot vendable",
    }
    return f"{item_name}. {score}{evidence}Action recommandée : {actions[quality.required_action]}."


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
    eligible_crops: list[dict[str, object]] = []
    detection_records: list[tuple[Detection, Optional[str]]] = []
    crop_number = 0
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
        label = str(yolo_model.names[class_id]).strip().lower()

        detection = Detection(
            label=label,
            display_label=label,
            class_id=class_id,
            confidence=confidence,
            bbox_xyxy=bbox_xyxy,
            bbox_normalized=bbox_normalized,
        )

        crop_id: Optional[str] = None
        # Gate VLM eligibility while retaining every detection in the result.
        if vlm_backend is not None and confidence >= vlm_confidence_gate:
            crop = crop_detection(image, bbox_xyxy)
            if crop.size > 0:
                crop_number += 1
                crop_id = f"CROP_{crop_number:03d}"
                eligible_crops.append({
                    "crop_id": crop_id,
                    "crop": crop.copy(),
                    "label": label,
                    "confidence": confidence,
                    "metrics": get_quality_metrics(label),
                })
        detection_records.append((detection, crop_id))

    assessments: dict[str, QualityAssessment] = {}
    if vlm_backend is not None:
        for start in range(0, len(eligible_crops), MAX_CROPS_PER_VLM_REQUEST):
            group = eligible_crops[start:start + MAX_CROPS_PER_VLM_REQUEST]
            collage = build_crop_collage(group)
            assessments.update(vlm_backend.analyze_collage(collage, group))

    for detection, crop_id in detection_records:
        quality = assessments.get(crop_id) if crop_id else None
        # Preserve a typed result when reasoning is disabled or gated out.
        if quality is None:
            quality = QualityAssessment(
                status=InspectionStatus.SKIPPED,
                overall_quality_score=None,
                quality_metrics={},
                defects=[],
                explanation="Analyse qualité non effectuée : confiance insuffisante ou VLM désactivé.",
                required_action=RequiredAction.NONE,
                vlm_backend=vlm_backend.name if vlm_backend else "none",
            )

        # Keep a concise explanation beside the raw metrics for operators.
        display_label = translate_object_label(detection.label)
        if quality.detected_class:
            candidate_label = str(quality.detected_class).strip().lower()
            if candidate_label and candidate_label != detection.label:
                display_label = translate_object_label(candidate_label)
        detection.display_label = display_label
        quality.commentary = build_quality_commentary(display_label, quality)
        items.append(InspectionItem(detection=detection, quality=quality))

    return InspectionResult(
        frame_id=frame_id,
        timestamp=datetime.utcnow(),
        source=source,
        image_size=ImageSize(width=w, height=h),
        num_detections=len(items),
        items=items,
    )
