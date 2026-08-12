"""
Unified JSON contract for the Food-Inspection pipeline.

Both the YOLO detection stage and the VLM reasoning stage write into these
models, so downstream consumers (dashboard, DB, notification service) only
ever need to understand one schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InspectionStatus(str, Enum):
    OK = "ok"
    DEFECT = "defect"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"


class DefectType(str, Enum):
    NONE = "none"
    MOLD = "mold"
    BRUISING = "bruising"
    DISCOLORATION = "discoloration"
    FRESHNESS = "freshness"
    OTHER = "other"


class RequiredAction(str, Enum):
    NONE = "none"
    FLAG_FOR_REVIEW = "flag_for_review"
    REMOVE = "remove"


class ImageSize(BaseModel):
    width: int
    height: int


class Detection(BaseModel):
    """Output of the YOLO stage for a single object."""

    crop_id: str = Field(description="Stable identifier linking this detection to its crop and VLM result.")
    label: str
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: List[float] = Field(min_length=4, max_length=4)
    bbox_normalized: List[float] = Field(min_length=4, max_length=4)


class QualityAssessment(BaseModel):
    """Output of the VLM stage for a single crop. Optional because the VLM
    stage may not run on every detection (e.g. skipped by confidence gate)."""

    status: InspectionStatus
    overall_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    defects: List[str] = Field(default_factory=list)
    explanation: str
    required_action: RequiredAction = RequiredAction.NONE
    vlm_backend: str
    latency_ms: Optional[float] = None


class InspectionItem(BaseModel):
    """One detected object plus its quality assessment."""

    detection: Detection
    quality: QualityAssessment


class InspectionResult(BaseModel):
    """Top-level unified result for one processed frame/image."""

    frame_id: int
    timestamp: datetime
    source: str
    image_size: ImageSize
    num_detections: int
    items: List[InspectionItem]
    commentary: str = Field(description="Exactly one backend-generated human-readable summary for the original image.")

    def to_legacy_detection_json(self) -> dict:
        """Backwards-compatible view matching the existing
        build_detection_json() output, for consumers not yet updated."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "image_size": self.image_size.model_dump(),
            "num_detections": self.num_detections,
            "detections": [item.detection.model_dump() for item in self.items],
        }
