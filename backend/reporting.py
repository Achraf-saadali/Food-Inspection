"""Farmer-facing report metrics derived from the unified inspection schema.

This module deliberately contains no model inference. It converts the same
InspectionResult returned by the API and live runner into management metrics
without creating a second inspection logic path.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from backend.schemas import InspectionResult, InspectionStatus, RequiredAction


def build_farmer_report(result: InspectionResult) -> dict[str, Any]:
    assessed = [
        item.quality
        for item in result.items
        if item.quality.status not in {InspectionStatus.SKIPPED}
    ]
    scores = [q.overall_quality_score for q in assessed if q.overall_quality_score is not None]
    metric_values: dict[str, list[float]] = {}
    defects: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    for item in result.items:
        quality = item.quality
        defects.update(quality.defects)
        actions[quality.required_action.value] += 1
        for name, value in quality.quality_metrics.items():
            metric_values.setdefault(name, []).append(float(value))

    defect_count = sum(item.quality.status == InspectionStatus.DEFECT for item in result.items)
    uncertain_count = sum(item.quality.status == InspectionStatus.UNCERTAIN for item in result.items)
    inspected_count = len(assessed)
    denominator = max(result.num_detections, 1)
    if result.num_detections == 0:
        summary = "No food items were detected. Retake the image with the produce clearly visible."
    elif defect_count:
        summary = (
            f"{defect_count} of {result.num_detections} detected item(s) show visible quality concerns. "
            "Separate affected items and review the recommended actions before sale or processing."
        )
    elif uncertain_count:
        summary = (
            f"The system detected {result.num_detections} item(s), but {uncertain_count} require manual review "
            "because the visual evidence was not conclusive."
        )
    else:
        summary = (
            f"The system detected {result.num_detections} item(s) and found no visible quality defects "
            "in the assessed image."
        )

    return {
        "summary_for_farmer": summary,
        "inspection_status": (
            "defect" if defect_count else "uncertain" if uncertain_count else "ok" if result.num_detections else "skipped"
        ),
        "metrics": {
            "detected_items": result.num_detections,
            "assessed_items": inspected_count,
            "assessment_coverage": round(inspected_count / denominator, 3) if result.num_detections else 0.0,
            "defect_rate": round(defect_count / denominator, 3) if result.num_detections else 0.0,
            "uncertain_rate": round(uncertain_count / denominator, 3) if result.num_detections else 0.0,
            "mean_detector_confidence": round(mean(item.detection.confidence for item in result.items), 3) if result.items else None,
            "mean_quality_score": round(mean(scores), 3) if scores else None,
            "metric_averages": {name: round(mean(values), 3) for name, values in metric_values.items()},
        },
        "recommended_actions": dict(actions),
        "most_frequent_defects": defects.most_common(),
    }
