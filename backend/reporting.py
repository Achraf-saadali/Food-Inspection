"""Farmer-facing report generation from the unified inspection schema.

This module does not run inference. It translates the existing InspectionResult
into an actionable report for a farmer while retaining technical traceability.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from backend.schemas import InspectionItem, InspectionResult, InspectionStatus, RequiredAction


OBJECT_LABELS: dict[str, str] = {
    "tomato": "Tomate",
    "apple": "Pomme",
    "banana": "Banane",
    "orange": "Orange",
    "potato": "Pomme de terre",
    "carrot": "Carotte",
    "pepper": "Poivron",
    "cucumber": "Concombre",
}

DEFECT_LABELS: dict[str, str] = {
    "mold": "Moisissure",
    "bruising": "Meurtrissure",
    "discoloration": "Décoloration",
    "freshness": "Fraîcheur insuffisante",
    "other": "Autre problème visible",
    "none": "Aucun défaut identifié",
}

ACTION_LABELS: dict[str, str] = {
    "none": "Aucune action nécessaire",
    "flag_for_review": "Vérifier manuellement",
    "remove": "Séparer du lot vendable",
}


def translate_object_label(label: str) -> str:
    normalized = label.strip().lower()
    return OBJECT_LABELS.get(normalized, label.replace("_", " ").capitalize())


def translate_defect(defect: str) -> str:
    normalized = defect.strip().lower()
    return DEFECT_LABELS.get(normalized, defect.replace("_", " ").capitalize())


def translate_defects(defects: list[str]) -> list[str]:
    return [translate_defect(defect) for defect in defects]


def translate_action(action: RequiredAction) -> str:
    return ACTION_LABELS.get(action.value, "Action non définie")


def reliability_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "Élevée"
    if confidence >= 0.60:
        return "Moyenne"
    return "Faible"


def farmer_result_label(item: InspectionItem) -> str:
    quality = item.quality
    if quality.status == InspectionStatus.SKIPPED:
        return "Non évalué"
    if quality.required_action == RequiredAction.REMOVE:
        return "À isoler"
    if quality.required_action == RequiredAction.FLAG_FOR_REVIEW:
        return "À vérifier"
    if quality.status == InspectionStatus.DEFECT:
        return "À isoler"
    if quality.status == InspectionStatus.UNCERTAIN:
        return "À vérifier"
    return "Acceptable"


def farmer_action_label(item: InspectionItem) -> str:
    if item.quality.status == InspectionStatus.SKIPPED:
        return "Reprendre une photo ou examiner manuellement"
    return translate_action(item.quality.required_action)


def farmer_explanation(item: InspectionItem) -> str:
    quality = item.quality
    defects = translate_defects(quality.defects)
    if defects:
        defect_text = ", ".join(defects[:3])
        if len(defects) > 3:
            defect_text += ", ..."
        if quality.explanation:
            return f"{defect_text}. {quality.explanation.strip()}"
        return f"Problème visible : {defect_text}."
    if quality.status == InspectionStatus.OK:
        return "Aucun défaut visuel identifiable sur cette image."
    if quality.status == InspectionStatus.SKIPPED:
        return "Cet objet a été détecté, mais sa qualité n’a pas été évaluée."
    return quality.explanation.strip() or "Une vérification supplémentaire est nécessaire."


def build_object_report(item: InspectionItem, object_id: int) -> dict[str, Any]:
    quality = item.quality
    detection = item.detection
    return {
        "object_id": object_id,
        "object_type": translate_object_label(detection.label),
        "result": farmer_result_label(item),
        "problem": translate_defects(quality.defects),
        "explanation": farmer_explanation(item),
        "action": farmer_action_label(item),
        "reliability": reliability_label(detection.confidence),
        "image_location": {
            "bbox_xyxy": detection.bbox_xyxy,
            "bbox_normalized": detection.bbox_normalized,
        },
        "technical_details": {
            "detector_confidence": round(detection.confidence, 3),
            "quality_score": (
                round(quality.overall_quality_score, 3)
                if quality.overall_quality_score is not None
                else None
            ),
            "quality_metrics": {
                name: round(float(value), 3)
                for name, value in quality.quality_metrics.items()
            },
            "status_code": quality.status.value,
            "required_action_code": quality.required_action.value,
            "vlm_backend": quality.vlm_backend,
            "latency_ms": quality.latency_ms,
        },
    }


def build_lot_summary(object_reports: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "detected": len(object_reports),
        "acceptable": sum(obj["result"] == "Acceptable" for obj in object_reports),
        "to_isolate": sum(obj["result"] == "À isoler" for obj in object_reports),
        "to_review": sum(obj["result"] == "À vérifier" for obj in object_reports),
        "not_assessed": sum(obj["result"] == "Non évalué" for obj in object_reports),
    }


def build_lot_decision(summary: dict[str, int]) -> dict[str, str]:
    if summary["detected"] == 0:
        return {
            "label": "Nouvelle photo nécessaire",
            "severity": "information",
            "explanation": "Aucun produit n’a été détecté clairement. Prenez une photo plus nette avec les produits visibles.",
        }
    if summary["to_isolate"] > 0:
        return {
            "label": "Lot à trier avant vente",
            "severity": "attention",
            "explanation": f"{summary['to_isolate']} produit(s) doivent être séparés et {summary['to_review'] + summary['not_assessed']} doivent être vérifiés.",
        }
    if summary["to_review"] > 0 or summary["not_assessed"] > 0:
        return {
            "label": "Vérification manuelle nécessaire",
            "severity": "warning",
            "explanation": f"{summary['to_review'] + summary['not_assessed']} produit(s) n’ont pas pu être évalués avec certitude.",
        }
    return {
        "label": "Lot visuellement acceptable",
        "severity": "success",
        "explanation": f"Les {summary['acceptable']} produit(s) inspectés ne présentent pas de défaut visible sur cette image.",
    }


def build_actions(object_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    to_isolate = [obj["object_id"] for obj in object_reports if obj["result"] == "À isoler"]
    to_review = [obj["object_id"] for obj in object_reports if obj["result"] == "À vérifier"]
    not_assessed = [obj["object_id"] for obj in object_reports if obj["result"] == "Non évalué"]
    actions: list[dict[str, Any]] = []
    if to_isolate:
        actions.append({
            "priority": 1,
            "type": "isolate",
            "label": "Séparer les produits présentant un défaut",
            "object_ids": to_isolate,
        })
    if to_review:
        actions.append({
            "priority": 2,
            "type": "review",
            "label": "Vérifier manuellement les produits incertains",
            "object_ids": to_review,
        })
    if not_assessed:
        actions.append({
            "priority": 3,
            "type": "retake_or_review",
            "label": "Reprendre une photo ou examiner les produits non évalués",
            "object_ids": not_assessed,
        })
    if not actions:
        actions.append({
            "priority": 1,
            "type": "none",
            "label": "Aucune action particulière nécessaire sur la base de cette image",
            "object_ids": [],
        })
    return actions


def build_warnings(result: InspectionResult, object_reports: list[dict[str, Any]]) -> list[str]:
    warnings = [
        "La décision concerne uniquement les défauts visibles sur cette image.",
        "Un résultat acceptable ne garantit pas l’absence de défaut non visible.",
    ]
    if any(obj["reliability"] == "Faible" for obj in object_reports):
        warnings.append("Certains objets sont difficiles à identifier avec certitude. Vérifiez-les manuellement.")
    if any(obj["result"] == "Non évalué" for obj in object_reports):
        warnings.append("Certains objets ont été détectés mais n’ont pas reçu d’évaluation qualité.")
    if result.num_detections != len(result.items):
        warnings.append("Le nombre d’objets annoncé et le nombre d’éléments détaillés diffèrent.")
    return warnings


def build_technical_details(result: InspectionResult) -> dict[str, Any]:
    scores = [item.quality.overall_quality_score for item in result.items if item.quality.overall_quality_score is not None]
    confidences = [item.detection.confidence for item in result.items]
    metric_values: dict[str, list[float]] = {}
    for item in result.items:
        for name, value in item.quality.quality_metrics.items():
            metric_values.setdefault(name, []).append(float(value))
    return {
        "report_id": result.report_id,
        "frame_id": result.frame_id,
        "source": result.source,
        "timestamp": result.timestamp.isoformat(),
        "image_size": result.image_size.model_dump(mode="json"),
        "mean_detector_confidence": round(mean(confidences), 3) if confidences else None,
        "mean_quality_score": round(mean(scores), 3) if scores else None,
        "metric_averages": {
            name: round(mean(values), 3) for name, values in metric_values.items() if values
        },
    }


def build_legacy_metrics(result: InspectionResult, object_reports: list[dict[str, Any]]) -> dict[str, Any]:
    assessed = [item.quality for item in result.items if item.quality.status != InspectionStatus.SKIPPED]
    scores = [quality.overall_quality_score for quality in assessed if quality.overall_quality_score is not None]
    denominator = max(result.num_detections, 1)
    defects = Counter(defect for item in result.items for defect in item.quality.defects)
    actions = Counter(item.quality.required_action.value for item in result.items)
    defect_count = sum(obj["result"] == "À isoler" for obj in object_reports)
    review_count = sum(obj["result"] == "À vérifier" for obj in object_reports)
    return {
        "detected_items": result.num_detections,
        "assessed_items": len(assessed),
        "assessment_coverage": round(len(assessed) / denominator, 3) if result.num_detections else 0.0,
        "defect_rate": round(defect_count / denominator, 3) if result.num_detections else 0.0,
        "uncertain_rate": round(review_count / denominator, 3) if result.num_detections else 0.0,
        "mean_detector_confidence": round(mean(item.detection.confidence for item in result.items), 3) if result.items else None,
        "mean_quality_score": round(mean(scores), 3) if scores else None,
        "metric_averages": build_technical_details(result)["metric_averages"],
        "recommended_actions": dict(actions),
        "most_frequent_defects": [
            {"code": defect, "label": translate_defect(defect), "count": count}
            for defect, count in defects.most_common()
        ],
    }


def build_farmer_report(
    result: InspectionResult,
    farm_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the farmer-facing report while preserving the existing input contract."""
    object_reports = [
        build_object_report(item, index + 1)
        for index, item in enumerate(result.items)
    ]
    summary = build_lot_summary(object_reports)
    report: dict[str, Any] = {
        "decision": build_lot_decision(summary),
        "summary": summary,
        "actions": build_actions(object_reports),
        "objects": object_reports,
        "warnings": build_warnings(result, object_reports),
        "technical_details": build_technical_details(result),
        "legacy_metrics": build_legacy_metrics(result, object_reports),
    }
    if farm_context:
        report["farm_context"] = farm_context
    return report
