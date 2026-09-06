"""
VLM reasoning stage.

Provides a backend-agnostic interface so the pipeline can call
            "detected_class": "apple",
`backend.analyze(crop, label, confidence)` without knowing which hosted
provider is used.

Both backends are expected to return a QualityAssessment. Model calls are
wrapped in structured-output parsing with a fallback to an UNCERTAIN status
if the response can't be parsed cleanly -- a VLM response that fails schema
validation should never crash the pipeline.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from dotenv import load_dotenv
from backend.quality_profiles import get_quality_metrics
from backend.schemas import InspectionStatus, QualityAssessment, RequiredAction

# Ensure environment variables are loaded
load_dotenv()

# Production quality reasoning is intentionally routed through OpenRouter.
ACTIVE_OPENROUTER_MODEL = "openrouter/free"

QUALITY_PROMPT_TEMPLATE = """{provider_guidance}

Inspecte le recadrage fourni. Le détecteur l’a identifié comme « {label} » avec une confiance de {confidence:.2f}.
Assess only visible quality evidence. The applicable quality metrics are:
{metrics_list}

Return exactly one JSON object and nothing else. Do not use Markdown fences, commentary, or a preamble. Rédige toutes les valeurs textuelles destinées à l’utilisateur en français.
Use these exact keys: detected_class, status, overall_quality_score, quality_metrics, defects, explanation, required_action.
Allowed status values are exactly: ok, defect, uncertain.
Allowed required_action values are exactly: none, flag_for_review, remove.
Every quality_metrics value is a direct defect-likelihood probability from 0.0 to 1.0 for that specific metric, not a general quality score. Interpret each metric independently: 0.0 means the named defect or condition is maximally unlikely or absent, 1.0 means it is maximally likely or visibly present, and intermediate values represent the corresponding likelihood. Higher values therefore mean higher likelihood that the named defect exists. The defects array must contain only defects visibly supported by the crop. Use null only for overall_quality_score when evidence is insufficient.
For status=ok, defects must be empty and the explanation must say in French that no visible defects were found. For status=defect, defects must contain every visibly supported defect and the explanation must mention them in French using concrete visual evidence. For status=uncertain, overall_quality_score must be null, quality_metrics must be empty, required_action must be flag_for_review, and the explanation must identify the limitation in French. Blur, occlusion, very small crops, poor lighting, or insufficient visual evidence must never produce a confident OK result. Do not mention blur, uncertainty, or insufficient evidence in an OK explanation, and do not claim no defects when defects is non-empty.
The explanation must be one concise sentence grounded in visible evidence.
The JSON object must have this shape:
{{"detected_class": "{label}", "status": "uncertain", "overall_quality_score": null, "quality_metrics": {{{metrics_json_schema}}}, "defects": [], "explanation": "Éléments visuels insuffisants.", "required_action": "flag_for_review"}}"""

COLLAGE_PROMPT_TEMPLATE = """{provider_guidance}

Inspect the attached collage of {count} labeled food-item crops.
Analyze each panel independently. Each panel has a printed crop_id. Do not merge panels or omit any crop_id.

Crop instructions:
{crop_context}

Return exactly one JSON object and nothing else. Do not use Markdown fences, commentary, or a preamble. Rédige toutes les valeurs textuelles destinées à l’utilisateur en français.
The top-level object must contain exactly one key named items. items must be an array with one object for every crop_id.
Each item must contain exactly these keys: crop_id, status, overall_quality_score, quality_metrics, defects, explanation, required_action.
Allowed status values are exactly: ok, defect, uncertain.
Allowed required_action values are exactly: none, flag_for_review, remove.
Every quality_metrics value is a direct defect-likelihood probability from 0.0 to 1.0 for that specific metric, not a general quality score. Interpret each metric independently: 0.0 means the named defect or condition is maximally unlikely or absent, 1.0 means it is maximally likely or visibly present, and intermediate values represent the corresponding likelihood. Higher values therefore mean higher likelihood that the named defect exists. The defects array must contain only defects visibly supported by the crop. Use null only for overall_quality_score when evidence is insufficient.
For status=ok, defects must be empty and the explanation must say in French that no visible defects were found. For status=defect, defects must contain every visibly supported defect and the explanation must mention them in French using concrete visual evidence. For status=uncertain, overall_quality_score must be null, quality_metrics must be empty, required_action must be flag_for_review, and the explanation must identify the limitation in French. Blur, occlusion, very small crops, poor lighting, or insufficient visual evidence must never produce a confident OK result. Do not mention blur, uncertainty, or insufficient evidence in an OK explanation, and do not claim no defects when defects is non-empty.
The explanation must be one concise sentence grounded in visible evidence.
Use this shape, replacing the example with the actual crop IDs:
{{"items": [{{"crop_id": "CROP_001", "status": "uncertain", "overall_quality_score": null, "quality_metrics": {{}}, "defects": [], "explanation": "Éléments visuels insuffisants.", "required_action": "flag_for_review"}}]}}"""


class VLMBackend(ABC):
    """Common interface for all VLM backends."""

    name: str = "base"

    @staticmethod
    def _extract_message_content(response) -> str:
        """Validate provider output before JSON parsing so failures are visible."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("VLM provider returned no completion choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
        if not isinstance(content, str) or not content.strip():
            refusal = getattr(message, "refusal", None) if message is not None else None
            detail = f" Refusal: {refusal}" if refusal else ""
            raise RuntimeError(f"VLM provider returned an empty response.{detail}")
        return content

    @abstractmethod
    def _call_model(self, crop: np.ndarray, prompt: str) -> str:
        """Send the crop + prompt to the model, return raw text response."""
        raise NotImplementedError

    def _provider_guidance(self) -> str:
        return "Respecte exactement le format de sortie demandé et rédige les textes en français."

    def analyze(
        self, crop: np.ndarray, label: str, confidence: float
    ) -> QualityAssessment:
        metrics = get_quality_metrics(label)
        metrics_list = "\n".join(f"- {m}" for m in metrics)
        metrics_json_schema = ", ".join(f'"{m}": 0.0' for m in metrics)

        prompt = QUALITY_PROMPT_TEMPLATE.format(
            provider_guidance=self._provider_guidance(),
            label=label,
            confidence=confidence,
            metrics_list=metrics_list,
            metrics_json_schema=metrics_json_schema,
        )

        print(f"[VLM] Analyzing {label} (conf: {confidence:.2f})...")
        start = time.perf_counter()
        try:
            raw = self._call_model(crop, prompt)
            parsed = self._parse_response(raw)
            print(f"[VLM] Result for {label}: {parsed.status.value.upper()} (Score: {parsed.overall_quality_score})")
        except Exception as exc:  # noqa: BLE001 - keep one failed crop from aborting the frame
            print(f"[VLM] Error analyzing {label}: {exc}")
            parsed = self._fallback_assessment(str(exc))
        
        latency_ms = (time.perf_counter() - start) * 1000
        parsed.vlm_backend = self.name
        parsed.latency_ms = latency_ms
        return parsed

    def analyze_collage(
        self, collage: np.ndarray, crops: list[dict[str, object]]
    ) -> dict[str, QualityAssessment]:
        crop_context = "\n".join(
            f"- {crop['crop_id']}: {crop['label']} (detector confidence: {float(crop['confidence']):.2f}); metrics: {', '.join(crop['metrics'])}"
            for crop in crops
        )
        prompt = COLLAGE_PROMPT_TEMPLATE.format(
            provider_guidance=self._provider_guidance(),
            count=len(crops),
            crop_context=crop_context,
        )
        start = time.perf_counter()
        try:
            raw = self._call_model(collage, prompt)
            data = self._parse_collage_response(raw)
            assessments: dict[str, QualityAssessment] = {}
            for crop in crops:
                crop_id = str(crop['crop_id'])
                item = data.get(crop_id)
                try:
                    if item is None:
                        raise ValueError(f"La réponse VLM ne contient pas {crop_id}.")
                    assessments[crop_id] = self._assessment_from_data(item)
                except Exception as exc:  # noqa: BLE001 - isolate malformed crop results
                    assessments[crop_id] = self._fallback_assessment(str(exc))
                assessments[crop_id].vlm_backend = self.name
                assessments[crop_id].latency_ms = (time.perf_counter() - start) * 1000
            return assessments
        except Exception as exc:  # noqa: BLE001 - preserve one collage failure per crop
            return {
                str(crop['crop_id']): self._with_metadata(self._fallback_assessment(str(exc)), start)
                for crop in crops
            }

    @staticmethod
    def _assessment_from_data(data: dict) -> QualityAssessment:
        status_val = str(data.get("status", "uncertain")).lower()
        if status_val not in [status.value for status in InspectionStatus]:
            status_val = "uncertain"
        assessment = QualityAssessment(
            detected_class=data.get("detected_class"),
            status=InspectionStatus(status_val),
            overall_quality_score=data.get("overall_quality_score"),
            quality_metrics=VLMBackend._numeric_metrics(data.get("quality_metrics", {})),
            defects=data.get("defects", []),
            explanation=data.get("explanation", ""),
            required_action=RequiredAction(data.get("required_action", "none")),
            vlm_backend="pending",
        )
        return VLMBackend._normalize_assessment(assessment)

    @staticmethod
    def _parse_collage_response(raw: str) -> dict[str, dict]:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise ValueError("La réponse VLM ne contient pas d’objet JSON.")
        payload = json.loads(cleaned[start_idx:end_idx + 1])
        return {
            str(item["crop_id"]): item
            for item in payload.get("items", [])
            if isinstance(item, dict) and item.get("crop_id")
        }

    def _with_metadata(self, assessment: QualityAssessment, started: float) -> QualityAssessment:
        assessment.vlm_backend = self.name
        assessment.latency_ms = (time.perf_counter() - started) * 1000
        return assessment

    @staticmethod
    def _numeric_metrics(metrics: object) -> dict[str, float]:
        if not isinstance(metrics, dict):
            return {}
        return {
            str(name): max(0.0, min(1.0, float(value)))
            for name, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    @staticmethod
    def _normalize_assessment(assessment: QualityAssessment) -> QualityAssessment:
        if assessment.status == InspectionStatus.UNCERTAIN:
            assessment.overall_quality_score = None
            assessment.quality_metrics = {}
            assessment.required_action = RequiredAction.FLAG_FOR_REVIEW
        elif assessment.status == InspectionStatus.OK and assessment.defects:
            assessment.status = InspectionStatus.DEFECT
            if assessment.required_action == RequiredAction.NONE:
                assessment.required_action = RequiredAction.FLAG_FOR_REVIEW
        elif assessment.status == InspectionStatus.DEFECT and not assessment.defects:
            assessment.status = InspectionStatus.UNCERTAIN
            assessment.overall_quality_score = None
            assessment.quality_metrics = {}
            assessment.required_action = RequiredAction.FLAG_FOR_REVIEW
        return assessment

    @staticmethod
    def _parse_response(raw: str) -> QualityAssessment:
        # Models sometimes wrap JSON in markdown fences despite instructions.
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        # Find the first '{' and last '}' to handle potential prefix/suffix text
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx:end_idx+1]

        data = json.loads(cleaned)
        
        # Normalize status
        status_val = data.get("status", "uncertain").lower()
        if status_val not in [s.value for s in InspectionStatus]:
            status_val = "uncertain"

        assessment = QualityAssessment(
            detected_class=data.get("detected_class"),
            status=InspectionStatus(status_val),
            overall_quality_score=data.get("overall_quality_score"),
            quality_metrics=VLMBackend._numeric_metrics(data.get("quality_metrics", {})),
            defects=data.get("defects", []),
            explanation=data.get("explanation", ""),
            required_action=RequiredAction(data.get("required_action", "none")),
            vlm_backend="pending",
        )
        return VLMBackend._normalize_assessment(assessment)

    @staticmethod
    def _fallback_assessment(error: str) -> QualityAssessment:
        return QualityAssessment(
            status=InspectionStatus.UNCERTAIN,
            overall_quality_score=None,
            quality_metrics={},
            defects=[],
            explanation=f"La réponse du modèle n’a pas pu être interprétée : {error}",
            required_action=RequiredAction.FLAG_FOR_REVIEW,
            vlm_backend="pending",
        )


class GPT4VLMBackend(VLMBackend):
    """API-based inference via GPT-4o vision."""

    name = "gpt-4o"

    def _provider_guidance(self) -> str:
        return "Tu es un classificateur visuel utilisant une API compatible OpenAI. Respecte exactement le contrat JSON, renvoie uniquement le JSON et rédige les textes en français."

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        import os

        from openai import OpenAI

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY introuvable dans l’environnement ou les arguments.")

        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        self.model = model

    def _call_model(self, crop: np.ndarray, prompt: str) -> str:
        import base64

        import cv2

        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError("Impossible d’encoder le recadrage en JPEG")
        b64_image = base64.b64encode(buf).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return self._extract_message_content(response)



class GeminiVLMBackend(VLMBackend):
    """Google AI Studio / Gemini API vision backend."""

    name = "gemma-api"

    def _provider_guidance(self) -> str:
        return (
            "Tu es Gemma 4 exécuté via l’API Google AI Studio. "
            "Ne révèle pas ton raisonnement. Renvoie uniquement l’objet JSON final avec des textes en français."
        )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        import os

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMMA_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY ou GEMMA_API_KEY introuvable dans l’environnement."
            )

        self.base_url = os.getenv(
            "GEMMA_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
         ).rstrip("/")

        self.model = (
            model
            or os.getenv("GEMMA_MODEL")
            or "gemma-4-31b-it"
        ).removeprefix("models/")

    def _call_model(self, crop: np.ndarray, prompt: str) -> str:
        import base64
        import cv2
        import requests

        if crop is None or getattr(crop, "size", 0) == 0:
            raise ValueError("Cannot send an empty image crop to Gemini.")

        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError("Impossible d’encoder le recadrage en JPEG")

        response = requests.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(buf).decode("ascii"),
                                }
                            },
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 600,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingLevel": "MINIMAL"},
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        candidates = payload.get("candidates") or []
        if not candidates:
            block_reason = payload.get("promptFeedback", {}).get("blockReason")
            detail = f" Prompt blocked: {block_reason}." if block_reason else ""
            raise RuntimeError(f"Gemini returned no candidates.{detail}")

        parts = candidates[0].get("content", {}).get("parts", [])
        visible_text = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and not part.get("thought", False)
        ]
        text = "".join(visible_text).strip()
        if not text:
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            ).strip()

        if not text:
            finish_reason = candidates[0].get("finishReason")
            detail = f" Finish reason: {finish_reason}." if finish_reason else ""
            raise RuntimeError(f"Gemini returned no text content.{detail}")

        return text


class OpenRouterBackend(VLMBackend):
    # The model can be switched to:
    # - nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
    # - google/gemma-4-31b-it:free
    """Single OpenRouter implementation for the approved hosted VLM models."""

    name = "openrouter"

    def _provider_guidance(self) -> str:
        return "Tu es un modèle multimodal hébergé derrière une API compatible OpenAI. Renvoie uniquement l’objet JSON demandé et rédige les textes en français."

    def __init__(self, api_key: Optional[str] = None, model: str = ACTIVE_OPENROUTER_MODEL):
        import os
        from openai import OpenAI

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY introuvable dans l’environnement.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
            max_retries=1,
        )
        self.model = model

    def _call_model(self, crop: np.ndarray, prompt: str) -> str:
        import base64
        import cv2

        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError("Impossible d’encoder le recadrage en JPEG")
        b64_image = base64.b64encode(buf).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return self._extract_message_content(response)


def get_backend(name: str, model: Optional[str] = None) -> VLMBackend:
    """Return only the Google AI Studio Gemini backend."""
    return GeminiVLMBackend()


