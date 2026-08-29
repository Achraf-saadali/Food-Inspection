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

QUALITY_PROMPT_TEMPLATE = """You are inspecting a food item detected by an object \
detector as "{label}" (detector confidence: {confidence:.2f}).

Inspect the image crop for visible quality defects and assess specific metrics.

Relevant quality metrics for {label}:
{metrics_list}

Respond with ONLY a JSON object, no other text, matching exactly this shape:
{{
    "detected_class": "{label}",
  "status": "ok" | "defect" | "uncertain",
  "overall_quality_score": <float 0.0-1.0, where 1.0 is perfect quality>,
  "quality_metrics": {{
    {metrics_json_schema}
  }},
  "defects": ["<defect1>", "<defect2>", ...],
  "explanation": "<one sentence, concrete visual evidence>",
  "required_action": "none" | "flag_for_review" | "remove"
}}"""

COLLAGE_PROMPT_TEMPLATE = """You are inspecting {count} labeled food-item crops in one collage.

Analyze every crop independently. Each crop has a unique crop_id printed in its panel.
Do not merge crops or omit a crop. Return exactly one result for every crop_id.

For each crop, assess these metrics:
{crop_context}

Respond with ONLY a JSON object matching exactly this shape:
{{
    "items": [
        {{
            "crop_id": "CROP_001",
            "status": "ok" | "defect" | "uncertain",
            "overall_quality_score": <float 0.0-1.0 or null>,
            "quality_metrics": {{"metric": <float 0.0-1.0>}},
            "defects": ["<defect>"],
            "explanation": "<one concise sentence>",
            "required_action": "none" | "flag_for_review" | "remove"
        }}
    ]
}}"""


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

    def analyze(
        self, crop: np.ndarray, label: str, confidence: float
    ) -> QualityAssessment:
        # Select quality dimensions from the detected ingredient class.
        metrics = get_quality_metrics(label)
        metrics_list = "\n".join([f"- {m}" for m in metrics])
        metrics_json_schema = ",\n    ".join([f'"{m}": <float 0.0-1.0>' for m in metrics])
        
        prompt = QUALITY_PROMPT_TEMPLATE.format(
            label=label, 
            confidence=confidence,
            metrics_list=metrics_list,
            metrics_json_schema=metrics_json_schema
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
        prompt = COLLAGE_PROMPT_TEMPLATE.format(count=len(crops), crop_context=crop_context)
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
                        raise ValueError(f"VLM response omitted {crop_id}.")
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
        return QualityAssessment(
            detected_class=data.get("detected_class"),
            status=InspectionStatus(status_val),
            overall_quality_score=data.get("overall_quality_score"),
            quality_metrics=VLMBackend._numeric_metrics(data.get("quality_metrics", {})),
            defects=data.get("defects", []),
            explanation=data.get("explanation", ""),
            required_action=RequiredAction(data.get("required_action", "none")),
            vlm_backend="pending",
        )

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
            raise ValueError("VLM response did not contain a JSON object.")
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
            str(name): float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

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

        return QualityAssessment(
            detected_class=data.get("detected_class"),
            status=InspectionStatus(status_val),
            overall_quality_score=data.get("overall_quality_score"),
            quality_metrics=VLMBackend._numeric_metrics(data.get("quality_metrics", {})),
            defects=data.get("defects", []),
            explanation=data.get("explanation", ""),
            required_action=RequiredAction(data.get("required_action", "none")),
            vlm_backend="pending",
        )

    @staticmethod
    def _fallback_assessment(error: str) -> QualityAssessment:
        return QualityAssessment(
            status=InspectionStatus.UNCERTAIN,
            overall_quality_score=None,
            quality_metrics={},
            defects=[],
            explanation=f"VLM response could not be parsed: {error}",
            required_action=RequiredAction.FLAG_FOR_REVIEW,
            vlm_backend="pending",
        )


class GPT4VLMBackend(VLMBackend):
    """API-based inference via GPT-4o vision. Used as the high-accuracy
    reference model, per the benchmark rationale in the README."""

    name = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        import os

        from openai import OpenAI

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or arguments.")

        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        self.model = model

    def _call_model(self, crop: np.ndarray, prompt: str) -> str:
        import base64

        import cv2

        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError("Failed to encode crop as JPEG")
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
                "GEMINI_API_KEY or GEMMA_API_KEY not found in environment."
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
            raise RuntimeError("Failed to encode crop as JPEG")

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
                    "maxOutputTokens": 300,
                    "responseMimeType": "application/json",
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

    def __init__(self, api_key: Optional[str] = None, model: str = ACTIVE_OPENROUTER_MODEL):
        import os
        from openai import OpenAI

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment.")

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
            raise RuntimeError("Failed to encode crop as JPEG")
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
    return GeminiVLMBackend(model=model)



