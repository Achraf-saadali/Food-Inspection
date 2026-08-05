"""
VLM reasoning stage.

Provides a backend-agnostic interface so the pipeline can call
`backend.analyze(crop, label, confidence)` without knowing whether the
underlying model is Qwen2.5-VL (local) or GPT-4o (API).

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

from quality_profiles import get_quality_metrics
from schemas import InspectionStatus, QualityAssessment, RequiredAction

QUALITY_PROMPT_TEMPLATE = """You are inspecting a food item detected by an object \
detector as "{label}" (detector confidence: {confidence:.2f}).

Inspect the image crop for visible quality defects and assess specific metrics.

Relevant quality metrics for {label}:
{metrics_list}

Respond with ONLY a JSON object, no other text, matching exactly this shape:
{{
  "status": "ok" | "defect" | "uncertain",
  "overall_quality_score": <float 0.0-1.0, where 1.0 is perfect quality>,
  "quality_metrics": {{
    {metrics_json_schema}
  }},
  "defects": ["<defect1>", "<defect2>", ...],
  "explanation": "<one sentence, concrete visual evidence>",
  "required_action": "none" | "flag_for_review" | "remove"
}}"""


class VLMBackend(ABC):
    """Common interface for all VLM backends."""

    name: str = "base"

    @abstractmethod
    def _call_model(self, crop: np.ndarray, prompt: str) -> str:
        """Send the crop + prompt to the model, return raw text response."""
        raise NotImplementedError

    def analyze(
        self, crop: np.ndarray, label: str, confidence: float
    ) -> QualityAssessment:
        metrics = get_quality_metrics(label)
        metrics_list = "\n".join([f"- {m}" for m in metrics])
        metrics_json_schema = ",\n    ".join([f'"{m}": <float 0.0-1.0>' for m in metrics])
        
        prompt = QUALITY_PROMPT_TEMPLATE.format(
            label=label, 
            confidence=confidence,
            metrics_list=metrics_list,
            metrics_json_schema=metrics_json_schema
        )

        start = time.perf_counter()
        try:
            raw = self._call_model(crop, prompt)
            parsed = self._parse_response(raw)
        except Exception as exc:  # noqa: BLE001 - VLM failures must degrade gracefully
            parsed = self._fallback_assessment(str(exc))
        latency_ms = (time.perf_counter() - start) * 1000

        parsed.vlm_backend = self.name
        parsed.latency_ms = latency_ms
        return parsed

    @staticmethod
    def _parse_response(raw: str) -> QualityAssessment:
        # Models sometimes wrap JSON in markdown fences despite instructions.
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return QualityAssessment(
            status=InspectionStatus(data["status"]),
            overall_quality_score=data.get("overall_quality_score"),
            quality_metrics=data.get("quality_metrics", {}),
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


class Qwen25VLBackend(VLMBackend):
    """Local inference via Qwen2.5-VL. Loads the model once and reuses it
    across calls -- construct one instance per process, not per request."""

    name = "qwen2.5-vl"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda"):
        # Imports are deferred so this module doesn't hard-require
        # transformers/torch just to import schemas or other backends.
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _call_model(self, crop: np.ndarray, prompt: str) -> str:
        from PIL import Image

        image = Image.fromarray(crop[:, :, ::-1])  # BGR (cv2) -> RGB
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt").to(
            self.model.device
        )
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]


class GPT4oBackend(VLMBackend):
    """API-based inference via GPT-4o vision. Used as the high-accuracy
    reference model, per the benchmark rationale in the README."""

    name = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        import os

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
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
        return response.choices[0].message.content


def get_backend(name: str) -> VLMBackend:
    """Factory so the pipeline/API can select a backend by string flag."""
    if name == "qwen":
        return Qwen25VLBackend()
    if name == "gpt4o":
        return GPT4oBackend()
    raise ValueError(f"Unknown VLM backend: {name}")
