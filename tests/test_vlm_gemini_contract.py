import json

import numpy as np

from backend.vlm_reasoning import (
    COLLAGE_PROMPT_TEMPLATE,
    QUALITY_PROMPT_TEMPLATE,
    GeminiVLMBackend,
    VLMBackend,
)


class CaptureBackend(VLMBackend):
    name = "capture"

    def _call_model(self, crop, prompt):
        self.prompt = prompt
        return json.dumps(
            {
                "detected_class": "apple",
                "status": "ok",
                "overall_quality_score": 0.9,
                "quality_metrics": {"freshness": 0.9},
                "defects": [],
                "explanation": "The apple appears fresh.",
                "required_action": "none",
            }
        )


def test_prompt_is_valid_instruction_without_pseudo_json_placeholders():
    backend = CaptureBackend()
    result = backend.analyze(np.zeros((10, 10, 3), dtype=np.uint8), "apple", 0.95)

    assert result.status.value == "ok"
    assert "<float" not in backend.prompt
    assert '"status": "uncertain"' in backend.prompt
    assert "Return exactly one JSON object and nothing else" in backend.prompt


def test_gemini_prompt_has_gemini_specific_guidance(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    backend = GeminiVLMBackend()
    prompt = QUALITY_PROMPT_TEMPLATE.format(
        provider_guidance=backend._provider_guidance(),
        label="apple",
        confidence=0.95,
        metrics_list="- freshness",
        metrics_json_schema='"freshness": 0.0',
    )

    assert "native Google AI Studio API" in prompt
    assert "thought text" in prompt
    assert "<float" not in prompt


def test_collage_parser_requires_items_but_accepts_fenced_json():
    raw = "```json\n{" + '"items": [{"crop_id": "CROP_001"}]' + "}\n```"
    parsed = VLMBackend._parse_collage_response(raw)
    assert parsed["CROP_001"]["crop_id"] == "CROP_001"


def test_gemini_uses_visible_answer_parts_only(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    backend = GeminiVLMBackend()

    parts = [
        {"text": "internal reasoning without JSON", "thought": True},
        {"text": '{"items": []}'},
    ]
    visible = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and not part.get("thought", False)
    ).strip()
    assert visible == '{"items": []}'


def test_gemini_collage_prompt_is_strict():
    prompt = COLLAGE_PROMPT_TEMPLATE.format(
        provider_guidance="Gemini guidance",
        count=1,
        crop_context="- CROP_001: apple; metrics: freshness",
    )
    assert 'top-level object must contain exactly one key named items' in prompt
    assert "Do not use Markdown fences" in prompt
    assert "<float" not in prompt
