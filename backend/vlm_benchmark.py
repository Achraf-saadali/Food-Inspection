"""Benchmark the three candidate free OpenRouter VLMs on identical crops.

This is an evaluation harness, not a replacement for the production pipeline.
Every candidate uses the OpenRouter adapter with a different model ID, while
all candidates share the same prompt, parser, fallback behavior, and schema.
API keys are read from the local environment and are never stored in the repo.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from backend.vlm_reasoning import get_backend

DEFAULT_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def run(image_dir: Path, label: str, models: list[str], output: Path) -> None:
    """Run every candidate over the same sorted image set.

    This records operational comparison data. It does not measure correctness
    without human reference labels or a task-specific evaluation rubric.
    """
    import cv2

    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    # Every instance uses OpenRouter; only the model ID changes.
    instances = {model: get_backend("openrouter", model=model) for model in models}
    rows: list[dict[str, object]] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for model, backend in instances.items():
            started = time.perf_counter()
            assessment = backend.analyze(image, label, confidence=1.0)
            wall_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "image": image_path.name,
                    "label": label,
                    "provider": "openrouter",
                    "model": model,
                    "status": assessment.status.value,
                    "overall_quality_score": assessment.overall_quality_score,
                    "defects": "|".join(assessment.defects),
                    "required_action": assessment.required_action.value,
                    "parse_or_provider_latency_ms": assessment.latency_ms,
                    "wall_latency_ms": round(wall_ms, 2),
                    "has_explanation": bool(assessment.explanation),
                    "explanation": assessment.explanation,
                }
            )

    if not rows:
        raise ValueError("No readable benchmark images were found")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} benchmark rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the three free OpenRouter VLM candidates on identical crops."
    )
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--label", default="food_item")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Candidate models; defaults to all three approved free models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_artifacts/vlm_benchmark_openrouter.csv"),
    )
    args = parser.parse_args()
    run(args.image_dir, args.label, args.models, args.output)


if __name__ == "__main__":
    main()
