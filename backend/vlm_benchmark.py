"""Benchmark registered VLM backends on identical image crops.

This is an evaluation harness, not a replacement for the production pipeline.
It uses the same VLMBackend.analyze() method and therefore the same prompt,
JSON parser, fallback behavior, and quality schema as application inference.
API keys are read from environment variables; no key is stored in the repo.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

def run(image_dir: Path, backends: list[str], label: str, model_names: dict[str, str | None], output: Path) -> None:
    import cv2
    from backend.vlm_reasoning import get_backend

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    rows: list[dict[str, object]] = []
    instances = {name: get_backend(name, model=model_names.get(name)) for name in backends}
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for name, backend in instances.items():
            started = time.perf_counter()
            assessment = backend.analyze(image, label, confidence=1.0)
            wall_ms = (time.perf_counter() - started) * 1000
            rows.append({
                "image": image_path.name,
                "backend": name,
                "model": model_names.get(name) or "default",
                "status": assessment.status.value,
                "overall_quality_score": assessment.overall_quality_score,
                "defects": "|".join(assessment.defects),
                "required_action": assessment.required_action.value,
                "parse_or_provider_latency_ms": assessment.latency_ms,
                "wall_latency_ms": round(wall_ms, 2),
                "has_explanation": bool(assessment.explanation),
                "explanation": assessment.explanation,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} benchmark rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gemma and NVIDIA VLM backends on identical crops.")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--label", default="food_item")
    parser.add_argument("--backends", nargs="+", default=["gemma", "nvidia"])
    parser.add_argument("--gemma-model", default=None)
    parser.add_argument("--nvidia-model", default=None)
    parser.add_argument("--output", type=Path, default=Path("runtime_artifacts/vlm_benchmark.csv"))
    args = parser.parse_args()
    model_names = {"gemma": args.gemma_model, "gemma-api": args.gemma_model, "nvidia": args.nvidia_model, "nvidia-api": args.nvidia_model}
    run(args.image_dir, args.backends, args.label, model_names, args.output)


if __name__ == "__main__":
    main()
