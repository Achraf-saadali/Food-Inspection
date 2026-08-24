"""Live camera runner for the unified food inspection pipeline.

Unlike the legacy ``backend/main.py`` loop, this runner deliberately calls
``run_inspection`` for every processed frame. Consequently, camera execution
and the FastAPI ``/inspect`` endpoint share the same detection, VLM prompt,
response parsing, farmer commentary, and typed result contract.

Example:
    python -m backend.live_inspection --source 0 --vlm-backend openrouter
    python -m backend.live_inspection --source sample.mp4 --vlm-backend gemma
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.inspection_pipeline import run_inspection
from backend.reporting import build_farmer_report


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def run_live(
    source: str | int,
    model_path: str,
    vlm_backend_name: str | None,
    vlm_model: str | None,
    confidence_gate: float,
    output_path: Path | None,
    display: bool,
    max_frames: int | None,
) -> None:
    import cv2
    from ultralytics import YOLO
    from backend.vlm_reasoning import get_backend

    model = YOLO(model_path)
    vlm_backend = get_backend(vlm_backend_name, model=vlm_model) if vlm_backend_name else None
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {source}")

    output_file = output_path.open("a", encoding="utf-8") if output_path else None
    frame_id = 0
    try:
        while max_frames is None or frame_id < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            started = time.perf_counter()
            result = run_inspection(
                image=frame,
                yolo_model=model,
                frame_id=frame_id,
                source=str(source),
                vlm_backend=vlm_backend,
                vlm_confidence_gate=confidence_gate,
            )
            payload = {
                "inspection": result.model_dump(mode="json"),
                "farmer_report": build_farmer_report(result),
                "processing_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            if output_file:
                output_file.write(json.dumps(payload) + "\n")
                output_file.flush()
            print(json.dumps(payload, ensure_ascii=False))

            if display:
                for item in result.items:
                    x1, y1, x2, y2 = [int(v) for v in item.detection.bbox_xyxy]
                    status = item.quality.status.value
                    color = (0, 180, 0) if status == "ok" else (0, 0, 255) if status == "defect" else (0, 190, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{item.detection.label} {item.detection.confidence:.2f} | {status}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow("Food Inspection - Shared Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if output_file:
            output_file.close()
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified food detection and inspection pipeline on a camera or video.")
    parser.add_argument("--source", default="0", help="Camera index or video/image path.")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--vlm-backend", default=None, help="Registered backend name, for example openrouter, gemma, or nvidia.")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--confidence-gate", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL report path.")
    parser.add_argument("--display", action="store_true", help="Show annotated frames in an OpenCV window.")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    run_live(parse_source(args.source), args.model, args.vlm_backend, args.vlm_model, args.confidence_gate, args.output, args.display, args.max_frames)


if __name__ == "__main__":
    main()
