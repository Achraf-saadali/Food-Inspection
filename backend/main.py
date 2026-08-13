import argparse
import json
import os
import textwrap
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from backend.inspection_pipeline import build_quality_commentary, crop_detection
from backend.schemas import (
    Detection,
    ImageSize,
    InspectionItem,
    InspectionResult,
    InspectionStatus,
    QualityAssessment,
    RequiredAction,
)
from backend.vlm_reasoning import get_backend

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_PATH = "models/best.pt"
DEFAULT_CONF_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_VLM_CONF_GATE = 0.6
VLM_REFRESH_SECONDS = 1.5
WAITING_EXPLANATION = "Waiting for VLM inspection."

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

OUTPUT_DIR = PROJECT_ROOT / "runtime_artifacts" / "outputs"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
LOG_DIR = OUTPUT_DIR / "logs"

WINDOW_NAME = "Food Inspection - Live Quality Analysis"


class FoodInspectionApp:
    """Real-time detector with asynchronous VLM quality updates per tracked item."""

    def __init__(
        self,
        model_path,
        source=0,
        conf=0.4,
        iou=0.5,
        vlm_backend_name=None,
        vlm_conf_gate=0.6,
        vlm_model=None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")

        print(f"[INFO] Loading YOLO model from {self.model_path} ...")
        self.model = YOLO(str(self.model_path))

        self.vlm_backend = None
        if vlm_backend_name:
            msg = f"[INFO] Initializing VLM backend: {vlm_backend_name}"
            if vlm_model:
                msg += f" (model: {vlm_model})"
            print(f"{msg} ...")
            try:
                self.vlm_backend = get_backend(vlm_backend_name, model=vlm_model)
            except Exception as exc:
                print(f"[ERROR] Failed to load VLM backend: {exc}")
                print("[WARN] Proceeding with detection only.")

        self.source = source
        self.conf = conf
        self.iou = iou
        self.vlm_conf_gate = vlm_conf_gate
        self.cap = None
        self.frame_id = 0
        self._next_track_id = 0
        self._tracks: Dict[str, dict] = {}
        self._vlm_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-inspection")
            if self.vlm_backend
            else None
        )

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = LOG_DIR / f"session_{timestamp}.jsonl"
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        print(f"[INFO] Logging results to {self.log_path}")

    def open_capture(self):
        source = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    @staticmethod
    def _iou(first_bbox: List[float], second_bbox: List[float]) -> float:
        """Return overlap for matching the same object across adjacent frames."""
        left = max(first_bbox[0], second_bbox[0])
        top = max(first_bbox[1], second_bbox[1])
        right = min(first_bbox[2], second_bbox[2])
        bottom = min(first_bbox[3], second_bbox[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        if intersection <= 0:
            return 0.0
        first_area = max(0.0, first_bbox[2] - first_bbox[0]) * max(0.0, first_bbox[3] - first_bbox[1])
        second_area = max(0.0, second_bbox[2] - second_bbox[0]) * max(0.0, second_bbox[3] - second_bbox[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _match_track(self, label: str, bbox_xyxy: List[float]) -> str:
        """Reuse recent quality work when the same labelled object remains visible."""
        now = time.monotonic()
        best_track_id = None
        best_overlap = 0.2
        for track_id, track in self._tracks.items():
            if track["label"] != label or now - track["last_seen"] > 3.0:
                continue
            overlap = self._iou(bbox_xyxy, track["bbox_xyxy"])
            if overlap > best_overlap:
                best_track_id = track_id
                best_overlap = overlap

        if best_track_id is None:
            self._next_track_id += 1
            best_track_id = f"{label}-{self._next_track_id}"
            self._tracks[best_track_id] = {
                "label": label,
                "bbox_xyxy": bbox_xyxy,
                "last_seen": now,
                "quality": None,
                "future": None,
                "last_vlm_at": 0.0,
            }
        else:
            self._tracks[best_track_id]["bbox_xyxy"] = bbox_xyxy
            self._tracks[best_track_id]["last_seen"] = now
        return best_track_id

    def _collect_finished_inspections(self):
        """Promote completed background VLM calls into displayable quality results."""
        for track in self._tracks.values():
            future: Optional[Future] = track.get("future")
            if future is None or not future.done():
                continue
            track["future"] = None
            try:
                quality = future.result()
            except Exception as exc:  # Defensive fallback keeps the camera loop running.
                quality = QualityAssessment(
                    status=InspectionStatus.UNCERTAIN,
                    overall_quality_score=None,
                    quality_metrics={},
                    defects=[],
                    explanation=f"VLM inspection failed: {exc}",
                    required_action=RequiredAction.FLAG_FOR_REVIEW,
                    vlm_backend=self.vlm_backend.name if self.vlm_backend else "none",
                )
            quality.commentary = build_quality_commentary(track["label"], quality)
            track["quality"] = quality

        now = time.monotonic()
        stale_tracks = [track_id for track_id, track in self._tracks.items() if now - track["last_seen"] > 5.0]
        for track_id in stale_tracks:
            self._tracks.pop(track_id, None)

    def _waiting_quality(self) -> QualityAssessment:
        return QualityAssessment(
            status=InspectionStatus.SKIPPED,
            overall_quality_score=None,
            quality_metrics={},
            defects=[],
            explanation=WAITING_EXPLANATION,
            commentary="Waiting for inspection. The detected item is queued for VLM quality analysis.",
            required_action=RequiredAction.NONE,
            vlm_backend=self.vlm_backend.name if self.vlm_backend else "none",
        )

    def _skipped_quality(self, reason: str) -> QualityAssessment:
        return QualityAssessment(
            status=InspectionStatus.SKIPPED,
            overall_quality_score=None,
            quality_metrics={},
            defects=[],
            explanation=reason,
            commentary=reason,
            required_action=RequiredAction.NONE,
            vlm_backend=self.vlm_backend.name if self.vlm_backend else "none",
        )

    def _quality_for_detection(
        self,
        frame,
        label: str,
        confidence: float,
        bbox_xyxy: List[float],
    ) -> QualityAssessment:
        if self.vlm_backend is None or self._vlm_executor is None:
            return self._skipped_quality("Detection only: VLM inspection is disabled.")
        if confidence < self.vlm_conf_gate:
            return self._skipped_quality("Detection only: confidence is below the VLM inspection gate.")

        track_id = self._match_track(label, bbox_xyxy)
        track = self._tracks[track_id]
        future: Optional[Future] = track.get("future")
        now = time.monotonic()

        if future is None and (track["quality"] is None or now - track["last_vlm_at"] >= VLM_REFRESH_SECONDS):
            crop = crop_detection(frame, bbox_xyxy).copy()
            if crop.size > 0:
                # YOLO continues per frame while this single crop is inspected in the background.
                track["future"] = self._vlm_executor.submit(self.vlm_backend.analyze, crop, label, confidence)
                track["last_vlm_at"] = now
                return self._waiting_quality()

        if track.get("future") is not None:
            return self._waiting_quality()
        if track.get("quality") is not None:
            return track["quality"]
        return self._waiting_quality()

    def _build_live_result(self, frame) -> InspectionResult:
        """Run low-latency YOLO now and attach the latest asynchronous VLM state."""
        height, width = frame.shape[:2]
        yolo_result = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        items: List[InspectionItem] = []

        for box in yolo_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox_xyxy = box.xyxy[0].tolist()
            label = self.model.names[class_id]
            detection = Detection(
                label=label,
                class_id=class_id,
                confidence=confidence,
                bbox_xyxy=bbox_xyxy,
                bbox_normalized=[
                    bbox_xyxy[0] / width,
                    bbox_xyxy[1] / height,
                    bbox_xyxy[2] / width,
                    bbox_xyxy[3] / height,
                ],
            )
            quality = self._quality_for_detection(frame, label, confidence, bbox_xyxy)
            items.append(InspectionItem(detection=detection, quality=quality))

        return InspectionResult(
            frame_id=self.frame_id,
            timestamp=datetime.utcnow(),
            source=str(self.source),
            image_size=ImageSize(width=width, height=height),
            num_detections=len(items),
            items=items,
        )

    @staticmethod
    def _draw_text(frame, text: str, origin, color, scale=0.45, max_chars=58, line_gap=16):
        """Draw a short wrapped VLM response without hiding the detection box."""
        x, y = origin
        for line in textwrap.wrap(text, width=max_chars)[:3]:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += line_gap
        return y

    def annotate_frame(self, frame, result):
        """Show class/confidence immediately, waiting state next, then VLM results."""
        for item in result.items:
            detection = item.detection
            quality = item.quality
            x1, y1, x2, y2 = [int(value) for value in detection.bbox_xyxy]
            waiting = quality.explanation == WAITING_EXPLANATION

            if waiting:
                color = (0, 165, 255)  # Orange: VLM work is still in progress.
            elif quality.status == InspectionStatus.DEFECT:
                color = (0, 0, 255)
            elif quality.status == InspectionStatus.UNCERTAIN:
                color = (0, 255, 255)
            elif quality.status == InspectionStatus.SKIPPED:
                color = (128, 128, 128)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.label} {detection.confidence:.2f}"
            if waiting:
                label += " | WAITING FOR INSPECTION"
            elif quality.status == InspectionStatus.SKIPPED:
                label += " | DETECTION ONLY"
            else:
                label += f" | {quality.status.value.upper()}"
                if quality.overall_quality_score is not None:
                    label += f" {quality.overall_quality_score:.2f}"

            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_top = max(text_height + 8, y1 - 6)
            cv2.rectangle(frame, (x1, label_top - text_height - 8), (x1 + text_width + 8, label_top), color, -1)
            cv2.putText(frame, label, (x1 + 4, label_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            text_y = min(frame.shape[0] - 8, y2 + 18)
            if waiting:
                self._draw_text(frame, "Waiting for inspection...", (x1, text_y), color, scale=0.48)
                continue

            if quality.status == InspectionStatus.SKIPPED:
                self._draw_text(frame, quality.commentary, (x1, text_y), (200, 200, 200), scale=0.42)
                continue

            # Farmer commentary is the primary readable VLM response on the live feed.
            text_y = self._draw_text(frame, quality.commentary, (x1, text_y), color, scale=0.43)
            if quality.defects:
                defects = ", ".join(defect.replace("_", " ") for defect in quality.defects[:3])
                text_y = self._draw_text(frame, f"Defects: {defects}", (x1, text_y), (0, 0, 255), scale=0.42)
            if quality.explanation:
                self._draw_text(frame, f"VLM: {quality.explanation}", (x1, text_y), (235, 235, 235), scale=0.38)

        return frame

    def run(self):
        self.open_capture()
        print("[INFO] Live inspection started.")
        print("       YOLO labels appear immediately; VLM results update asynchronously.")
        print("       Press 's' to save current frame + JSON")
        print("       Press 'q' or ESC to quit")

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                self.frame_id += 1
                self._collect_finished_inspections()
                result = self._build_live_result(frame)
                annotated = self.annotate_frame(frame.copy(), result)
                cv2.imshow(WINDOW_NAME, annotated)

                payload = result.model_dump(mode="json")
                self.log_file.write(json.dumps(payload) + "\n")
                self.log_file.flush()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[INFO] Quit requested via keyboard.")
                    break
                if key in (ord("s"), ord("S")):
                    self.save_snapshot(annotated, payload)
        finally:
            self.cleanup()

    def save_snapshot(self, annotated_frame, payload):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_path = SNAPSHOT_DIR / f"frame_{timestamp}.jpg"
        json_path = SNAPSHOT_DIR / f"frame_{timestamp}.json"
        cv2.imwrite(str(image_path), annotated_frame)
        with open(json_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
        print(f"[SAVED] {image_path.name} + {json_path.name}")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self._vlm_executor:
            self._vlm_executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()
        if self.log_file:
            self.log_file.close()
        print("[INFO] Resources released.")


def main():
    parser = argparse.ArgumentParser(description="Food Quality Inspection System")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to YOLO weights")
    parser.add_argument("--source", default="0", help="Webcam index or video file")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="YOLO confidence")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="YOLO IoU threshold")
    parser.add_argument("--vlm", choices=["gpt4o", "openai", "qwen", "qwen-api", "openrouter"], help="Enable VLM reasoning backend")
    parser.add_argument("--vlm-model", help="Specific model ID to use with the VLM backend")
    parser.add_argument("--vlm-conf", type=float, default=DEFAULT_VLM_CONF_GATE, help="VLM confidence gate")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    qwen_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if args.vlm is None:
        if openai_key:
            print("[INFO] OPENAI_API_KEY detected. Defaulting to --vlm openai")
            args.vlm = "openai"
        elif qwen_key:
            print("[INFO] QWEN_API_KEY/DASHSCOPE_API_KEY detected. Defaulting to --vlm qwen-api")
            args.vlm = "qwen-api"
        elif openrouter_key:
            print("[INFO] OPENROUTER_API_KEY detected. Defaulting to --vlm openrouter")
            args.vlm = "openrouter"

    if args.vlm in ["gpt4o", "openai"] and not openai_key:
        print(f"[ERROR] --vlm {args.vlm} requested but OPENAI_API_KEY is not set.")
    elif args.vlm == "qwen-api" and not qwen_key:
        print(f"[ERROR] --vlm {args.vlm} requested but QWEN_API_KEY or DASHSCOPE_API_KEY is not set.")
    elif args.vlm == "openrouter" and not openrouter_key:
        print(f"[ERROR] --vlm {args.vlm} requested but OPENROUTER_API_KEY is not set.")

    app = FoodInspectionApp(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        vlm_backend_name=args.vlm,
        vlm_conf_gate=args.vlm_conf,
        vlm_model=args.vlm_model,
    )
    app.run()


if __name__ == "__main__":
    main()
