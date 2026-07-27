"""
Food Inspection — Live Ingredient Detection (YOLO)
====================================================
Real-time detection of fruits & vegetables using a custom-trained YOLOv9
model (63 ingredient classes, trained on LVIS Fruits & Vegetables).

Streams a live annotated feed in a cv2 window, prints a structured JSON
detection report to the terminal for every frame that has detections,
and lets you save an annotated snapshot + its JSON on demand.

Works with:
  - a webcam            (default)
  - a video file         (--source path/to/video.mp4)
  - a single image       (--source path/to/photo.jpg)

Usage
-----
  python live_inference.py
  python live_inference.py --model models/best.pt --camera 0 --conf 0.4
  python live_inference.py --source samples/fridge.mp4
  python live_inference.py --source samples/plate.jpg

Controls (live/video window)
-----------------------------
  s   -> save current frame (annotated .jpg + .json) to outputs/snapshots
  q   -> quit
  ESC -> quit


"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_PATH = "models/best.pt"
DEFAULT_CONF_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.5

OUTPUT_DIR = Path("outputs")
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
LOG_DIR = OUTPUT_DIR / "logs"

WINDOW_NAME = "Food Inspection - Live Ingredient Detection"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------------- #

class LiveIngredientDetector:
    def __init__(self, model_path, source=0, conf=0.4, iou=0.5,
                 print_empty_frames=False, log_to_file=True):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n"
                f"Copy your trained weights there, e.g.:\n"
                f"  cp runs/detect/train4/weights/best.pt models/best.pt"
            )

        print(f"[INFO] Loading model from {self.model_path} ...")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        print(f"[INFO] Loaded {len(self.class_names)} ingredient classes.")

        self.source = source
        self.conf = conf
        self.iou = iou
        self.print_empty_frames = print_empty_frames
        self.log_to_file = log_to_file

        self.cap = None
        self.frame_id = 0
        self.is_single_image = False

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.log_file = None
        if self.log_to_file:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = LOG_DIR / f"session_{ts}.jsonl"
            self.log_file = open(self.log_path, "a", encoding="utf-8")
            print(f"[INFO] Logging detections to {self.log_path}")

    # ----------------------------------------------------------------- #
    # Source handling
    # ----------------------------------------------------------------- #

    def _resolve_source(self):
        """Decide whether source is a webcam index, a video file, or a still image."""
        src = self.source
        if isinstance(src, str) and Path(src).suffix.lower() in IMAGE_EXTENSIONS:
            self.is_single_image = True
            return src

        # Webcam index passed as string "0", "1", ...
        if isinstance(src, str) and src.isdigit():
            return int(src)

        return src  # int camera index or video file path

    def open_capture(self):
        resolved = self._resolve_source()
        if self.is_single_image:
            return  # handled separately in run_single_image()

        self.cap = cv2.VideoCapture(resolved)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {resolved}")

    # ----------------------------------------------------------------- #
    # Detection -> JSON
    # ----------------------------------------------------------------- #

    def build_detection_json(self, result, frame_shape):
        h, w = frame_shape[:2]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label": self.class_names[cls_id],
                "class_id": cls_id,
                "confidence": round(score, 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_normalized": [
                    round(x1 / w, 4), round(y1 / h, 4),
                    round(x2 / w, 4), round(y2 / h, 4),
                ],
            })

        return {
            "frame_id": self.frame_id,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "source": str(self.source),
            "image_size": {"width": w, "height": h},
            "num_detections": len(detections),
            "detections": detections,
        }

    def print_json(self, payload):
        print(json.dumps(payload, ensure_ascii=False))

    def log_json(self, payload):
        if self.log_file:
            self.log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.log_file.flush()

    def save_snapshot(self, annotated_frame, payload):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = SNAPSHOT_DIR / f"frame_{ts}.jpg"
        json_path = SNAPSHOT_DIR / f"frame_{ts}.json"
        cv2.imwrite(str(img_path), annotated_frame)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {img_path.name} + {json_path.name}")

    # ----------------------------------------------------------------- #
    # Run loops
    # ----------------------------------------------------------------- #

    def run(self):
        self.open_capture()
        if self.is_single_image:
            self.run_single_image()
        else:
            self.run_stream()

    def run_single_image(self):
        img_path = self._resolve_source()
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        self.frame_id += 1
        result = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        annotated = result.plot()
        payload = self.build_detection_json(result, frame.shape)

        self.print_json(payload)
        self.log_json(payload)

        print("[INFO] Press 's' to save, any other key to quit.")
        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            self.save_snapshot(annotated, payload)

        self.cleanup()

    def run_stream(self):
        print("[INFO] Live detection started.")
        print("       Press 's' to save current frame + JSON")
        print("       Press 'q' or ESC to quit")

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[WARN] Failed to grab frame — stream ended or camera disconnected.")
                    break

                self.frame_id += 1

                result = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
                annotated = result.plot()
                payload = self.build_detection_json(result, frame.shape)

                if payload["num_detections"] > 0 or self.print_empty_frames:
                    self.print_json(payload)
                    self.log_json(payload)

                cv2.imshow(WINDOW_NAME, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    print("[INFO] Quit requested.")
                    break
                elif key == ord('s'):
                    self.save_snapshot(annotated, payload)
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.log_file:
            self.log_file.close()
        print("[INFO] Session ended, resources released.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Live ingredient detection with YOLO")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help="Path to trained weights (best.pt)")
    parser.add_argument("--source", default="0",
                         help="Webcam index (default 0), or path to a video/image file")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                         help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD,
                         help="IoU threshold for NMS")
    parser.add_argument("--print-empty-frames", action="store_true",
                         help="Print JSON even for frames with zero detections")
    parser.add_argument("--no-log-file", action="store_true",
                         help="Disable JSONL logging to disk")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = LiveIngredientDetector(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        print_empty_frames=args.print_empty_frames,
        log_to_file=not args.no_log_file,
    )
    detector.run()


if __name__ == "__main__":
    main()
