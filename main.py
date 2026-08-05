import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import os
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

from inspection_pipeline import run_inspection
from vlm_reasoning import get_backend
from schemas import InspectionStatus

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_PATH = "models/best.pt"
DEFAULT_CONF_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.5
load_dotenv()

DEFAULT_VLM_CONF_GATE = 0.6  # Only run VLM on high-confidence detections

OUTPUT_DIR = Path("outputs")
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
LOG_DIR = OUTPUT_DIR / "logs"

WINDOW_NAME = "Food Inspection - Live Quality Analysis"

# --------------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------------- #

class FoodInspectionApp:
    def __init__(self, model_path, source=0, conf=0.4, iou=0.5, 
                 vlm_backend_name=None, vlm_conf_gate=0.6):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")

        print(f"[INFO] Loading YOLO model from {self.model_path} ...")
        self.model = YOLO(str(self.model_path))
        
        self.vlm_backend = None
        if vlm_backend_name:
            print(f"[INFO] Initializing VLM backend: {vlm_backend_name} ...")
            try:
                self.vlm_backend = get_backend(vlm_backend_name)
            except Exception as e:
                print(f"[ERROR] Failed to load VLM backend: {e}")
                print("[WARN] Proceeding with detection only.")

        self.source = source
        self.conf = conf
        self.iou = iou
        self.vlm_conf_gate = vlm_conf_gate

        self.cap = None
        self.frame_id = 0

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = LOG_DIR / f"session_{ts}.jsonl"
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        print(f"[INFO] Logging results to {self.log_path}")

    def open_capture(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
        
        # Set resolution for better quality if possible
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def annotate_frame(self, frame, result):
        """Draw detections and quality assessments on the frame."""
        for item in result.items:
            det = item.detection
            q = item.quality
            
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            
            # Choose color based on quality
            color = (0, 255, 0)  # Green for OK
            if q:
                if q.status == InspectionStatus.DEFECT:
                    color = (0, 0, 255)  # Red for Defect
                elif q.status == InspectionStatus.UNCERTAIN:
                    color = (0, 255, 255)  # Yellow for Uncertain
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{det.label} {det.confidence:.2f}"
            if q:
                label += f" | {q.status.value.upper()}"
                if q.defects:
                    label += f": {', '.join(q.defects[:2])}"
                elif q.overall_quality_score is not None:
                    label += f" (Q: {q.overall_quality_score:.2f})"
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw explanation if defect
            if q and q.status == InspectionStatus.DEFECT and q.explanation:
                cv2.putText(frame, q.explanation, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def run(self):
        self.open_capture()
        print("[INFO] Live inspection started.")
        print("       Press 's' to save current frame + JSON")
        print("       Press 'q' or ESC to quit")

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                self.frame_id += 1
                
                # Run the full pipeline
                result = run_inspection(
                    image=frame,
                    yolo_model=self.model,
                    frame_id=self.frame_id,
                    source=str(self.source),
                    vlm_backend=self.vlm_backend,
                    vlm_confidence_gate=self.vlm_conf_gate
                )

                # Annotate and show
                annotated = self.annotate_frame(frame.copy(), result)
                cv2.imshow(WINDOW_NAME, annotated)

                # Log results
                payload = result.model_dump(mode='json')
                self.log_file.write(json.dumps(payload) + "\n")
                self.log_file.flush()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    self.save_snapshot(annotated, payload)
        finally:
            self.cleanup()

    def save_snapshot(self, annotated_frame, payload):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = SNAPSHOT_DIR / f"frame_{ts}.jpg"
        json_path = SNAPSHOT_DIR / f"frame_{ts}.json"
        cv2.imwrite(str(img_path), annotated_frame)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {img_path.name} + {json_path.name}")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.log_file:
            self.log_file.close()
        print("[INFO] Resources released.")

def main():
    parser = argparse.ArgumentParser(description="Food Quality Inspection System")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to YOLO weights")
    parser.add_argument("--source", default="0", help="Webcam index or video file")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="YOLO confidence")
    parser.add_argument("--vlm", choices=["gpt4o", "qwen"], help="Enable VLM reasoning backend")
    parser.add_argument("--vlm-conf", type=float, default=DEFAULT_VLM_CONF_GATE, help="VLM confidence gate")
    
    args = parser.parse_args()
    
    app = FoodInspectionApp(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        vlm_backend_name=args.vlm,
        vlm_conf_gate=args.vlm_conf
    )
    app.run()

if __name__ == "__main__":
    main()
