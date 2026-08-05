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
# Load environment variables from .env file
env_path = Path(".") / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Also try loading from the current environment if .env is missing
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
                 vlm_backend_name=None, vlm_conf_gate=0.6, vlm_model=None):
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
            if q.status == InspectionStatus.DEFECT:
                color = (0, 0, 255)  # Red for Defect
            elif q.status == InspectionStatus.UNCERTAIN:
                color = (0, 255, 255)  # Yellow for Uncertain
            elif q.status == InspectionStatus.SKIPPED:
                color = (128, 128, 128)  # Gray for Skipped
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{det.label} {det.confidence:.2f}"
            if q.status != InspectionStatus.SKIPPED:
                label += f" | {q.status.value.upper()}"
                if q.overall_quality_score is not None:
                    label += f" (Score: {q.overall_quality_score:.2f})"
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw structured metrics and defects
            if q.status != InspectionStatus.SKIPPED:
                y_offset = y2 + 20
                
                # Show defects if any
                if q.defects:
                    defects_text = f"Defects: {', '.join(q.defects)}"
                    cv2.putText(frame, defects_text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20
                
                # Show top 3 quality metrics
                sorted_metrics = sorted(q.quality_metrics.items(), key=lambda x: x[1])[:3]
                for metric, score in sorted_metrics:
                    m_text = f"{metric}: {score:.2f}"
                    # Color metric red if score is low
                    m_color = (0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
                    cv2.putText(frame, m_text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, m_color, 1)
                    y_offset += 15

                # Draw explanation
                if q.explanation:
                    cv2.putText(frame, q.explanation, (x1, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

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

                # Capture keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Handle commands (case-insensitive)
                if key in (ord('q'), ord('Q'), 27):
                    print("[INFO] Quit requested via keyboard.")
                    break
                elif key in (ord('s'), ord('S')):
                    self.save_snapshot(annotated, payload)
                elif key != 255:
                    print(f"[DEBUG] Key pressed: {key} ('{chr(key) if key < 128 else '?'}')")
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
    parser.add_argument("--vlm", choices=["gpt4o", "openai", "qwen", "qwen-api", "openrouter"], help="Enable VLM reasoning backend")
    parser.add_argument("--vlm-model", help="Specific model ID to use with the VLM backend")
    parser.add_argument("--vlm-conf", type=float, default=DEFAULT_VLM_CONF_GATE, help="VLM confidence gate")
    
    args = parser.parse_args()
    
    # Auto-detect VLM if not specified
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
            
    # Validation for requested backends
    if args.vlm in ["gpt4o", "openai"] and not openai_key:
        print(f"[ERROR] --vlm {args.vlm} requested but OPENAI_API_KEY is not set.")
        print("[TIP] Create a .env file with: OPENAI_API_KEY=your_key_here")
    elif args.vlm == "qwen-api" and not qwen_key:
        print(f"[ERROR] --vlm {args.vlm} requested but QWEN_API_KEY or DASHSCOPE_API_KEY is not set.")
        print("[TIP] Create a .env file with: QWEN_API_KEY=your_key_here")
    elif args.vlm == "openrouter" and not openrouter_key:
        print(f"[ERROR] --vlm {args.vlm} requested but OPENROUTER_API_KEY is not set.")
        print("[TIP] Create a .env file with: OPENROUTER_API_KEY=your_key_here")
    
    app = FoodInspectionApp(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        vlm_backend_name=args.vlm,
        vlm_conf_gate=args.vlm_conf,
        vlm_model=args.vlm_model
    )
    app.run()

if __name__ == "__main__":
    main()
