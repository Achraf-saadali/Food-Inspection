# Food Detection Module — Live Inference (YOLO)

## Overview

This module performs real-time detection and labeling of food ingredients
(fruits and vegetables) using a custom-trained YOLOv9c model. It is the
first stage of a food-inspection pipeline: instead of a defect/anomaly
detection approach, ingredients are directly detected, classified, and
reported in a structured JSON format.

The model was fine-tuned on the LVIS Fruits & Vegetables dataset and
recognizes 63 ingredient classes (e.g. tomato, onion, garlic, apple,
banana, potato, cucumber, bell pepper, etc.).

## Contents

| File / Folder            | Description                                              |
|---------------------------|-----------------------------------------------------------|
| `live_inference.py`       | Main script: runs YOLO inference on webcam, video, or image |
| `models/best.pt`          | Trained YOLOv9c weights (63 classes)                      |
| `requirements.txt`        | Python dependencies                                       |
| `outputs/snapshots/`      | Saved annotated frames + their JSON, generated at runtime  |
| `outputs/logs/`           | Per-session JSONL detection logs, generated at runtime     |
| `training_runs/train4/`  | Training artifacts (metrics, curves, confusion matrix)     |

## Model Summary

- **Architecture:** YOLOv9c
- **Task:** Object detection (bounding box + class)
- **Classes:** 63 (fruits & vegetables)
- **Training data:** LVIS Fruits & Vegetables dataset
- **Training config:** 30 epochs, image size 640, batch size 8, lr0 = 0.001
- **Weights used at inference:** `runs/detect/train4/weights/best.pt`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Trained weights are copied into the module before first use:

```bash
mkdir -p models
cp runs/detect/train4/weights/best.pt models/best.pt
```

## Usage

### Webcam (default)

```bash
python live_inference.py
```

Opens webcam index `0`, runs detection frame by frame, displays an
annotated live window, and prints a JSON detection report to the terminal
for every frame containing at least one detection.

### Specific camera index

```bash
python live_inference.py --source 1
```

### Video file

```bash
python live_inference.py --source path/to/video.mp4
```

### Single image

```bash
python live_inference.py --source path/to/photo.jpg
```

### Additional options

| Flag                    | Description                                         | Default          |
|--------------------------|------------------------------------------------------|-------------------|
| `--model`                | Path to trained weights                              | `models/best.pt`  |
| `--source`               | Webcam index, video path, or image path              | `0`               |
| `--conf`                 | Confidence threshold                                 | `0.4`             |
| `--iou`                  | IoU threshold for non-max suppression                | `0.5`             |
| `--print-empty-frames`   | Print JSON even when no detections are found         | off               |
| `--no-log-file`          | Disable JSONL logging to disk                        | logging enabled   |

## Controls (live/video window)

| Key       | Action                                             |
|-----------|-----------------------------------------------------|
| `s`       | Save current annotated frame (`.jpg`) + its detection report (`.json`) to `outputs/snapshots/` |
| `q` / `ESC` | Stop the session and release the camera            |

## JSON Output Format

Every detection report — printed to the terminal, appended to
`outputs/logs/session_<timestamp>.jsonl`, and written alongside any saved
snapshot — follows this schema:

```json
{
  "frame_id": 42,
  "timestamp": "2026-07-24T15:03:11.482",
  "source": "0",
  "image_size": { "width": 640, "height": 480 },
  "num_detections": 2,
  "detections": [
    {
      "label": "tomato",
      "class_id": 59,
      "confidence": 0.87,
      "bbox_xyxy": [120.4, 88.1, 240.7, 210.3],
      "bbox_normalized": [0.188, 0.183, 0.376, 0.438]
    },
    {
      "label": "onion",
      "class_id": 43,
      "confidence": 0.71,
      "bbox_xyxy": [300.2, 150.0, 400.9, 260.5],
      "bbox_normalized": [0.469, 0.313, 0.626, 0.543]
    }
  ]
}
```

| Field              | Description                                                   |
|---------------------|-----------------------------------------------------------------|
| `frame_id`          | Sequential index of the processed frame within the session      |
| `timestamp`         | ISO-8601 timestamp of the detection                              |
| `source`            | Input source used (camera index, video path, or image path)     |
| `image_size`        | Width/height of the processed frame in pixels                   |
| `num_detections`    | Number of objects detected in the frame                         |
| `detections`        | List of individual detections                                   |
| `label`             | Predicted ingredient class name                                 |
| `class_id`          | Predicted class index (0–62)                                    |
| `confidence`        | Model confidence score (0–1)                                    |
| `bbox_xyxy`         | Bounding box in pixel coordinates: `[x1, y1, x2, y2]`            |
| `bbox_normalized`   | Bounding box normalized to `[0, 1]` relative to image dimensions |

## Logging Behavior

Each run creates one JSONL log file at `outputs/logs/session_<timestamp>.jsonl`,
containing one JSON object per line for every frame with detections
(or every frame, if `--print-empty-frames` is set). This log is intended
for later ingestion by downstream components (dashboard, database, or
further pipeline stages).

## Integration Notes

The JSON output produced by `build_detection_json()` in `live_inference.py`
is the payload shape intended for downstream integration, for example:

- Exposing detection results through a FastAPI endpoint (e.g. `/detect`),
  replacing the `cv2.imshow` display loop with a request/response handler.
- Triggering a secondary decision stage (rule-based or VLM-based) when
  `num_detections == 0`, when confidence is low, or when specific class
  combinations are detected.
- Persisting detection logs to a database instead of local JSONL files
  for long-running or multi-session deployments.