# Food Inspection — YOLO + VLM

## Quick Start (Presentation Mode)

To launch the live webcam inspection with both **YOLO detection** and **VLM quality reasoning**, run:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API Key (copy .env.example to .env and add your key)
cp .env.example .env

# Run with GPT-4o reasoning
python main.py --vlm gpt4o

# Run detection-only (no VLM costs/latency)
python main.py
```

**Controls:**
- `s`: Save an annotated snapshot and JSON report to `outputs/snapshots/`.
- `q` or `ESC`: Quit the application.

## Full System Architecture

The diagram below presents the complete end-to-end architecture of the platform, from image acquisition at the conveyor to alerting on the operator dashboard.

[![Industrial Food Inspection Platform Architecture](architecture-diagram.png)](https://github.com/Achraf-saadali/Food-Inspection/blob/main/diagram-export-31-07-2026-16_31_48.png)

The system is organized into eight stages:

1. **Image Acquisition** — a conveyor-mounted inspection camera captures frames and encodes them as a JPEG byte stream.
2. **Communication** — a client application receives the camera feed and transmits it to the backend through a `POST /detect` request.
3. **Backend Processing** — a FastAPI server receives the JPEG bytes, decodes them with OpenCV into a NumPy array, and hands the resulting tensor to the detector.
4. **Object Detection** — the YOLOv9c model (63 classes) processes the image through its backbone, neck, and detection head, producing bounding boxes, class identifiers, and confidence scores.
5. **Cropping** — each detected object is cropped from the source image and paired with a quality-inspection prompt.
6. **Vision Reasoning (VLM)** — a general-purpose or fine-tuned Vision-Language Model evaluates each crop for spoilage, damage, or defect evidence, producing a structured output containing item status, defect type, freshness score, explanation, and required follow-up action.
7. **Decision Layer** — detection metadata from YOLO and reasoning output from the VLM are merged into a unified inspection result.
8. **Visualization and Alerts** — results are displayed on a quality-monitoring dashboard and, when a defect exceeds a defined severity threshold, routed to a notification service that alerts quality-control staff.

This architecture reflects the target design of the system. **Stages 1 and 4
(image acquisition and object detection) were the first to be implemented.
Stages 2–3, 5–7 (backend API, cropping, VLM reasoning, and result merging)
have since moved from planned to implemented** — see Sections 8–9 and 18
below. Stage 8 (dashboard and alerting) remains planned.

---

## Overview

**Food Inspection** is a computer-vision pipeline for food ingredient inspection, combining **object detection with YOLO** and a **Vision-Language Model (VLM)** reasoning stage.

The first stage of the pipeline is **real-time detection and classification of food ingredients**, particularly fruits and vegetables. A custom-trained **YOLOv9c** model detects and labels 63 ingredient classes and produces structured JSON detection reports.

The second stage is a VLM reasoning layer that evaluates the visual condition of each detected ingredient — freshness defects, bruising, mold, or other quality anomalies — and returns a structured assessment. This stage is now integrated behind a backend-agnostic interface supporting both a local open-source VLM and a hosted API model (Section 8).

### Pipeline (Detection + Reasoning)

```text
Camera / Image / Video
        │
        ▼
   YOLOv9c Detection
        │
        ├── Detected objects
        ├── Classes
        ├── Confidence scores
        └── Bounding boxes
        │
        ▼
  Crop detected objects
        │
        ▼
     VLM Reasoning
        │
        ├── Quality status
        ├── Defect type
        ├── Explanation
        └── Required action
        │
        ▼
 Structured Inspection Result
```

The YOLO detection stage is functional and versioned. The VLM reasoning stage
is integrated and callable end-to-end; the benchmark comparing candidate
models (Section 9) is still in progress.

---

# 1. Project Objectives

The project is designed as a two-stage food-inspection system.

### Stage 1 — Object Detection

Detect and classify food ingredients in an image or video stream.

The YOLO model recognizes **63 fruit and vegetable classes** and returns:

* Object class
* Class ID
* Confidence score
* Bounding box coordinates
* Normalized bounding box coordinates

### Stage 2 — Visual Quality Reasoning

Detected objects are passed to a VLM together with a quality-inspection prompt, for example:

> Does this food item show any visible freshness defect, bruising, mold, discoloration, or other quality issue?

The VLM returns a structured assessment containing:

* Inspection status
* Defect type
* Explanation
* Required action

This stage is implemented (Section 8) and pending final model selection from the benchmark (Section 9).

---

# 2. Repository Structure

```text
Food-Inspection/
│
├── models/
│   └── best.pt
│
├── notebooks/
│
├── outputs/
│   ├── snapshots/
│   └── logs/
│
├── trainning_runs/
│   └── train4/
│
├── schemas.py
├── vlm_reasoning.py
├── inspection_pipeline.py
├── api.py
├── benchmark_vlm.py
├── live_inference.py
├── main.py
├── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

| File / Folder             | Description                                                        |
| -------------------------- | ------------------------------------------------------------------- |
| `main.py`                  | Unified entrypoint for live quality inspection (YOLO + VLM)        |
| `live_inference.py`        | Legacy inference script for detection only                          |
| `models/best.pt`           | Trained YOLOv9c weights for 63 classes                              |
| `schemas.py`                | Unified JSON contract shared by the detection and VLM stages        |
| `vlm_reasoning.py`          | Backend-agnostic VLM interface (Qwen2.5-VL, GPT-4o)                 |
| `inspection_pipeline.py`    | Orchestrator: image → YOLO → crop → VLM → merged result             |
| `api.py`                    | FastAPI service exposing `/detect` and `/inspect`                   |
| `benchmark_vlm.py`          | Latency / VRAM / agreement comparison across VLM backends           |
| `outputs/snapshots/`       | Annotated images and JSON reports saved during inference            |
| `outputs/logs/`             | Per-session JSONL detection logs                                    |
| `trainning_runs/train4/`   | Training artifacts, metrics, curves and confusion matrix            |
| `notebooks/`                | Experimentation and training notebooks                              |
| `ARCHITECTURE.md`           | Design decisions, data flow, and learnings from the VLM integration |
| `requirements.txt`          | Python dependencies                                                 |

---

# 3. YOLO Detection Module

## 3.1 Model

The current detection model is defined as follows:

* **Architecture:** YOLOv9c
* **Task:** Object detection
* **Classes:** 63
* **Dataset:** LVIS Fruits & Vegetables
* **Input size:** 640 × 640
* **Training:** 30 epochs
* **Batch size:** 8
* **Initial learning rate:** `0.001`
* **Inference weights:** `models/best.pt`

The trained model detects ingredients such as:

* Tomato
* Onion
* Garlic
* Apple
* Banana
* Potato
* Cucumber
* Bell pepper
* And other fruits and vegetables

---

# 4. YOLOv9c

Several YOLO generations were considered for the detection stage.

| Model       | Approx. mAP@0.5 | Speed         | Main strengths                              | Main limitations                     |
| ----------- | --------------: | ------------- | ------------------------------------------- | ------------------------------------ |
| YOLOv5      |         ~50–56% | Very fast     | Mature ecosystem, lightweight               | Older architecture                   |
| YOLOv8      |         ~53–57% | Fast          | Anchor-free, good precision/speed trade-off | Marginal gain on small datasets      |
| **YOLOv9c** |     **~55–58%** | Fast–moderate | PGI + GELAN, good fine-object detection     | Slightly more expensive training     |
| YOLOv10     |         ~54–58% | Very fast     | NMS-free inference                          | Younger ecosystem                    |
| YOLO11      |         ~55–59% | Fast          | Refined backbone, good overall balance      | Less long-term production experience |

### Model choice: YOLOv9c

YOLOv9c was selected on the basis of four factors:

1. Detection accuracy
2. Training cost
3. GPU memory requirements
4. Ability to detect medium and relatively small objects, with near-real-time inference

The **PGI (Programmable Gradient Information)** and **GELAN** architecture are particularly relevant when information must be preserved through deeper layers, which is beneficial for small or partially occluded objects.

The compact **`c`** variant was chosen over the larger `e` variant because it reduces training time and GPU memory requirements while remaining appropriate for a dataset of approximately 8,000 images and 63 classes.

---

# 5. Dataset

## LVIS Fruits & Vegetables Detection Dataset

The detection model was trained on a filtered subset of the **LVIS (Large Vocabulary Instance Segmentation)** dataset.

| Attribute              | Value                                             |
| ---------------------- | ------------------------------------------------- |
| Dataset                | LVIS Fruits & Vegetables Detection Dataset        |
| Parent dataset         | LVIS                                              |
| Parent dataset size    | ~160,000 images                                   |
| Parent dataset classes | 1,203                                             |
| Filtering              | Images containing at least one fruit or vegetable |
| Total images           | **8,221**                                         |
| Training images        | **6,721**                                         |
| Validation images      | **1,500**                                         |
| Additional test images | **180 manually annotated images**                 |
| Number of classes      | **63**                                            |
| Annotation format      | YOLO bounding boxes                               |

The original annotations were provided in COCO JSON format and converted to YOLO TXT format.

Each annotation follows the standard YOLO format:

```text
class_id x_center y_center width height
```

All bounding-box coordinates are normalized relative to the image dimensions.

---

# 6. Dataset Split

The dataset source provides a predefined split:

```text
Training:   6,721 images
Validation: 1,500 images
Total:      8,221 images
```

This corresponds approximately to:

```text
82% training
18% validation
```

The split is inherited from the dataset source rather than constructed as a random split for this project. This provides reproducibility between experiments, although it does not necessarily guarantee balanced representation across classes.

### Alternative splitting strategies

#### Random split

Each image is independently assigned to training or validation.

**Advantages:** simple, fast, easy to reproduce.

**Limitation:** rare classes can become over- or under-represented.

#### Stratified split

The class distribution is preserved between training and validation. This is preferable when working with a strongly imbalanced dataset.

#### Scenario/source-based split

Images can be separated according to their acquisition context when multiple environments exist, for example:

```text
Plate
Shelf
Conveyor
Other acquisition environment
```

This can reduce information leakage between training and validation.

---

# 7. Class Imbalance

The LVIS dataset follows a **long-tail distribution**: a relatively small number of classes account for a large proportion of the available instances, while other classes appear in only a limited number of images.

In this project, common vegetable classes have considerably more training examples than rarer ingredient classes.

Two duplicated classes are also inherited from the original LVIS data, differing only by capitalization:

```text
Tomato
Strawberry
```

These classes contain relatively few images and do not currently have a significant effect on training.

## Approaches to Class Imbalance

Several methods are applicable to this type of imbalance.

### Per-class evaluation

Rather than relying solely on global mAP, per-class mAP, precision, recall, and instance counts can be monitored individually. This allows classes whose performance is limited by insufficient training data to be identified.

### Targeted augmentation

Rare classes can be augmented through techniques such as:

* Mosaic
* Copy-paste
* Horizontal/vertical flips
* Brightness variations
* Other image transformations

Ultralytics already applies several augmentation techniques during training.

### Oversampling

Images containing rare classes can be sampled more frequently during training.

### Loss weighting

The contribution of underrepresented classes to the loss function can be increased through weighting.

### Class consolidation

Classes that are duplicated, or that provide limited value for the inspection use case, may be merged or removed.

> The class-frequency distribution has not yet been generated from the annotation files and remains to be added to the repository.

---

# 8. VLM Reasoning Stage

The second stage of the pipeline is based on a **Vision-Language Model**, and is now implemented behind a backend-agnostic interface (`vlm_reasoning.py`).

Rather than sending the complete image to the VLM, YOLO detections identify relevant objects and their crops are supplied to the VLM. This reduces unnecessary visual information and focuses the reasoning stage on the detected food item.

```text
Original Image
      │
      ▼
    YOLO
      │
      ▼
Detected bounding boxes
      │
      ▼
Object crops
      │
      ▼
     VLM
      │
      ▼
Quality assessment
```

The VLM receives:

1. The detected object crop
2. The detected ingredient class
3. The YOLO confidence score
4. A quality-inspection prompt

Prompt used in the current implementation:

```text
Inspect this food item for visible quality defects.

Determine whether the item shows:
- signs of mold
- bruising
- discoloration
- freshness problems
- other visible abnormalities

Return:
- status
- defect type
- explanation
- required action
```

### Implementation notes

* The VLM is only called on detections above a confidence gate (default
  `0.4`), to avoid spending reasoning calls on likely false positives.
* A response that fails to parse against the expected schema is not
  discarded or retried silently — it is mapped to `status: uncertain` with
  `required_action: flag_for_review`, so a malformed VLM output surfaces as
  a signal rather than a silent gap in coverage.
* Two backends are implemented and interchangeable at call time: a local
  open-source model and a hosted reference model (Section 9). Design
  rationale for this interface is in `ARCHITECTURE.md` §3.2.

---

# 9. VLM Benchmark

Several VLM candidates were considered for the reasoning stage.

| Model                  | Type            | Visual reasoning                                        | Speed         | Open source | Approx. GPU memory     |
| ----------------------- | --------------- | -------------------------------------------------------- | ------------- | ----------- | ----------------------- |
| LLaVA 1.5 / 1.6         | Open-source VLM | Good general description, weaker fine reasoning          | Moderate      | Yes         | ~8 GB for quantized 7B |
| **Qwen2.5-VL**          | Open-source VLM | High; strong spatial grounding and fine visual detail    | Moderate–fast | Yes         | ~8 GB for 3B            |
| InternVL 2.5 / 3        | Open-source VLM | High; strong perception and OCR                          | Moderate      | Yes         | ~16 GB+                 |
| Phi-3 / Phi-3.5 Vision  | Open-source VLM | Adequate for its size, less robust on ambiguous cases     | Fast          | Yes         | ~4–8 GB                 |
| GPT-4o Vision           | Proprietary API | Very high contextual reasoning                            | API-dependent | No          | Hosted                  |

## Model Selection Rationale

**Qwen2.5-VL** is implemented as the primary, high-volume backend (`Qwen25VLBackend`), based on a balance of:

* Visual reasoning quality
* GPU memory requirements
* Inference cost
* Deployment flexibility

**GPT-4o Vision** is implemented as the reference backend (`GPT4oBackend`), used for ambiguous cases and to evaluate the output quality of the open-source VLM — not as the default backend for every detection.

Both backends share one interface (`VLMBackend.analyze()`), so switching between them is a configuration change, not a code change. Final backend selection for production use is still pending the evaluation below.

### Evaluation status

The VLM candidates are being evaluated using real crops generated by the YOLOv9c detector, via `benchmark_vlm.py`. The comparison examines:

* Real inference latency
* GPU VRAM usage
* Agreement with human annotations
* Quality of defect classification
* Explanation quality
* Robustness on ambiguous cases

🔄 In progress — results will be added to this section once the benchmark run completes.

---

# 10. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirement.txt
```

---

# 11. Running the Detector

## Webcam

```bash
python live_inference.py
```

The default webcam index is:

```text
0
```

The application processes the webcam stream frame by frame and displays the detected objects. For frames containing detections, a structured JSON report is also generated.

## Image

```bash
python live_inference.py --source path/to/image.jpg
```

## Video

```bash
python live_inference.py --source path/to/video.mp4
```

## Custom model

```bash
python live_inference.py --model path/to/model.pt
```

## With VLM reasoning enabled

```bash
python live_inference.py --enable-vlm --vlm-backend qwen
```

## Via the API

```bash
uvicorn api:app --reload --port 8000

# detection only
curl -X POST -F file=@sample.jpg http://localhost:8000/detect

# detection + VLM reasoning
curl -X POST -F file=@sample.jpg "http://localhost:8000/inspect?vlm_backend=qwen&confidence_gate=0.4"
```

---

# 12. Inference Options

| Argument          | Description                                | Default          |
| ------------------ | -------------------------------------------- | ----------------- |
| `--model`           | Path to trained YOLO weights                | `models/best.pt` |
| `--source`          | Webcam index, image path, or video path     | `0`               |
| `--enable-vlm`      | Run the VLM reasoning stage on each detection | off               |
| `--vlm-backend`     | `qwen` or `gpt4o`                            | `qwen`            |

---

# 13. Live Controls

| Key   | Action                                            |
| ----- | -------------------------------------------------- |
| `s`   | Save the current annotated frame and JSON report  |
| `q`   | Stop the inference session                        |
| `ESC` | Stop the inference session                        |

Saved snapshots are stored in:

```text
outputs/snapshots/
```

---

# 14. JSON Output

The pipeline now produces a **unified** JSON representation covering both
detection and (optional) VLM reasoning, defined in `schemas.py`. Consumers
that only need the detection fields can still read the same object — the
`quality` field is simply `null` when the VLM stage was not run.

Example (with VLM reasoning enabled):

```json
{
  "frame_id": 42,
  "timestamp": "2026-07-24T15:03:11.482",
  "source": "0",
  "image_size": { "width": 640, "height": 480 },
  "num_detections": 2,
  "items": [
    {
      "detection": {
        "label": "tomato",
        "class_id": 59,
        "confidence": 0.87,
        "bbox_xyxy": [120.4, 88.1, 240.7, 210.3],
        "bbox_normalized": [0.188, 0.183, 0.376, 0.438]
      },
      "quality": {
        "status": "ok",
        "defect_type": "none",
        "freshness_score": 0.91,
        "explanation": "Skin is uniformly red with no visible bruising or mold.",
        "required_action": "none",
        "vlm_backend": "qwen2.5-vl",
        "latency_ms": 340.2
      }
    },
    {
      "detection": {
        "label": "onion",
        "class_id": 43,
        "confidence": 0.71,
        "bbox_xyxy": [300.2, 150.0, 400.9, 260.5],
        "bbox_normalized": [0.469, 0.313, 0.626, 0.543]
      },
      "quality": null
    }
  ]
}
```

### JSON fields

| Field                       | Description                                              |
| ---------------------------- | ---------------------------------------------------------- |
| `frame_id`                   | Sequential index of the processed frame                  |
| `timestamp`                  | ISO-8601 detection timestamp                              |
| `source`                     | Camera index, video path, or image path                  |
| `image_size`                 | Width and height of the processed frame                  |
| `num_detections`             | Number of detected objects                                |
| `items`                      | List of detection + quality-assessment pairs              |
| `items[].detection.label`    | Predicted ingredient class                                |
| `items[].detection.class_id` | Predicted class index, 0–62                                |
| `items[].detection.confidence` | Detection confidence between 0 and 1                    |
| `items[].detection.bbox_xyxy` | Bounding box in pixel coordinates                        |
| `items[].detection.bbox_normalized` | Bounding box normalized to `[0, 1]`                 |
| `items[].quality`            | VLM assessment, or `null` if the VLM stage did not run    |
| `items[].quality.status`     | `ok`, `defect`, or `uncertain`                             |
| `items[].quality.defect_type`| `none`, `mold`, `bruising`, `discoloration`, `freshness`, `other` |
| `items[].quality.freshness_score` | Freshness estimate, 0–1                              |
| `items[].quality.explanation`| One-sentence visual justification from the VLM            |
| `items[].quality.required_action` | `none`, `flag_for_review`, or `remove`               |
| `items[].quality.vlm_backend`| Which backend produced the assessment                     |
| `items[].quality.latency_ms` | VLM call latency for that crop                             |

A backwards-compatible `to_legacy_detection_json()` helper is available on
`InspectionResult` for any consumer still expecting the original
detection-only shape.

---

# 15. Logging

Each inference session generates a JSONL log:

```text
outputs/logs/session_<timestamp>.jsonl
```

Each line represents one inspection report (detection, or detection + VLM assessment when `--enable-vlm` is set).

The logs are intended for future integration with:

* A database
* A monitoring dashboard
* A downstream alerting service

---

# 16. Current Architecture (Detection + Reasoning)

```text
                 ┌─────────────────────┐
                 │ Camera / Image /     │
                 │ Video                │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     YOLOv9c         │
                 │ Object Detection    │
                 └──────────┬──────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
          Class / Score            Bounding Box
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
                    Object Crop(s)
                            │
                            ▼
                 ┌─────────────────────┐
                 │        VLM          │
                 │ Visual Reasoning    │
                 │ (Qwen2.5-VL /       │
                 │  GPT-4o)            │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Quality Inspection  │
                 │ Status / Defect /   │
                 │ Explanation / Action│
                 └─────────────────────┘
```

This diagram now corresponds to stages 4–7 of the full system architecture
presented at the beginning of this document (detection, cropping, VLM
reasoning, and merged result) — implemented via `inspection_pipeline.py` and
served over HTTP via `api.py`. Acquisition and alerting remain as described
there.

---



```text
Camera
   ↓
YOLO                     ✅
   ↓
Object Detection         ✅
   ↓
Object Crop              ✅
   ↓
VLM                      ✅
   ↓
Quality Reasoning        ✅
   ↓
Structured JSON          ✅
   ↓
Database / Dashboard / Alerts   🔄
```

---



---


