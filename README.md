# Food Inspection — YOLO + VLM

## Overview

**Food Inspection** is a computer-vision pipeline for food ingredient inspection, combining **object detection with YOLO** and a planned **Vision-Language Model (VLM)** reasoning stage.

The current implementation focuses on the first stage of the pipeline: **real-time detection and classification of food ingredients**, particularly fruits and vegetables. A custom-trained **YOLOv9c** model detects and labels 63 ingredient classes and produces structured JSON detection reports.

The long-term objective is to extend this detection stage with a VLM capable of reasoning about the visual condition of detected ingredients, including potential freshness defects, bruising, mold, or other quality anomalies.

### Pipeline

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
        └── Recommended action
        │
        ▼
 Structured Inspection Result
```

The YOLO detection stage is currently functional and versioned. The VLM reasoning stage is under integration.

---

# 1. Project Objectives

The project is being developed as a two-stage food-inspection system:

### Stage 1 — Object Detection

Detect and classify food ingredients in an image or video stream.

The current YOLO model recognizes **63 fruit and vegetable classes** and returns:

* Object class
* Class ID
* Confidence score
* Bounding box coordinates
* Normalized bounding box coordinates

### Stage 2 — Visual Quality Reasoning

The detected objects can subsequently be passed to a VLM together with a quality-inspection prompt.

For example:

> Does this food item show any visible freshness defect, bruising, mold, discoloration, or other quality issue?

The VLM is expected to return a structured assessment containing:

* Inspection status
* Defect type
* Explanation
* Recommended action

This second stage is currently under development.

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
├── live_inference.py
├── requirement.txt
└── README.md
```

| File / Folder            | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `live_inference.py`      | Main inference script for webcam, video, or image input  |
| `models/best.pt`         | Trained YOLOv9c weights for 63 classes                   |
| `outputs/snapshots/`     | Annotated images and JSON reports saved during inference |
| `outputs/logs/`          | Per-session JSONL detection logs                         |
| `trainning_runs/train4/` | Training artifacts, metrics, curves and confusion matrix |
| `notebooks/`             | Experimentation and training notebooks                   |
| `requirement.txt`        | Python dependencies                                      |

---

# 3. YOLO Detection Module

## 3.1 Model

The current detection model is:

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

# 4. Why YOLOv9c?

Several YOLO generations were considered for the detection stage.

| Model       | Approx. mAP@0.5 | Speed         | Main strengths                              | Main limitations                     |
| ----------- | --------------: | ------------- | ------------------------------------------- | ------------------------------------ |
| YOLOv5      |         ~50–56% | Very fast     | Mature ecosystem, lightweight               | Older architecture                   |
| YOLOv8      |         ~53–57% | Fast          | Anchor-free, good precision/speed trade-off | Marginal gain on small datasets      |
| **YOLOv9c** |     **~55–58%** | Fast–moderate | PGI + GELAN, good fine-object detection     | Slightly more expensive training     |
| YOLOv10     |         ~54–58% | Very fast     | NMS-free inference                          | Younger ecosystem                    |
| YOLO11      |         ~55–59% | Fast          | Refined backbone, good overall balance      | Less long-term production experience |

### Selected model: YOLOv9c

YOLOv9c was selected because it provides a good balance between:

1. Detection accuracy
2. Training cost
3. GPU memory requirements
4. Detection of medium and relatively small objects
5. Near-real-time inference capability

The **PGI (Programmable Gradient Information)** and **GELAN** architecture are particularly relevant when information needs to be preserved through deeper layers, which can help with small or partially occluded objects.

The compact **`c`** variant was preferred over the larger `e` variant because it reduces training time and GPU memory requirements while remaining appropriate for a dataset of approximately 8,000 images and 63 classes.

---

# 5. Dataset

## LVIS Fruits & Vegetables Detection Dataset

The detection model was trained using a filtered subset of the **LVIS (Large Vocabulary Instance Segmentation)** dataset.

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

The split is inherited from the dataset source rather than being a random split created specifically for this project.

This provides reproducibility between experiments, although it does not necessarily guarantee perfect class stratification.

### Possible splitting strategies

#### Random split

Each image is independently assigned to training or validation.

**Advantages:**

* Simple
* Fast
* Easy to reproduce

**Limitation:**

Rare classes can become over- or under-represented.

#### Stratified split

The class distribution is preserved between training and validation.

This is preferable when working with a strongly imbalanced dataset.

#### Scenario/source-based split

Images can be separated according to their acquisition context when multiple environments exist, for example:

```text
Plate
Shelf
Conveyor
Other acquisition environment
```

This can help reduce information leakage between training and validation.

---

# 7. Class Imbalance

The LVIS dataset naturally follows a **long-tail distribution**.

A relatively small number of classes can contain a large proportion of the available instances, while other classes may appear only in a limited number of images.

For this project, classes such as common vegetables may have considerably more training examples than rare ingredient classes.

There are also two duplicated classes inherited from the original LVIS data with capitalization variants:

```text
Tomato
Strawberry
```

These classes contain relatively few images and currently do not significantly affect the training process.

## Recommended strategies

Several approaches can be used to address class imbalance:

### Per-class evaluation

Instead of relying only on global mAP, monitor:

* Per-class mAP
* Precision
* Recall
* Number of instances per class

This makes it possible to identify classes whose performance is limited by insufficient training data.

### Targeted augmentation

Rare classes can benefit from additional augmentation techniques such as:

* Mosaic
* Copy-paste
* Horizontal/vertical flips
* Brightness variations
* Other image transformations

Ultralytics already applies several augmentation techniques during training.

### Oversampling

Images containing rare classes can be sampled more frequently during training.

### Loss weighting

The contribution of underrepresented classes can potentially be increased through loss weighting.

### Class consolidation

Classes that are duplicated or provide little value for the final inspection use case may eventually be merged or removed.

> The exact class-frequency distribution still needs to be generated directly from the annotation files and added to the repository.

---

# 8. VLM Reasoning Stage

The second stage of the pipeline is based on a **Vision-Language Model**.

Instead of sending the complete image to the VLM, the intended approach is to use the YOLO detections to identify relevant objects and provide their crops to the VLM.

Conceptually:

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

This reduces unnecessary visual information and focuses the reasoning stage on the detected food item.

The VLM can receive:

1. The detected object crop
2. The detected ingredient class
3. The YOLO confidence score
4. A quality-inspection prompt

Example prompt:

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
- recommended action
```

---

# 9. VLM Benchmark

Several VLM candidates were considered for the reasoning stage.

| Model                  | Type            | Visual reasoning                                       | Speed         | Open source | Approx. GPU memory     |
| ---------------------- | --------------- | ------------------------------------------------------ | ------------- | ----------- | ---------------------- |
| LLaVA 1.5 / 1.6        | Open-source VLM | Good general description, weaker fine reasoning        | Moderate      | Yes         | ~8 GB for quantized 7B |
| **Qwen2.5-VL**         | Open-source VLM | High, strong spatial grounding and fine visual details | Moderate–fast | Yes         | ~8 GB for 3B           |
| InternVL 2.5 / 3       | Open-source VLM | High, strong perception and OCR                        | Moderate      | Yes         | ~16 GB+                |
| Phi-3 / Phi-3.5 Vision | Open-source VLM | Good for its size, less robust on ambiguous cases      | Fast          | Yes         | ~4–8 GB                |
| GPT-4o Vision          | Proprietary API | Very high contextual reasoning                         | API-dependent | No          | Hosted                 |

## Current recommendation

**Qwen2.5-VL 7B** is currently the primary open-source candidate because it offers a good compromise between:

* Visual reasoning quality
* GPU memory requirements
* Inference cost
* Deployment flexibility

**GPT-4o Vision** can be used as a high-accuracy reference for ambiguous cases and for validating the quality of the open-source VLM results.

The final VLM selection remains pending integration experiments.

### Planned evaluation

The VLM candidates will be evaluated using real crops generated by the YOLOv9c detector.

The comparison should include:

* Real inference latency
* GPU VRAM usage
* Agreement with human annotations
* Quality of defect classification
* Explanation quality
* Robustness on ambiguous cases

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

The application processes the webcam stream frame by frame and displays the detected objects.

For frames containing detections, a structured JSON report is also generated.

---

## Image

Specify an image as the source:

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

---

# 12. Inference Options

| Argument   | Description                             | Default          |
| ---------- | --------------------------------------- | ---------------- |
| `--model`  | Path to trained YOLO weights            | `models/best.pt` |
| `--source` | Webcam index, image path, or video path | `0`              |

---

# 13. Live Controls

| Key   | Action                                           |
| ----- | ------------------------------------------------ |
| `s`   | Save the current annotated frame and JSON report |
| `q`   | Stop the inference session                       |
| `ESC` | Stop the inference session                       |

Saved snapshots are stored in:

```text
outputs/snapshots/
```

---

# 14. JSON Detection Output

The detection stage produces a structured JSON representation.

Example:

```json
{
  "frame_id": 42,
  "timestamp": "2026-07-24T15:03:11.482",
  "source": "0",
  "image_size": {
    "width": 640,
    "height": 480
  },
  "num_detections": 2,
  "detections": [
    {
      "label": "tomato",
      "class_id": 59,
      "confidence": 0.87,
      "bbox_xyxy": [
        120.4,
        88.1,
        240.7,
        210.3
      ],
      "bbox_normalized": [
        0.188,
        0.183,
        0.376,
        0.438
      ]
    },
    {
      "label": "onion",
      "class_id": 43,
      "confidence": 0.71,
      "bbox_xyxy": [
        300.2,
        150.0,
        400.9,
        260.5
      ],
      "bbox_normalized": [
        0.469,
        0.313,
        0.626,
        0.543
      ]
    }
  ]
}
```

### JSON fields

| Field             | Description                             |
| ----------------- | --------------------------------------- |
| `frame_id`        | Sequential index of the processed frame |
| `timestamp`       | ISO-8601 detection timestamp            |
| `source`          | Camera index, video path, or image path |
| `image_size`      | Width and height of the processed frame |
| `num_detections`  | Number of detected objects              |
| `detections`      | List of individual detections           |
| `label`           | Predicted ingredient class              |
| `class_id`        | Predicted class index, 0–62             |
| `confidence`      | Detection confidence between 0 and 1    |
| `bbox_xyxy`       | Bounding box in pixel coordinates       |
| `bbox_normalized` | Bounding box normalized to `[0, 1]`     |

---

# 15. Logging

Each inference session generates a JSONL log:

```text
outputs/logs/session_<timestamp>.jsonl
```

Each line represents one detection report.

The logs are intended for future integration with:

* A database
* A monitoring dashboard
* A FastAPI backend
* A downstream VLM service
* Long-running inspection sessions

---

# 16. Current Architecture

The current implementation can be represented as:

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
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Quality Inspection  │
                 │ Status / Defect /   │
                 │ Explanation / Action│
                 └─────────────────────┘
```

---

# 17. Integration Notes

The existing `build_detection_json()` function provides the basic payload structure required by downstream components.

The detection output can be extended to support the complete inspection pipeline.

Potential integration points include:

### FastAPI

Expose the complete inspection pipeline through an endpoint such as:

```text
POST /inspect
```

The endpoint would replace the current local `cv2.imshow`-based interaction with a request/response interface.

### VLM Triggering

The VLM stage can be triggered according to detection results, for example:

* Low YOLO confidence
* Specific detected ingredients
* Specific combinations of ingredients
* Detection events requiring quality inspection

### Database

Instead of keeping all results as local JSONL files, inspection results can eventually be persisted in a database.

### Dashboard

The structured detection and VLM results can be consumed by a real-time monitoring dashboard.

---

# 18. Current Project Status

| Component                       | Status         |
| ------------------------------- | -------------- |
| Dataset preparation             | ✅ Completed    |
| YOLO dataset conversion         | ✅ Completed    |
| YOLOv9c training                | ✅ Completed    |
| YOLO inference                  | ✅ Functional   |
| Webcam inference                | ✅ Functional   |
| JSON detection output           | ✅ Functional   |
| Detection logging               | ✅ Functional   |
| VLM benchmark                   | 🔄 In progress |
| Qwen2.5-VL integration          | 🔄 In progress |
| GPT-4o Vision comparison        | 🔄 Planned     |
| Per-class distribution analysis | 🔄 Planned     |
| Unified YOLO + VLM JSON schema  | 🔄 Planned     |
| FastAPI `/inspect` endpoint     | 🔄 Planned     |

---

# 19. Next Steps

The next development steps are:

1. **Finalize the VLM benchmark**

   * Compare Qwen2.5-VL and GPT-4o Vision on real YOLO-generated crops.
   * Measure latency, GPU memory, and agreement with human annotations.

2. **Analyze class distribution**

   * Generate the training/validation class-frequency histogram.
   * Identify rare and underrepresented classes.

3. **Improve class imbalance handling**

   * Evaluate targeted augmentation.
   * Consider oversampling or loss weighting for rare classes.
   * Review duplicated/low-frequency classes.

4. **Unify YOLO and VLM outputs**

   * Extend `build_detection_json()`.
   * Create a common JSON contract containing both detection and reasoning results.

5. **Expose the pipeline through FastAPI**

   * Implement `/inspect`.
   * Replace the current local display loop with an API-based inspection pipeline.

6. **Build the complete inspection pipeline**

```text
Camera
   ↓
YOLO
   ↓
Object Detection
   ↓
Object Crop
   ↓
VLM
   ↓
Quality Reasoning
   ↓
Structured JSON
   ↓
Database / Dashboard / Alerts
```

---

# 20. Future Direction

The final objective is to evolve the current food-detection module into a complete **AI-assisted food inspection system**.

The system will combine:

* Real-time object detection
* Visual quality assessment
* Multimodal reasoning
* Structured inspection results
* API-based deployment
* Persistent inspection logs
* Potential real-time monitoring and alerting

The current YOLOv9c implementation therefore represents the **first operational stage** of the larger inspection architecture.

---

## Author

**Achraf Saadali**

Food Inspection — YOLO + VLM

Repository: [Food-Inspection](https://github.com/Achraf-saadali/Food-Inspection)
