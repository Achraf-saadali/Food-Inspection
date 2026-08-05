# Industrial Food Inspection Platform — YOLOv9 & Vision-Language Reasoning

This repository implements a two-stage automated inspection pipeline designed for high-precision food quality control. The system integrates **real-time object detection** using a fine-tuned **YOLOv9c** model with **structured visual reasoning** via **Vision-Language Models (VLM)**.

## 1. System Overview

The platform is engineered to automate the visual inspection of fruits and vegetables on a production line. The architecture follows a decoupled, two-stage approach:

1.  **Stage 1: Ingredient Localization & Classification** — A custom-trained YOLOv9c model detects food items in the video stream, providing bounding boxes and class labels for 63 distinct ingredient categories.
2.  **Stage 2: Ingredient-Aware Quality Assessment** — Detected objects are cropped and passed to a VLM (GPT-4o or Qwen2.5-VL). The VLM performs a structured assessment based on a **Quality Profile** specific to the detected ingredient (e.g., evaluating "ripeness" for tomatoes vs. "browning" for bananas).

---

## 2. Machine Learning Workflow

### 2.1 Data Sourcing and Processing
The detection model was trained on a specialized subset of the **LVIS (Large Vocabulary Instance Segmentation)** dataset, specifically filtered for fruits and vegetables.

*   **Source Data**: LVIS Fruits & Vegetables Dataset (~8,200 images).
*   **Preprocessing**: Original COCO-style segmentation masks were converted into YOLO-compatible bounding box coordinates.
*   **Data Augmentation**: To improve model robustness against occlusion and varying lighting conditions, the following augmentations were applied during training:
    *   **Mosaic Augmentation (1.0)**: Combines four training images into one, forcing the model to detect objects at smaller scales.
    *   **RandAugment**: A stochastic strategy for applying a diverse set of geometric and color transformations.
    *   **Horizontal Flipping (0.5)**: Increases spatial invariance.
    *   **HSV Color Jittering**: Enhances robustness to lighting variations.

### 2.2 Model Architecture: YOLOv9c
We selected the **YOLOv9c (Compact)** architecture due to its innovative approach to information preservation during deep feature extraction.

*   **Programmable Gradient Information (PGI)**: Addresses the problem of information bottleneck in deep networks by providing auxiliary paths for gradient flow.
*   **Generalized ELAN (GELAN)**: Optimizes the computational block design for better feature aggregation and efficiency.
*   **Efficiency**: The `c` variant provides an optimal trade-off between inference latency and detection precision, suitable for near-real-time deployment.

### 2.3 Fine-Tuning & Training Details
The model was fine-tuned using a transfer learning approach, starting from COCO-pretrained weights.

| Parameter | Value |
| :--- | :--- |
| **Base Model** | YOLOv9c.pt |
| **Input Resolution** | 640 x 640 pixels |
| **Epochs** | 30 |
| **Batch Size** | 8 |
| **Initial Learning Rate** | 0.001 |
| **Optimizer** | Auto-selected (SGD/AdamW) |
| **Hardware** | NVIDIA Tesla T4 (via Kaggle) |

### 2.4 Performance Metrics
The following metrics were achieved at the conclusion of the 30-epoch fine-tuning run:

| Metric | Value |
| :--- | :--- |
| **Precision (B)** | 34.3% |
| **Recall (B)** | 25.3% |
| **mAP@50** | 23.8% |
| **mAP@50-95** | 17.4% |

> **Note**: These metrics reflect the high complexity of the 63-class food detection task. The model demonstrates strong localization capabilities, while classification precision continues to improve with further training iterations.

---

## 3. Structured Quality Assessment

Unlike generic inspection systems, this platform utilizes a **Class-Specific Quality Framework**. When an object is detected, the system retrieves a relevant **Quality Profile** to guide the VLM's reasoning.

### 3.1 Quality Schema
The VLM returns a machine-readable JSON object following this authoritative schema:

```json
{
  "status": "ok | defect | uncertain",
  "overall_quality_score": 0.0 - 1.0,
  "quality_metrics": {
    "ripeness": 0.95,
    "bruising": 0.05,
    "mold": 0.0
  },
  "defects": ["minor_bruising"],
  "explanation": "One small bruise visible on the lower right quadrant.",
  "required_action": "none | flag_for_review | remove"
}
```

### 3.2 Ingredient-Aware Reasoning
The visual characteristics evaluated are dynamically selected based on the detected class:
*   **Banana**: Evaluates browning, black spots, and peel integrity.
*   **Tomato**: Evaluates color uniformity, cracking, and skin firmness.
*   **Potato**: Evaluates sprouting, greening, and rot evidence.

For detailed configuration, see [QUALITY_ASSESSMENT.md](QUALITY_ASSESSMENT.md).

---

## 4. Getting Started

### 4.1 Prerequisites
*   Python 3.9+
*   OpenCV, PyTorch, Ultralytics
*   OpenAI API Key (for GPT-4o reasoning)

### 4.2 Installation
```bash
# Clone the repository
git clone https://github.com/Achraf-saadali/Food-Inspection.git
cd Food-Inspection

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY
```

### 4.3 Running the Inspection

#### Option A: Live Webcam Interface
Launches a real-time window showing detections and quality reasoning.
```bash
python main.py --vlm gpt4o --conf 0.4
```
*   `s`: Save a snapshot (Image + JSON Report).
*   `q`: Quit.

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


