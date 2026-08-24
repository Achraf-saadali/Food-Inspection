# Explainable Food Inspection Using Object Detection and Vision-Language Reasoning

**Internship thesis draft**  
**Author:** [Student name]  
**Host organization:** [Organization]  
**Academic supervisor:** [Name]  
**Industry supervisor:** [Name]  
**Date:** [Date]

> **Evidence note.** This manuscript is grounded in the repository implementation, the supplied YOLO training notebook, and the supplied continuation log. Values that are not present in those materials are marked as items to measure rather than invented results. The draft is intentionally honest about the difference between an implemented capability and a completed scientific evaluation.

## Abstract

Food-quality inspection requires more than detecting that an object is present. A useful inspection system must identify the food item, assess visible quality characteristics, communicate the result in understandable language, and preserve evidence for later review. This project develops an explainable prototype that combines a YOLO11m object detector with a vision-language model (VLM) quality-assessment stage. The detector produces object classes, confidences, and bounding boxes. Each detected item is cropped and passed to a provider-independent VLM adapter that evaluates class-specific quality dimensions such as ripeness, bruising, mold, discoloration, freshness, or leakage. The result is validated against a typed schema and converted into farmer-readable commentary and report metrics.

The project includes a resumable Google Colab training workflow for a 63-class fruits-and-vegetables dataset. The supplied training configuration uses a pretrained `yolo11m.pt` initialization, 640-pixel images, an 80-epoch target, automatic batch sizing, AdamW selected explicitly to reproduce the original automatic optimizer choice, and Drive-backed checkpointing. The supplied continuation log reaches epoch 62. In the observed window, validation mAP50 varies between 0.227 and 0.244 and mAP50–95 varies between 0.163 and 0.177; the best values in that window are approximately 0.244 and 0.177 at epoch 60. These values are treated as intermediate evidence rather than final performance because the complete run, independent test evaluation, and per-class results were not supplied.

A minimal backend restructuring makes the web application and a new live-camera runner use the same detection-to-inspection function. The restructuring adds a farmer-report builder, provider-selectable Gemma and NVIDIA adapters, a VLM benchmark harness, and a report endpoint without replacing the existing frontend or schema. The thesis concludes that the system demonstrates a practical path from visual detection to explainable inspection, while identifying dataset quality, independent evaluation, VLM agreement, and measured camera throughput as the next scientific priorities.

**Keywords:** food inspection, object detection, YOLO11m, transfer learning, vision-language model, explainable AI, real-time inference, quality assessment

## 1. Introduction

### 1.1 Context

Food inspection is commonly performed by people who combine visual recognition with domain experience. An inspector does not merely identify an apple, tomato, or pepper. The inspector also considers ripeness, surface damage, discoloration, mold, freshness, leakage, and whether the item should be accepted, reviewed, or removed. Automating part of this process is attractive because repeated visual checks can be time-consuming and can vary between operators.

Computer vision offers a natural starting point. An object detector can locate several objects in an image and classify each object. However, a detector’s output is not equivalent to a quality report. A bounding box and confidence score indicate what the model believes is visible, but they do not provide an operational explanation. The central design problem in this project is therefore to connect fast visual localization with a second stage that can describe visible quality conditions.

### 1.2 Problem statement

The project addresses the following problem: how can a food-inspection application combine object detection and visual reasoning while remaining understandable, reusable across interfaces, and sufficiently structured for reporting? The implementation must support image upload through a web frontend and live or video input through an additional runner. Both paths should produce the same logical result rather than two slightly different interpretations of the same frame.

### 1.3 Objectives

The project objectives are to train or fine-tune a food object detector, integrate a VLM quality-assessment stage, produce farmer-readable commentary, support saved inspection reports, compare alternative VLM providers, and create a maintainable backend architecture. A further objective is to document the machine-learning pipeline rigorously, including data ingestion, preprocessing, augmentation, model initialization, training parameters, loss functions, optimizer choice, checkpointing, and evaluation.

### 1.4 Contributions

The main contributions are fourfold. First, the project implements an end-to-end inspection pipeline rather than a detection-only demo. Second, it defines a typed result contract that is shared by detection, reasoning, reporting, and frontend rendering. Third, it introduces a live runner that calls the same pipeline as the web backend, avoiding duplicated detection and commentary logic. Fourth, it provides a benchmark harness and a thesis-level interpretation of the first training experiment.

### 1.5 Thesis organization

Chapter 2 reviews the technical background. Chapter 3 presents requirements and system architecture. Chapter 4 explains the machine-learning pipeline. Chapter 5 describes implementation. Chapter 6 analyzes the first experiment. Chapter 7 presents the live-pipeline restructuring and VLM benchmark methodology. Chapter 8 discusses limitations and validity. Chapter 9 proposes the next experiment. Chapter 10 concludes the work. Appendices provide configuration tables, endpoint contracts, and a presentation plan.

## 2. Technical background

### 2.1 Object detection

Object detection combines classification and localization. For each detected object, the model predicts a class, a confidence, and a bounding box. In this project, the YOLO family is used because it is designed for single-stage detection and is practical for near-real-time applications. The detector is responsible for answering: “Which food items are visible, and where are they?”

The detector is not asked to produce a natural-language quality judgment. This separation is deliberate. It makes the detector measurable with standard object-detection metrics such as precision, recall, mAP50, and mAP50–95, while the VLM can be evaluated separately on quality reasoning and explanation.

### 2.2 Vision-language reasoning

A VLM accepts visual input together with language instructions and generates text. In this application, the VLM receives a crop of one detected item and a prompt containing the detected class, detector confidence, and relevant quality dimensions. The expected answer is a JSON object containing status, overall quality score, metric values, visible defects, explanation, and required action.

The VLM is not used to replace the detector. The detector narrows the image to candidate objects, reducing the VLM’s task from open-ended scene understanding to focused inspection. This also makes the quality prompt class-specific. For example, a tomato profile may emphasize ripeness, mold, bruising, and leakage, while a banana profile may emphasize ripeness, bruising, discoloration, and freshness.

### 2.3 Explainability and farmer communication

A raw model output is difficult to use operationally. The project therefore derives a commentary sentence from structured fields. The commentary includes the item name, quality score when available, visible defects or weakest metric, and recommended action. This is a practical form of post-hoc explanation: it does not expose the model’s internal reasoning, but it translates the observed structured assessment into a concise operational statement.

The system also generates aggregate report metrics. These include detected items, assessed items, assessment coverage, defect rate, uncertain rate, mean detector confidence, mean quality score, average quality metrics, recommended-action counts, and frequent defects. These metrics are intended to support a farm-inspector style summary while remaining traceable to individual detections.

## 3. Requirements and architecture

### 3.1 Functional requirements

The system must accept an image upload, decode it safely, run object detection, crop each detection, optionally invoke a VLM, validate the response, and return a typed inspection result. It must support detection-only mode when VLM reasoning is disabled or gated. It must save completed full inspections and expose report history and summary endpoints. It must also support a camera or video source without introducing a second inference implementation.

### 3.2 Non-functional requirements

The backend should be understandable to an internship manager and future maintainers. Provider credentials must remain outside source control. A malformed or refused VLM response must not crash the whole frame. A low-confidence detection should be able to bypass VLM reasoning through a confidence gate. The live runner should be able to produce JSONL output for later analysis. The implementation should avoid unnecessary rewrites of the existing frontend and database.

### 3.3 Architecture

The architecture is organized into five layers.

| Layer | Responsibility | Main repository element |
| --- | --- | --- |
| Input | Upload, camera, or video frame acquisition | FastAPI upload; `live_inspection.py` |
| Detection | Object classes, confidence, and boxes | YOLO model and `run_inspection()` |
| Reasoning | Crop-level quality assessment | `VLMBackend` adapters |
| Reporting | Commentary and aggregate metrics | `build_quality_commentary()` and `reporting.py` |
| Delivery | Frontend, persistence, JSONL, and export | API, SQLite, live runner |

The critical design principle is that interfaces differ only at the input and delivery edges. The core detection-to-reasoning function remains the same.

### 3.4 Data contract

The top-level `InspectionResult` contains a frame identifier, timestamp, source, image size, detection count, and a list of inspection items. Every item contains a `Detection` and a `QualityAssessment`. The quality assessment includes a status, optional score, metric dictionary, defects, explanation, commentary, required action, provider name, and optional latency.

This contract is important because it prevents the frontend from needing to understand provider-specific response formats. The VLM adapter parses provider output into the same Pydantic model regardless of whether the provider is Gemma, NVIDIA, OpenRouter, Qwen, or a local model.

## 4. Machine-learning pipeline

### 4.1 Data ingestion

The notebook uses a YOLO-format fruits-and-vegetables dataset stored in Google Drive. At the start of a run, the dataset is copied to local Colab storage. The notebook then discovers the dataset YAML, validates that `train`, `val`, and `names` exist, resolves paths, and writes a stable local YAML. This design addresses two practical issues: Drive storage is persistent but slower, and local training caches must be writable.

The experiment configuration identifies a 63-class dataset. The exact class distribution, number of images in each split, and label-quality statistics are not present in the supplied evidence and should be added to the final thesis from a dataset audit. Those values must not be guessed because class imbalance can strongly affect mAP and per-class performance.

### 4.2 Preprocessing

The configured image size is 640. YOLO preprocessing resizes and pads images to the configured shape while transforming bounding-box coordinates consistently. During training, labels are read in YOLO format. During validation, predictions are compared with ground-truth boxes using IoU thresholds to calculate precision, recall, and average precision.

The notebook uses a local disk cache rather than a Drive cache. This is a performance and reproducibility choice: the source dataset remains persistent in Drive, while the training process reads from a local copy with a stable path.

### 4.3 Data augmentation

The notebook explicitly uses the Ultralytics training pipeline and sets `close_mosaic=10`, meaning mosaic augmentation is disabled near the end of training to allow the model to adapt to more natural image composition. Other augmentation parameters appear to remain at the framework defaults unless overridden elsewhere in the notebook. The final thesis should report the effective values from the saved `args.yaml`, not only the intended notebook configuration.

Augmentation is useful because food images may vary in scale, position, lighting, and background. However, augmentation must preserve the semantic quality signal. Excessive geometric distortion or unrealistic color changes can make the training distribution diverge from the inspection environment. For quality assessment, color-related transformations deserve particular caution because ripeness and discoloration are meaningful visual attributes.

### 4.4 Model initialization and transfer learning

The experiment initializes `YOLO(FRESH_MODEL)` with `FRESH_MODEL='yolo11m.pt'`. This means the detector begins with pretrained weights rather than random weights. The correct description is therefore supervised transfer learning or fine-tuning of a pretrained YOLO11m detector.

Training from scratch would initialize the detector’s weights randomly and require the project dataset to teach both low-level visual features and task-specific features. Fine-tuning starts with general visual representations and adapts them to the food classes. Fine-tuning is usually more data-efficient and converges faster, while training from scratch can be justified when the domain is very different, the dataset is very large, or pretrained weights are unavailable.

The statement “I trained a CNN from scratch” is technically correct only for a model whose weights were randomly initialized and trained without pretrained checkpoints. The fact that a training function was called does not determine whether training was from scratch; initialization does.

### 4.5 Training parameters

| Parameter | Supplied configuration | Rationale |
| --- | ---: | --- |
| Model | `yolo11m.pt` | Medium detector balancing accuracy and compute |
| Target epochs | 80 | Longer than the earlier 30-epoch baseline because prior curves were still improving |
| Image size | 640 | Standard practical resolution for detection and memory control |
| Batch | `-1` | Automatic safe batch selection for variable free-Colab GPUs |
| Workers | 2 | Conservative Colab setting |
| Cache | `disk` | Local speed without using Drive as a training cache |
| Device | 0 | Assigned GPU |
| Seed | 0 | Reproducibility control |
| Patience | 20 | Early stopping if validation fitness plateaus |
| Save period | 5 | Additional periodic checkpoint preservation |
| Optimizer | AdamW | Explicit reproduction of the optimizer selected by the previous `auto` run |
| Initial learning rate | 0.000149 | Small adaptation step for transfer learning |
| Momentum | 0.9 | Controls the running direction of AdamW updates in the configured framework |
| Weight decay | 0.0005 | Regularization against overfitting |
| Warm-up | 3 epochs | Stabilizes early optimization |
| Final LR fraction | 0.01 | Learning-rate decay toward the end of training |
| Box loss gain | 7.5 | Emphasizes localization error |
| Classification loss gain | 0.5 | Balances class prediction against localization |
| DFL loss gain | 1.5 | Supports distribution-based box regression |

### 4.6 Loss functions

YOLO detection training combines localization, classification, and distribution-focused regression terms. Box loss penalizes inaccurate box geometry. Classification loss penalizes incorrect class assignment. Distribution Focal Loss contributes to more precise localization by modeling the distribution of box boundaries rather than predicting only a single coordinate value.

The configured gains are not arbitrary application losses written by the student. They are the framework’s validated detection-loss balance retained for the first experiment. This is a defensible choice for a baseline because changing multiple loss weights before understanding the data would make the experiment harder to interpret. A later ablation could compare the default gains with carefully selected alternatives, but the first comparison should keep them fixed.

### 4.7 Optimizer and regularization

AdamW is used because it combines adaptive per-parameter learning rates with decoupled weight decay. Adaptive updates are useful when pretrained features and newly adapted task-specific features have different gradient scales. Weight decay acts as a regularizer by discouraging unnecessarily large weights. The learning rate is deliberately small because fine-tuning should adapt the pretrained representation without destroying useful general features.

Other regularization mechanisms include data augmentation, early stopping through patience, and validation-based best-checkpoint selection. Checkpointing itself is not regularization; it is an experiment-resilience mechanism. The notebook’s Drive backups protect progress after Colab interruption and make it possible to distinguish a true interrupted resume from a new fine-tuning stage.

### 4.8 Resume versus fine-tuning

An exact interrupted resume uses `last.pt` and continues the same run, including optimizer and scheduler state when supported by the checkpoint. It is intended for an interruption before the target run is complete. A new fine-tuning stage starts from a completed checkpoint such as `best.pt` or `last.pt` and creates a new experiment identity. It is not an exact resume because the stage may have a new learning rate, warm-up, epoch budget, or experiment directory.

This distinction should be stated explicitly in the school report. Otherwise, the comparison between the first and next experiment can become scientifically ambiguous.

## 5. First experiment analysis

### 5.1 Observed evidence

The supplied continuation log reports epochs 50–62. The validation set contains 1,492 images and 33,683 instances in the displayed summary. The metrics fluctuate from epoch to epoch, which is normal in stochastic optimization and can also reflect class imbalance or a validation set that contains difficult examples.

| Epoch | Precision | Recall | mAP50 | mAP50–95 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.338 | 0.260 | 0.239 | 0.172 |
| 55 | 0.399 | 0.247 | 0.236 | 0.170 |
| 60 | 0.365 | 0.265 | 0.244 | 0.177 |
| 62 | 0.378 | 0.259 | 0.239 | 0.173 |

The detector losses generally decline across the window: box loss moves from 0.9910 at epoch 50 to 0.9454 at epoch 62, classification loss from 0.7819 to 0.6993, and DFL loss from 0.9957 to 0.9788. Validation mAP does not improve monotonically. The best supplied-window values are mAP50 0.244 and mAP50–95 0.177 at epoch 60.

### 5.2 Interpretation

The evidence suggests that optimization is still making progress in training loss, but generalization has reached a noisy plateau in the observed window. This does not prove overfitting because training and validation loss curves, per-class AP, and the complete run are not available. It does indicate that simply increasing epochs may not solve the main problem.

Before changing the architecture, the next analysis should inspect label quality, class frequencies, small-object frequency, train/validation overlap, and per-class AP. A low overall mAP can be caused by a small group of rare or visually ambiguous classes. Aggregate metrics alone cannot identify that cause.

### 5.3 Validity limitations

The log snapshot is not a final test result. It is a continuation record. The displayed validation count suggests a substantial evaluation set, but the independent test split and its results are not supplied. The final thesis should include the final `best.pt` validation and test metrics, confusion matrix, precision-recall curves, and per-class AP table.

## 6. Implementation of the inspection pipeline

### 6.1 Detection stage

`run_inspection()` accepts a NumPy image, YOLO model, frame identifier, source label, optional VLM backend, and confidence gate. It invokes the YOLO model once, extracts classes and boxes, normalizes coordinates, and builds typed `Detection` objects.

### 6.2 Crop and VLM stage

For each detection above the confidence gate, the backend crops the image and calls `VLMBackend.analyze`. The adapter builds a class-specific prompt using the quality profile. The provider output is cleaned, parsed, and validated into a `QualityAssessment`. If the provider fails or returns invalid JSON, the system creates an `uncertain` assessment requiring manual review rather than terminating the entire frame.

### 6.3 Commentary stage

The commentary builder converts structured values into a concise paragraph. For a defect, it reports the main visible issues. For an uncertain response, it explicitly states that manual review is needed. For an acceptable result, it indicates that no visible quality defects were identified. The phrasing is intentionally operational rather than academic.

### 6.4 Report stage

The report builder aggregates item-level results. It computes assessment coverage, defect rate, uncertain rate, mean detector confidence, mean quality score, metric averages, recommended-action counts, and frequent defects. The report includes a summary for the farmer and preserves the original inspection result so that aggregate decisions remain traceable.

## 7. Shared frontend and live execution

### 7.1 Existing frontend flow

The frontend submits an image to `/inspect`. The API returns a job identifier immediately, and the frontend polls the status endpoint until the result is complete. This avoids keeping the browser request open while several VLM crops are analyzed.

### 7.2 New live runner

The new `backend/live_inspection.py` runner accepts a camera index, video path, or other OpenCV source. For each frame, it calls the same `run_inspection()` function as the API. It writes optional JSONL records containing the inspection result, farmer report, and processing time. When display mode is enabled, it renders boxes and status labels.

This choice intentionally prioritizes semantic consistency over a separate low-latency implementation. The legacy `backend/main.py` contains tracking and asynchronous VLM logic, but it reimplements the detection orchestration. The new runner gives the project a clean reference path for proving that frontend and live execution are behaviorally identical. If throughput becomes insufficient, asynchronous scheduling can later be optimized behind the same pipeline contract.

## 8. Gemma versus NVIDIA VLM benchmark

### 8.1 Provider integration

The project now supports hosted Gemma through the Gemini API and NVIDIA-hosted or self-hosted NIM-style VLM access through an OpenAI-compatible client. Google documents hosted Gemma 4 models, including `gemma-4-31b-it` and `gemma-4-26b-a4b-it`, with image understanding through the Gemini API [1]. NVIDIA documents an OpenAI-compatible VLM inference endpoint at `/v1/chat/completions`, along with model, health, metadata, and metrics endpoints [2].

The provider adapters do not change the production prompt or parser. This is essential for a fair comparison: the model provider is the independent variable, while crop, prompt, output schema, and postprocessing remain fixed.

### 8.2 Benchmark protocol

The benchmark should use a fixed set of representative crops sampled across classes, lighting conditions, defect types, and image quality levels. Each crop should have a human reference label covering status, defects, quality metrics, and recommended action. The same image bytes and prompt should be sent to both providers with temperature zero where supported.

The benchmark script records provider, model, status, score, defects, required action, provider latency, wall latency, explanation presence, and explanation text. The final analysis should report valid-JSON rate, status accuracy, defect precision and recall, action agreement, score correlation, median latency, p95 latency, and failure rate.

### 8.3 Free-tier interpretation

Google’s current pricing page describes a free tier with limited access to certain models, free input and output tokens, and a data-use condition stating that free-tier content may be used to improve products [3]. NVIDIA’s official documentation describes free access through the NVIDIA Developer Program for self-hosting NIMs, while hosted API availability and quotas should be verified for the selected account and model. “Free” therefore has different meanings: a hosted API free tier may be quota-limited, while self-hosting may have no per-request fee but requires suitable GPU resources and setup.

### 8.4 Decision rule

NVIDIA should be recommended as the primary provider only if it wins on the metrics that matter for this application. The decision rule is: first require reliable structured outputs, then compare agreement with human references, then compare latency and operational cost. A model should not be selected solely because its explanations sound more fluent.

## 9. Limitations and ethical considerations

The detector’s quality depends on the dataset labels and visual coverage. A model can appear accurate on images similar to training data and fail under different cameras, backgrounds, lighting, cultivars, packaging, or disease manifestations. The VLM can also overinterpret ambiguous visual evidence. For this reason, uncertain outputs must remain visible and must trigger manual review rather than automatic removal.

The system is a decision-support prototype, not a replacement for a qualified inspector or food-safety authority. A quality score derived from an image is not a laboratory measurement. It should not be presented as proof of microbiological safety. The report language must remain limited to visible quality indicators and recommended review actions.

Provider data-use terms must be documented before production deployment. The current Google free-tier documentation states that free-tier content may be used to improve products [3]. This may be unacceptable for confidential operational images. Self-hosting or an enterprise plan may be needed if privacy requirements are strict.

## 10. Proposed next experiment

The next experiment should freeze the dataset and experiment identity before training. It should record the exact class list, image counts, instance counts, label statistics, augmentation values from `args.yaml`, and hardware. Training should complete the 80-epoch target or terminate only through recorded early stopping. The final `best.pt` should be evaluated on validation and independent test splits.

The detector study should report overall precision, recall, mAP50, mAP50–95, per-class AP, confusion matrix, and error examples. It should compare the first run with the new run under one changed factor at a time. If the goal is to test fine-tuning strategy, keep dataset and evaluation fixed while changing initialization or fine-tuning schedule. If the goal is to improve data quality, keep the model and hyperparameters fixed while changing labels or augmentation.

The end-to-end study should create a human-labeled crop benchmark. At least two human reviewers should label a subset so that disagreement can be measured. The Gemma and NVIDIA providers should then be compared using identical prompts and crops. Latency should be measured over repeated requests, and failures should be included rather than discarded.

## 11. Conclusion

This project demonstrates a practical and explainable food-inspection architecture. YOLO11m provides object localization and classification, while a VLM evaluates visible quality dimensions. The typed schema and farmer-report layer turn model outputs into operationally understandable records. The backend restructuring is intentionally conservative: it adds a shared live runner, reporting utilities, provider adapters, and benchmark tooling without replacing the existing application.

The first experiment provides useful evidence but should not be overstated. The notebook records transfer learning from pretrained YOLO11m, reproducible training settings, resumable checkpoints, and progress through epoch 62. The observed metrics show modest and noisy validation performance, with the best supplied-window mAP50–95 of approximately 0.177. The next contribution should therefore be controlled evaluation and error analysis rather than an unmotivated increase in model complexity.

The project’s strongest engineering principle is consistency: image upload and live camera input now have one detection-to-inspection path. Its strongest research principle is separation of evidence from expectation: a provider may seem better in practice, but the final recommendation should be supported by a fixed benchmark, human references, latency measurements, and privacy or quota analysis.

## References

[1]: https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api "Run Gemma with the Gemini API — Google AI for Developers"
[2]: https://docs.nvidia.com/nim/vision-language-models/latest/api-reference.html "NVIDIA NIM for Vision Language Models — API Reference"
[3]: https://ai.google.dev/gemini-api/docs/pricing "Gemini Developer API pricing — Google AI for Developers"

## Appendix A — Environment variables

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` or `GEMMA_API_KEY` | Hosted Gemma authentication |
| `GEMMA_API_BASE_URL` | Optional override for the Gemma API base URL |
| `NVIDIA_API_KEY` | NVIDIA API or NIM authentication |
| `NVIDIA_API_BASE_URL` | Hosted or self-hosted NVIDIA OpenAI-compatible endpoint |
| `NVIDIA_VLM_MODEL` | Default NVIDIA VLM model identifier |
| `OPENROUTER_API_KEY` | Existing OpenRouter backend |

## Appendix B — Demonstration commands

```bash
# Run the shared pipeline on a webcam with detection only.
python -m backend.live_inspection --source 0 --display

# Run the shared pipeline with Gemma and write JSONL reports.
python -m backend.live_inspection --source 0 --vlm-backend gemma --output runtime_artifacts/gemma_live.jsonl

# Run the shared pipeline with NVIDIA.
python -m backend.live_inspection --source 0 --vlm-backend nvidia --output runtime_artifacts/nvidia_live.jsonl

# Benchmark equal crops after setting provider credentials.
python -m backend.vlm_benchmark path/to/crops --backends gemma nvidia --output runtime_artifacts/vlm_benchmark.csv
```

## Appendix C — Page-count expansion plan

To reach a school-specific 40-page requirement without padding, expand the manuscript with a dataset audit table, class-distribution chart, annotation examples, architecture diagram, training curves, confusion matrix, per-class AP table, error taxonomy, VLM prompt examples, benchmark results, latency distributions, screenshots of the frontend, API request/response examples, and a detailed comparison of the first and next experiment. The current draft supplies the argument and evidence structure; these measured artifacts should be inserted after the experiments are complete.
