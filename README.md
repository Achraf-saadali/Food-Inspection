# Food Inspection AI

Food Inspection AI is a two-stage computer-vision system for detecting food ingredients and assessing their visible quality. The repository combines a YOLO-based object detector, optional vision-language reasoning, a FastAPI service, a React/TypeScript dashboard, training artifacts, and saved inspection outputs.

The documentation below describes the **implemented repository**, not an idealized future architecture. Where a dataset split, class list, or metric is not directly reproducible from the files committed here, that limitation is stated explicitly.

## System at a glance

The system has two operational paths. The detection-only path is designed for low-latency localization. The full inspection path first detects objects, crops each detected object, selects ingredient-specific quality dimensions, sends eligible crops to the configured VLM adapter, validates the returned JSON, and exposes a unified result to the dashboard.

```text
Browser image upload
        |
        v
React LiveInspection page
        |
        | multipart/form-data
        v
FastAPI POST /detect --------------------> YOLO detection
        |
        | POST /inspect -> job_id
        v
In-memory background job
        |
        v
YOLO detections -> crop -> confidence gate -> OpenRouter VLM
        |                                      |
        |                                      v
        |                              validated QualityAssessment
        v
GET /inspect/status/{job_id} <--------- unified InspectionResult
        |
        v
React overlays, cards, status, metrics, export
```

## What is implemented

| Capability | Implementation | Current status |
|---|---|---|
| Ingredient localization | Ultralytics YOLO model loaded from `models/best.pt` | Implemented |
| Detection API | `POST /detect` | Implemented |
| Full quality inspection | `POST /inspect` plus status polling | Implemented |
| Structured quality reasoning | Ingredient profiles plus VLM JSON parsing | Implemented |
| Browser upload | Drag-and-drop/file selection in `LiveInspection.tsx` | Implemented |
| Live webcam/video CLI | `backend/live_inference.py` | Implemented |
| Full local quality CLI | `backend/main.py` | Implemented |
| Farmer-facing quality commentary | Deterministic text derived from score, defects, and action | Implemented |
| Dashboard reports/statistics | SQLite-backed API endpoints and React report views | Implemented; no mock inspection data |
| Persisted inspection history | Local SQLite database plus uploaded-image references | Implemented for completed full inspections |
| Async job state | Process-local dictionary during active processing | Not durable; clears on restart |
| Dataset manifest and split metadata | Dataset YAML is referenced externally from the training run | Not committed |

## Repository organization

```text
.
├── backend/
│   ├── api.py                  # FastAPI endpoints, jobs, reports, and export
│   ├── database.py             # Single SQLite connection, report storage, and summaries
│   ├── inspection_pipeline.py  # YOLO -> crop -> optional VLM orchestration
│   ├── live_inference.py       # Detection-only webcam/video/image CLI
│   ├── main.py                 # Full local webcam quality-inspection CLI
│   ├── quality_profiles.py     # Ingredient-specific VLM metrics
│   ├── schemas.py              # Shared Pydantic response contract
│   └── vlm_reasoning.py       # VLM interface, adapters, parsing, fallback
├── frontend/
│   ├── client/src/api/         # Axios and inspection request functions
│   ├── client/src/components/  # Layout, inspection cards, overlays, UI primitives
│   ├── client/src/hooks/       # React state orchestration
│   ├── client/src/pages/       # Dashboard, live inspection, reports, model views
│   ├── client/src/types/       # TypeScript mirror of backend contracts
│   └── package.json             # Frontend scripts and dependencies
├── models/
│   └── best.pt                 # Runtime detector weights
├── training/
│   ├── notebooks/              # Training notebook
│   └── runs/                   # Ultralytics run configuration, plots, metrics, weights
├── runtime_artifacts/
│   ├── outputs/                # Saved uploads, snapshots, and JSONL sessions
│   └── food_inspection.db      # Local SQLite report history, created at runtime
├── docs/
│   ├── QUALITY_ASSESSMENT.md   # Quality-assessment contract and profiles
│   ├── Final_Report.md         # Timeout diagnosis and async-job history
│   ├── diagnosis.md            # Earlier timeout analysis
│   └── assets/architecture.png # Architecture visual
├── requirements.txt
└── README.md
```

All existing artifacts were retained during restructuring. The spelling of the former `trainning_runs` directory was corrected by moving it to `training/runs`; no training weights, predictions, plots, snapshots, logs, or notebooks were removed.

## Machine-learning training record

The repository now uses the second experiment, `lvis_fruits_yolo11m_80_v1`, as its canonical training pipeline. The notebook in `training/notebooks/food_detection.ipynb` replaces the original ad-hoc YOLOv9c/30-epoch workflow with a reproducible YOLO11m pipeline that copies the dataset to writable local storage, audits labels before training, removes the two case-duplicate classes (`Strawberry` and `Tomato`), remaps the remaining taxonomy to 61 contiguous classes, and checks for train/validation overlap.

| Group | Integrated values |
|---|---|
| Architecture and schedule | pretrained `yolo11m.pt`, `epochs=80`, `imgsz=640`, `batch=-1` automatic GPU sizing |
| Optimizer and learning rate | `AdamW`, `lr0=0.000149`, `momentum=0.9`, `weight_decay=0.0005`, `lrf=0.01` |
| Training control | `patience=20`, `warmup_epochs=3`, `cos_lr=true`, `close_mosaic=10`, `amp=true`, `cache=disk`, `workers=2` |
| Reproducibility | `seed=0`, `deterministic=true`, pretrained initialization |
| YOLO loss weights | `box=7.5`, `cls=0.5`, `dfl=1.5` |
| Dataset processing | 63-class source taxonomy reduced to 61 classes; images containing dropped classes or no remaining annotations are removed |
| Checkpointing | `best.pt`, `last.pt`, and immutable epoch checkpoints every five epochs |

The processed dataset audit records 4,707 training images with 119,422 instances and 1,532 validation images with 33,695 instances. The dataset YAML, audit manifest, class distribution, training configuration, and epoch metrics are retained under `training/data/lvis_fruits_61class/` and `training/runs/lvis_fruits_yolo11m_80_v1/`. The raw dataset remains external and is not committed.

### Experiment comparison

The second experiment is the more relevant engineering baseline because it uses a larger YOLO11m backbone, a longer controlled schedule, explicit AdamW settings, cosine learning-rate decay, automatic batch sizing, writable disk caching, label validation, duplicate-class remediation, overlap checks, and periodic checkpoint persistence. It is not a clean isolated hyperparameter ablation because the architecture, dataset taxonomy, number of images, and schedule changed together.

| Metric | Experiment 1: YOLOv9c, 30 epochs | Experiment 2: YOLO11m, cleaned 61 classes, 80 epochs | Difference in best recorded value |
|---|---:|---:|---:|
| Best precision (B) | 0.46271 | 0.40785 | −0.05486 |
| Best recall (B) | 0.26704 | 0.26546 | −0.00158 |
| Best mAP@50 (B) | 0.23750 | 0.24364 | +0.00614 |
| Best mAP@50–95 (B) | 0.17418 | 0.17677 | +0.00259 |

The second run’s best recorded validation metrics occurred around logged epoch 60, while its final epoch-80 metrics declined to mAP@50 `0.23402` and mAP@50–95 `0.16956`. Therefore the application should use `best.pt`, not `last.pt`, and the comparison should not claim uniform improvement on every metric. The two experiments also use different effective class spaces and splits, so the metric deltas are directional evidence rather than a controlled scientific attribution.

### Runtime model and reproducibility

The runtime application now reports the second experiment’s metadata through `/model-info` and loads the integrated YOLO11m `models/best.pt` checkpoint. The checkpoint is tracked with Git LFS because it is approximately 116 MB. The training notebook is configured for a fresh run by default; interrupted runs must use the same experiment identity and `resume_interrupted` mode with the corresponding `last.pt`, while a completed checkpoint followed by new training must use `extend_completed` and a new experiment identity.

The metrics above are the values recorded by Ultralytics in `results.csv`; they are not a substitute for a final independent test benchmark. A production evaluation should still include a locked held-out test split, per-class AP, confidence-threshold analysis, duplicate/near-duplicate checks across splits, and hardware-specific latency measurements.


## Detection and quality contracts

`backend/schemas.py` is the central contract shared by the API and the frontend. `Detection` stores the label, integer class ID, confidence, absolute `bbox_xyxy` coordinates, and normalized coordinates. `QualityAssessment` stores the VLM status, optional overall score, metric dictionary, defect labels, explanation, required action, backend name, and measured latency. `InspectionItem` combines one detection with one quality result. `InspectionResult` adds frame ID, timestamp, source, image size, detection count, and item list.

The status enum is `ok`, `defect`, `uncertain`, or `skipped`. `skipped` is used when the VLM is disabled, when a detection is below the confidence gate, or when a crop cannot be processed. A failed or unparsable VLM response degrades to `uncertain` and requests `flag_for_review`; it does not crash the entire frame pipeline.

A representative result is shaped like this:

```json
{
  "frame_id": 17,
  "timestamp": "2026-08-13T10:00:00",
  "source": "upload.jpg",
  "image_size": {"width": 1280, "height": 720},
  "num_detections": 1,
  "items": [
    {
      "detection": {
        "label": "banana",
        "class_id": 4,
        "confidence": 0.91,
        "bbox_xyxy": [120.0, 80.0, 420.0, 500.0],
        "bbox_normalized": [0.09375, 0.11111, 0.32812, 0.69444]
      },
      "quality": {
        "status": "defect",
        "overall_quality_score": 0.45,
        "quality_metrics": {"browning": 0.70, "bruising": 0.50},
        "defects": ["browning", "bruising"],
        "explanation": "Visible browning and bruising are present.",
        "required_action": "remove",
        "vlm_backend": "openrouter",
        "latency_ms": 1840.2
      }
    }
  ]
}
```

## Ingredient-aware VLM reasoning

`backend/quality_profiles.py` maps supported labels to visual dimensions. Profiles are present for tomato, banana, apple, potato, strawberry, and onion. Unknown labels fall back to `COMMON_METRICS`: ripeness, bruising, mold, discoloration, and freshness. The profile is inserted into the prompt so the VLM evaluates dimensions appropriate to the detected ingredient rather than using one generic checklist.

`backend/vlm_reasoning.py` defines the abstract `VLMBackend` interface. Its `analyze()` method builds the prompt, times the model call, strips common Markdown fences or surrounding text, parses JSON, validates the semantic fields through Pydantic, and records latency. The module contains adapters for local Qwen2.5-VL, GPT-4o, DashScope Qwen, and OpenRouter. **The current factory returns `OpenRouterBackend` for every requested name.** The other adapters remain in the codebase, but the runtime selection is intentionally documented as OpenRouter-only until the factory is changed.

The default OpenRouter model in the adapter is `google/gemini-flash-1.5-8b`. The API path passes `openrouter` explicitly. The local CLI still exposes several historical `--vlm` choices, but the factory behavior takes precedence, so users should configure `OPENROUTER_API_KEY` for the implemented full-quality path.

## Farmer-facing quality commentary

The VLM still returns machine-readable status, score, metrics, defects, explanation, and action. `backend/inspection_pipeline.py` then derives a short operator paragraph from those fields, without a second model call. The commentary prioritizes explicit defects, otherwise names the weakest available metric, and always states the quality score when available and recommended action. The dashboard and report detail cards show this **Quality summary** before the raw visual evidence and metric bars.

For example: *“Tomato. Quality score: 0.62/1.00. Observed issue: bruising. Recommended action: flag for review.”* The original structured values remain in the API result and JSON export for traceability.

## Persistent reports and end-to-end image flow

The browser entry point is `frontend/client/src/pages/LiveInspection.tsx`. A user drops an image or selects a file. The page passes the file and selected mode to `useInspection`, which delegates to `frontend/client/src/api/inspectionApi.ts`.

For detection-only mode, the client sends the image as a `multipart/form-data` field named `file` to `POST /detect`. FastAPI reads the upload, decodes the bytes with OpenCV, loads the cached YOLO model, and calls `backend.inspection_pipeline.run_inspection()` with no VLM backend.

For full inspection mode, the client sends the same multipart field plus `confidence_gate` to `POST /inspect`. The endpoint decodes and saves the uploaded image under `runtime_artifacts/outputs/uploads/`, creates a UUID job, stores `pending` state in the process-local `_jobs` dictionary, and schedules `process_inspection_job` through FastAPI `BackgroundTasks`. The immediate response is `{"job_id": "...", "status": "pending"}`.

The frontend then polls `GET /inspect/status/{job_id}` approximately once per second. While waiting, it presents upload, YOLO, and VLM progress stages. When processing completes, the API writes the full result, image reference, quality scores, defects, actions, and farmer commentary into `runtime_artifacts/food_inspection.db`, then returns the persisted `InspectionResult` with a stable `report_id`. If processing fails, it returns a failed status and error message. This avoids holding the original HTTP request open while external VLM calls execute.

Inside `run_inspection()`, YOLO produces boxes, class IDs, labels, and confidence scores. Coordinates are clamped while cropping and normalized against the original image width and height. A crop is sent to the VLM only when a backend exists and the detector confidence is at least the configured gate. The result is then rendered by the frontend through `BoundingBoxOverlay`, `DetectionCard`, `ConfidenceBar`, `StatusBadge`, and the live inspection result panel.

The frontend API base URL is `VITE_API_BASE_URL`, defaulting to `http://localhost:8000`. The Axios client uses a 120-second timeout for polling requests and normalizes backend error fields into user-facing errors.

## Important file-by-file guide

| File | Responsibility and important behavior |
|---|---|
| `backend/api.py` | FastAPI application, CORS, cached YOLO/VLM constructors, upload decoding, `/detect`, `/inspect`, status polling, and `/health`. The job store is in memory and is not durable or shared across workers. |
| `backend/database.py` | The single SQLite module. It creates the local database, persists completed inspection results and image paths, reads real report history, aggregates dashboard metrics, and produces JSON exports. |
| `backend/inspection_pipeline.py` | Framework-neutral orchestration. It converts Ultralytics boxes into Pydantic detections, crops objects, applies the VLM confidence gate, generates farmer-facing commentary, and preserves typed skipped results. |
| `backend/live_inference.py` | Detection-only CLI for webcam, video, and still images. It annotates frames and writes JSONL logs plus snapshots under `runtime_artifacts/outputs`. |
| `backend/main.py` | Full local webcam/video application. It runs detection and optional quality reasoning, draws status-colored boxes, writes structured session logs, and saves snapshots. |
| `backend/quality_profiles.py` | Controlled ingredient-to-metric mapping and fallback metrics for unknown classes. |
| `backend/schemas.py` | Pydantic models and enums shared by detection, VLM reasoning, API serialization, and frontend expectations. |
| `backend/vlm_reasoning.py` | Prompt generation, provider adapters, JSON extraction, semantic normalization, fallback assessment, and latency measurement. |
| `frontend/client/src/api/client.ts` | Axios base URL, polling timeout, development request logging, and normalized API errors. |
| `frontend/client/src/api/inspectionApi.ts` | Multipart upload, job submission, one-second status polling, real SQLite report/history/summary requests, export, and frontend-facing progress callbacks. |
| `frontend/client/src/hooks/useInspection.ts` | React state for result, loading, stage, and error; selects detection-only or full inspection mode. |
| `frontend/client/src/pages/LiveInspection.tsx` | Main real inspection screen: file input, drag/drop, mode selection, progress, image overlay, overall status, and per-item results. |
| `frontend/client/src/components/inspection/BoundingBoxOverlay.tsx` | Positions normalized detection boxes over the displayed image. |
| `frontend/client/src/components/inspection/DetectionCard.tsx` | Displays class, confidence, quality score, farmer-facing commentary, visual evidence, defects, metrics, action, backend, and latency. |
| `frontend/client/src/components/inspection/ConfidenceBar.tsx` | Visual confidence/score indicator. |
| `frontend/client/src/components/inspection/StatusBadge.tsx` | Visual status mapping for quality states. |
| `frontend/client/src/pages/Dashboard.tsx` | Dashboard shell and summary presentation backed by the real persisted-report summary endpoint. |
| `frontend/client/src/pages/Reports.tsx` | Search, sort, filter, farmer-ready quality details, defect-frequency summary, JSON preview, and export for real persisted inspection history. |
| `frontend/client/src/pages/ModelInfo.tsx` | Model and API presentation page populated from the backend’s deployed detector and committed training artifacts. |
| `frontend/client/src/types/inspection.ts` | TypeScript mirror of the backend inspection contracts plus reporting/model view types. |
| `training/notebooks/food_detection.ipynb` | Exploratory/training notebook retained as a source artifact. |
| `training/runs/train4/` | Evidence from the recorded training run: configuration, CSV metrics, plots, predictions, validation images, and weights. |
| `runtime_artifacts/outputs/` | Existing JSONL sessions and saved image/JSON snapshots retained as runtime examples. |
| `docs/QUALITY_ASSESSMENT.md` | Detailed quality schema and profile extension notes. |
| `docs/Final_Report.md` and `docs/diagnosis.md` | Historical timeout analysis explaining the move from synchronous inspection to background jobs with polling. |

The many files under `frontend/client/src/components/ui/` are reusable generated-style UI primitives. They are intentionally not duplicated in this README individually because they do not contain domain-specific ML or transport logic; the domain-specific inspection components are documented above.

## Running the project

### Backend installation

Use Python 3.9 or newer and install the project requirements:

```bash
git clone https://github.com/Achraf-saadali/Food-Inspection.git
cd Food-Inspection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the environment variables required by the selected runtime. Full VLM inspection currently needs `OPENROUTER_API_KEY`. The local Qwen and alternative API adapters remain available as classes but are not selected by the current factory.

### Start the FastAPI backend

Run from the repository root:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints are:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/detect` | Multipart image upload and YOLO-only result |
| `POST` | `/inspect` | Multipart image upload and immediate job ID |
| `GET` | `/inspect/status/{job_id}` | Poll job status and retrieve completed result |
| `GET` | `/reports` | Read real persisted inspection history |
| `GET` | `/reports/summary` | Read real dashboard and defect metrics |
| `GET` | `/reports/export` | Export saved results as JSON |

A minimal detection request is:

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@path/to/image.jpg"
```

A full inspection request is:

```bash
curl -X POST "http://localhost:8000/inspect?confidence_gate=0.4" \
  -F "file=@path/to/image.jpg"
```

### Start the frontend

```bash
cd frontend
pnpm install
pnpm dev
```

To use another backend URL, create `frontend/.env.local` with:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

### Run local CLI inference

Detection-only inference:

```bash
python -m backend.live_inference
python -m backend.live_inference --source path/to/image.jpg
python -m backend.live_inference --source path/to/video.mp4 --conf 0.4
```

Full webcam quality inspection:

```bash
OPENROUTER_API_KEY=your_key_here python -m backend.main --vlm openrouter --source 0
```

Press `q` or `Esc` to exit. Press `s` to save an annotated image and JSON report. CLI outputs are written to `runtime_artifacts/outputs/`.

## Operational limitations and next engineering steps

Completed full-inspection results are persisted locally in SQLite, but the process-local `_jobs` dictionary used while work is in progress disappears on restart and is not shared between workers. A production deployment should replace the active-job dictionary with durable job storage and a worker system, then add job expiration and concurrency limits.

The API currently allows all CORS origins. This is convenient for development but should be restricted to the deployed frontend origin in production. Upload validation should also enforce file size, MIME type, image dimensions, and a safe decoding policy.

The repository contains no committed dataset manifest or test suite for detector quality. To make the training result reproducible, commit a sanitized dataset YAML, split counts, class names, data provenance, and an evaluation script. To make the product metrics trustworthy, replace the mocked frontend reporting helpers with API-backed persistence and add end-to-end tests for upload, polling, schema validation, and failed VLM calls.

## References

[1]: https://docs.ultralytics.com/modes/train/ "Ultralytics training mode documentation"
[2]: https://docs.ultralytics.com/tasks/detect/ "Ultralytics object detection documentation"
[3]: https://fastapi.tiangolo.com/tutorial/background-tasks/ "FastAPI background tasks documentation"
[4]: https://fastapi.tiangolo.com/tutorial/request-files/ "FastAPI file upload documentation"
[5]: https://docs.pydantic.dev/latest/ "Pydantic documentation"
[6]: https://github.com/WongKinYiu/yolov9 "YOLOv9 reference repository"
[7]: https://www.lvisdataset.org/ "LVIS dataset project"
