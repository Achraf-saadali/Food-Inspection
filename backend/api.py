"""
FastAPI surface for the inspection pipeline.

    POST /detect   -> YOLO detection only (matches current backend/live_inference.py output)
    POST /inspect  -> YOLO detection + VLM quality reasoning (unified JSON)

Run with:
    uvicorn backend.api:app --reload --port 8000

This replaces the current local cv2.imshow loop with a request/response
interface, per README Section 17 ("Integration Notes").
"""

from __future__ import annotations

import csv
import os
import uuid
from pathlib import Path
from typing import Dict, Any

import yaml
from functools import lru_cache
from dotenv import load_dotenv

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from backend.database import (
    export_reports,
    get_report,
    get_report_summary,
    initialize_database,
    list_reports,
    save_inspection,
    save_uploaded_image,
)
from backend.inspection_pipeline import run_inspection
from backend.reporting import build_farmer_report
from backend.schemas import InspectionResult, InspectionStatus
from backend.vlm_reasoning import get_backend

load_dotenv()

app = FastAPI(title="Food Inspection API", version="0.5.0")
initialize_database()

# Allow the Vite dev server (port 3000) and any production frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "best.pt")
TRAINING_ARGS_PATH = PROJECT_ROOT / "training" / "runs" / "lvis_fruits_yolo11m_80_v1" / "args.yaml"
TRAINING_RESULTS_PATH = PROJECT_ROOT / "training" / "runs" / "lvis_fruits_yolo11m_80_v1" / "results.csv"
_frame_counter = 0

# In-memory job store for async inspection
# Format: { "job_id": {"status": "pending"|"processing"|"completed"|"failed", "result": InspectionResult|None, "error": str|None} }
_jobs: Dict[str, Dict[str, Any]] = {}


@lru_cache(maxsize=1)
def get_yolo_model() -> YOLO:
    return YOLO(MODEL_PATH)


@lru_cache(maxsize=4)
def get_vlm_backend(name: str):
    return get_backend(name)


def _decode_upload(raw_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Impossible de décoder l’image")
    return image


@app.post("/detect", response_model=InspectionResult)
async def detect(file: UploadFile = File(...)) -> InspectionResult:
    """Detection-only endpoint. No VLM call, no extra latency/cost."""
    global _frame_counter
    image = _decode_upload(await file.read())
    _frame_counter += 1
    return run_inspection(
        image=image,
        yolo_model=get_yolo_model(),
        frame_id=_frame_counter,
        source=file.filename or "upload",
        vlm_backend=None,
    )


def process_inspection_job(
    job_id: str,
    image: np.ndarray,
    filename: str,
    image_path: str,
    vlm_backend_name: str,
    vlm_model: str | None,
    confidence_gate: float,
):
    global _frame_counter
    import time
    
    _jobs[job_id]["status"] = "processing"
    
    try:
        start_time = time.perf_counter()
        _frame_counter += 1
        backend = get_backend(vlm_backend_name, model=vlm_model)
        
        print(f"[API] Processing job {job_id} for {filename} (Frame {_frame_counter})")
        
        result = run_inspection(
            image=image,
            yolo_model=get_yolo_model(),
            frame_id=_frame_counter,
            source=filename or "upload",
            vlm_backend=backend,
            vlm_confidence_gate=confidence_gate,
        )
        
        total_time = time.perf_counter() - start_time
        print(f"[API] Job {job_id} complete. Total time: {total_time:.3f}s")
        
        result = save_inspection(result, report_id=job_id, image_path=image_path)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result
    except Exception as e:
        print(f"[API] Job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


@app.post("/inspect")
async def inspect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_gate: float = Query(default=0.35, ge=0.0, le=1.0),
    vlm_backend: str = Query(default="gemini"),
    vlm_model: str | None = Query(default=None),
):
    """Async Detection + VLM reasoning endpoint. Returns a job_id.

    The detection, prompt, VLM parsing, commentary, and schema are shared with
    the live CLI runner. The provider is selected per request so experiments
    can compare backends without changing application logic.
    """
    raw_bytes = await file.read()
    image = _decode_upload(raw_bytes)

    job_id = str(uuid.uuid4())
    image_path = save_uploaded_image(raw_bytes, file.filename, job_id)
    _jobs[job_id] = {"status": "pending", "result": None, "error": None}
    
    background_tasks.add_task(
        process_inspection_job,
        job_id=job_id,
        image=image,
        filename=file.filename or "upload",
        image_path=image_path,
        vlm_backend_name=vlm_backend,
        vlm_model=vlm_model,
        confidence_gate=confidence_gate
    )
    
    return {"job_id": job_id, "status": "pending"}


@app.get("/inspect/status/{job_id}")
async def get_inspection_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Tâche introuvable")
    
    job = _jobs[job_id]
    if job["status"] == "completed":
        return {"status": "completed", "result": job["result"]}
    elif job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}
    else:
        return {"status": job["status"]}


@app.get("/reports")
async def reports(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: InspectionStatus | None = None,
    search: str | None = None,
) -> list[InspectionResult]:
    """Return real persisted inspections, newest first."""
    return list_reports(limit=limit, offset=offset, status=status, search=search)


@app.get("/reports/summary")
async def reports_summary() -> dict:
    """Return dashboard metrics calculated from persisted inspections only."""
    return get_report_summary()


@app.get("/reports/export")
async def reports_export() -> list[dict]:
    """Return all saved reports for a user-controlled local export."""
    return export_reports()


@app.get("/reports/{report_id}/farmer-report")
async def farmer_report_detail(report_id: str) -> dict:
    report = get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Rapport introuvable")
    return {"inspection": report.model_dump(mode="json"), "farmer_report": build_farmer_report(report)}


@app.get("/reports/{report_id}", response_model=InspectionResult)
async def report_detail(report_id: str) -> InspectionResult:
    report = get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Rapport introuvable")
    return report


@app.get("/model-info")
async def model_info() -> dict:
    """Read model details from the deployed detector and committed training run."""
    with TRAINING_ARGS_PATH.open(encoding="utf-8") as config_file:
        training_args = yaml.safe_load(config_file) or {}
    with TRAINING_RESULTS_PATH.open(encoding="utf-8", newline="") as results_file:
        final_metrics = list(csv.DictReader(results_file))[-1]
    model = get_yolo_model()
    return {
        "name": "Détecteur YOLO pour l’inspection alimentaire",
        "version": "lvis_fruits_yolo11m_80_v1",
        "architecture": str(training_args.get("model", "YOLO detector")),
        "num_classes": len(model.names),
        "class_names": [str(name) for _, name in sorted(model.names.items())],
        "input_size": f"{training_args.get('imgsz', 'unknown')} × {training_args.get('imgsz', 'unknown')}",
        "training_epochs": int(training_args.get("epochs", 0)),
        "map50": float(final_metrics.get("metrics/mAP50(B)", 0)) * 100,
        "precision": float(final_metrics.get("metrics/precision(B)", 0)) * 100,
        "recall": float(final_metrics.get("metrics/recall(B)", 0)) * 100,
        "vlm_backends": ["openrouter", "gpt4vlm", "gemini"],
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "database": "ready",
        "model_loaded": get_yolo_model.cache_info().currsize > 0,
    }
