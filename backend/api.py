"""
FastAPI surface for the inspection pipeline.

    POST /detect   -> YOLO detection only (matches current live_inference.py output)
    POST /inspect  -> YOLO detection + VLM quality reasoning (unified JSON)

Run with:
    uvicorn backend.api:app --reload --port 8000

This replaces the current local cv2.imshow loop with a request/response
interface, per README Section 17 ("Integration Notes").
"""

from __future__ import annotations

import os
import io
import uuid
import asyncio
from typing import Dict, Any
from functools import lru_cache
from dotenv import load_dotenv

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from backend.inspection_pipeline import run_inspection
from backend.schemas import InspectionResult
from backend.vlm_reasoning import get_backend

load_dotenv()

app = FastAPI(title="Food Inspection API", version="0.4.0")

# Allow the Vite dev server (port 3000) and any production frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/best.pt"
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
        raise HTTPException(status_code=400, detail="Could not decode image")
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


def process_inspection_job(job_id: str, image: np.ndarray, filename: str, vlm_backend_name: str, confidence_gate: float):
    global _frame_counter
    import time
    
    _jobs[job_id]["status"] = "processing"
    
    try:
        start_time = time.perf_counter()
        _frame_counter += 1
        backend = get_vlm_backend(vlm_backend_name)
        
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
    confidence_gate: float = Query(default=0.4, ge=0.0, le=1.0),
):
    """Async Detection + VLM reasoning endpoint. Returns a job_id.
    Note: The VLM backend is now hardcoded to OpenRouter in vlm_reasoning.py.
    """
    image = _decode_upload(await file.read())
    
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None, "error": None}
    
    background_tasks.add_task(
        process_inspection_job,
        job_id=job_id,
        image=image,
        filename=file.filename,
        vlm_backend_name="openrouter",
        confidence_gate=confidence_gate
    )
    
    return {"job_id": job_id, "status": "pending"}


@app.get("/inspect/status/{job_id}")
async def get_inspection_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    if job["status"] == "completed":
        return {"status": "completed", "result": job["result"]}
    elif job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}
    else:
        return {"status": job["status"]}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
