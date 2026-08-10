"""
FastAPI surface for the inspection pipeline.

    POST /detect   -> YOLO detection only (synchronous)
    POST /inspect  -> Async YOLO + VLM reasoning (returns job_id)
    GET  /inspect/status/{job_id} -> Poll for results

Run with:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import os
import io
import uuid
import asyncio
import time
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional, Any
from dotenv import load_dotenv

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, BackgroundTasks
from ultralytics import YOLO

from inspection_pipeline import run_inspection
from schemas import InspectionResult
from vlm_reasoning import get_backend

load_dotenv()

app = FastAPI(title="Food Inspection API", version="0.3.0")

MODEL_PATH = "models/best.pt"
_frame_counter = 0

# In-memory job store
jobs: Dict[str, Dict[str, Any]] = {}

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
    print(f"\n[API] Received /detect request for file: {file.filename}")
    content = await file.read()
    print(f"[API] File size: {len(content)} bytes")
    image = _decode_upload(content)
    _frame_counter += 1
    
    print(f"[API] Running synchronous detection for frame {(_frame_counter)}...")
    result = run_inspection(
        image=image,
        yolo_model=get_yolo_model(),
        frame_id=_frame_counter,
        source=file.filename or "upload",
        vlm_backend=None,
    )
    print(f"[API] /detect finished. Found {result.num_detections} items.")
    return result


async def background_inspection_task(
    job_id: str,
    image: np.ndarray,
    frame_id: int,
    source: str,
    vlm_backend_name: str,
    confidence_gate: float
):
    """Background task to run the full inspection pipeline."""
    jobs[job_id]["status"] = "processing"
    print(f"[Job {job_id}] Started processing...")
    
    try:
        backend = get_vlm_backend(vlm_backend_name)
        result = run_inspection(
            image=image,
            yolo_model=get_yolo_model(),
            frame_id=frame_id,
            source=source,
            vlm_backend=backend,
            vlm_confidence_gate=confidence_gate,
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        print(f"[Job {job_id}] Completed successfully. Found {result.num_detections} items.")
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"[Job {job_id}] Failed: {e}")


@app.post("/inspect")
async def inspect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vlm_backend: str = Query(default="qwen", description="qwen | gpt4o | openrouter"),
    confidence_gate: float = Query(default=0.4, ge=0.0, le=1.0),
):
    """
    Detection + VLM reasoning endpoint. 
    Asynchronous: returns a job_id immediately.
    """
    global _frame_counter
    job_id = str(uuid.uuid4())
    print(f"\n[API] Received /inspect request for file: {file.filename}")
    print(f"[API] Job ID assigned: {job_id}")
    print(f"[API] VLM Backend: {vlm_backend}, Confidence Gate: {confidence_gate}")
    
    content = await file.read()
    print(f"[API] File size: {len(content)} bytes")
    image = _decode_upload(content)
    _frame_counter += 1
    
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "filename": file.filename
    }
    
    background_tasks.add_task(
        background_inspection_task,
        job_id,
        image,
        _frame_counter,
        file.filename or "upload",
        vlm_backend,
        confidence_gate
    )
    
    return {"job_id": job_id, "status": "pending"}


@app.get("/inspect/status/{job_id}")
async def get_inspect_status(job_id: str):
    """Poll for the status of an inspection job."""
    if job_id not in jobs:
        print(f"[API] Status check for unknown Job ID: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = jobs[job_id]
    print(f"[API] Status check for Job {job_id}: {job_info['status']}")
    return job_info


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
