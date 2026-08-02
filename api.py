"""
FastAPI surface for the inspection pipeline.

    POST /detect   -> YOLO detection only (matches current live_inference.py output)
    POST /inspect  -> YOLO detection + VLM quality reasoning (unified JSON)

Run with:
    uvicorn api:app --reload --port 8000

This replaces the current local cv2.imshow loop with a request/response
interface, per README Section 17 ("Integration Notes").
"""

from __future__ import annotations

import os
import io
from functools import lru_cache
from dotenv import load_dotenv

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from ultralytics import YOLO

from inspection_pipeline import run_inspection
from schemas import InspectionResult
from vlm_reasoning import get_backend

load_dotenv()

app = FastAPI(title="Food Inspection API", version="0.2.0")

MODEL_PATH = "models/best.pt"
_frame_counter = 0


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


@app.post("/inspect", response_model=InspectionResult)
async def inspect(
    file: UploadFile = File(...),
    vlm_backend: str = Query(default="qwen", description="qwen | gpt4o"),
    confidence_gate: float = Query(default=0.4, ge=0.0, le=1.0),
) -> InspectionResult:
    """Detection + VLM reasoning endpoint. Returns the unified inspection JSON."""
    global _frame_counter
    image = _decode_upload(await file.read())
    _frame_counter += 1
    backend = get_vlm_backend(vlm_backend)
    return run_inspection(
        image=image,
        yolo_model=get_yolo_model(),
        frame_id=_frame_counter,
        source=file.filename or "upload",
        vlm_backend=backend,
        vlm_confidence_gate=confidence_gate,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
