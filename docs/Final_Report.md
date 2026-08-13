# Food Inspection Timeout Debugging Report

**Author:** Manus AI  
**Date:** August 10, 2026  

## 1. Root Cause of the Timeout

The timeout issue observed during the `/inspect` request was not caused by a frontend syntax error, but rather by the **synchronous nature of the backend inference pipeline** coupled with the latency of external Vision Language Model (VLM) API calls. 

When a user triggered an inspection, the following sequence occurred:
1. The frontend sent a synchronous HTTP POST request to `/inspect`.
2. The backend received the image and ran YOLO detection (which is fast, typically ~100ms).
3. The backend then sequentially iterated over every detected object crop.
4. For each crop, it made a synchronous network call to the configured VLM API (e.g., Qwen or GPT-4o) to assess quality.
5. A single VLM API call typically takes between 1 and 3 seconds. Therefore, an image with 3 or 4 detections could easily cause the total backend processing time to exceed 10–12 seconds.

While the frontend Axios client was technically configured with a 30-second timeout (`30_000` ms) in `frontend/client/src/api/client.ts`, the user's report of a timeout "around 3000 ms" suggests either a platform-level proxy timeout (common in serverless or containerized environments) or browser-level request stalling. Regardless of the exact timeout threshold, the root cause was the architectural decision to block the HTTP request while waiting for multiple slow, external VLM inferences.

## 2. Exact Files Causing the Bottleneck

The bottleneck was located in two primary files:
- **`backend/api.py`**: The `/inspect` endpoint was defined as a standard synchronous endpoint that blocked until `run_inspection` completed.
- **`backend/inspection_pipeline.py`**: The `run_inspection` function sequentially looped over `yolo_results.boxes` and synchronously called `vlm_backend.analyze(crop, label, confidence)`.

## 3. Why It Happened

The original architecture was designed for simplicity, treating the full pipeline (YOLO + VLM) as a single blocking operation. This works well for a local CLI or a single-item test, but it scales linearly with the number of detected items. Because VLM APIs are inherently slow due to the massive compute required for multi-modal reasoning, stacking these calls synchronously within a single HTTP request cycle inevitably leads to timeout failures when multiple objects are detected.

## 4. The Architectural Fix

To resolve this without adding heavy external dependencies (like Redis or Celery), I implemented a **Lightweight Async Job Queue with Polling**:

1. **Backend Asynchronous Processing (`backend/api.py`)**:
   - I introduced an in-memory job store (`_jobs` dictionary) to track the state of inspections.
   - The `/inspect` endpoint was refactored to use FastAPI's `BackgroundTasks`. It now immediately returns a unique `job_id` and a `pending` status, while the heavy YOLO and VLM inference runs in the background.
   - A new endpoint, `GET /inspect/status/{job_id}`, was added to allow the frontend to poll for the result.
   - I also added comprehensive timing logs using `time.perf_counter()` to track exact YOLO inference times, VLM latencies, and total job durations.

2. **Frontend Polling and UX Improvements (`frontend/client/src/api/inspectionApi.ts` & `LiveInspection.tsx`)**:
   - The Axios client timeout was safely increased to 120 seconds for the polling cycle.
   - The `inspectImage` API function was rewritten to submit the job and then poll the `/inspect/status/{job_id}` endpoint every 1 second until completion.
   - The `useInspection` hook and the `LiveInspection.tsx` component were updated to surface real-time progress stages to the user:
     - `Uploading image...`
     - `YOLO detecting...`
     - `Analyzing with VLM...` (This sets user expectations for the slower reasoning phase)
     - `Complete`
   - Error handling was improved to provide specific, actionable messages rather than generic timeout errors.

## Conclusion

The repository has been successfully updated and pushed to the `main` branch. The frontend now smoothly handles long-running inspections by polling the backend, providing a vastly improved user experience with clear progress indicators, while the backend processes VLM requests asynchronously without holding HTTP connections open.
