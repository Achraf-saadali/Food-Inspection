# Root Cause Analysis and Architecture Fix

## 1. Root Cause of Timeout
The timeout issue occurs because the full inference pipeline (YOLO detection + VLM reasoning) takes longer than the frontend's hardcoded timeout limit, and the backend processes the request synchronously.

## 2. Exact File Causing It
- **Frontend Timeout Source**: `frontend/client/src/api/client.ts`
  The `apiClient` is configured with a timeout of `30_000` ms (30 seconds), but the user reported a timeout around `3000` ms (3 seconds).
- **Wait, why 3 seconds?**
  Let's re-examine `vite.config.ts` or the frontend server. If there's a reverse proxy (like Nginx) or a hosting platform (like Vercel/Netlify/Heroku serverless functions) involved, they often have strict timeouts (e.g., Vercel hobby plan is 10s, sometimes 3s for certain edge functions or proxies).
  However, locally, the Axios timeout is 30s. The user mentioned "request appears to timeout around 3000 ms". This could be a typo in their observation, or an implicit timeout in a browser/proxy.
- **Backend Bottleneck**: `backend/vlm_reasoning.py` and `backend/inspection_pipeline.py`
  The backend endpoint `/inspect` in `backend/api.py` synchronously calls `run_inspection`, which synchronously loops over YOLO detections and calls the VLM API (`vlm_backend.analyze`) for each detection sequentially. If an image has 3 items, it makes 3 sequential HTTP calls to an external VLM (like GPT-4o or Qwen API). Each VLM call takes ~1-3 seconds.
  Total latency = YOLO time + (N * VLM time). If N=3, latency easily exceeds 3-9 seconds.

## 3. Why It Happens
The `/inspect` endpoint is fully synchronous.
1. Frontend blocks waiting for the POST response.
2. Backend receives image, runs YOLO (fast, ~100ms).
3. Backend crops N detections.
4. For each crop, it calls a VLM API synchronously.
5. External VLM API latency is typically 1-3 seconds per call.
6. The total request time scales linearly with the number of detections.

## 4. Best Fix Architecture
**Option 1 (Increase Timeout)**: Simple, but doesn't fix the UX. The user is left staring at a spinning wheel for 10+ seconds if there are many detections. Also, it's vulnerable to platform-level timeouts if deployed.
**Option 2 (Async Jobs)**: Best for production. However, it requires a database (Redis/Postgres) and a task queue (Celery/RQ) to store job state, which adds significant architectural complexity to a simple FastAPI app.
**Option 3 (Background Tasks with Polling/Dict)**: A lightweight version of Option 2. We can use FastAPI's `BackgroundTasks` or a simple in-memory dictionary to store job status. The frontend polls for updates.

**Chosen Architecture: Lightweight Async Jobs with Polling**
Since this is a standalone FastAPI app without Redis/Celery, we will implement an in-memory job store.
1. **Backend**:
   - Add a global dictionary `jobs = {}`
   - Modify `/inspect` to generate a `job_id`, start a background thread/task to process the image, and immediately return `{"job_id": "..."}`.
   - Add a new endpoint `GET /inspect/status/{job_id}` that returns the current status (`pending`, `processing`, `completed`, `failed`) and the result if completed.
2. **Frontend**:
   - Modify `useInspection.ts` to handle the `job_id` response.
   - Implement a polling loop (e.g., every 1 second) to check `/inspect/status/{job_id}`.
   - Update the UI to show progress states ("Uploading...", "YOLO Detecting...", "VLM Reasoning...", "Complete").

*Wait*, before doing full async, the prompt says: "Choose the most appropriate solution for this project."
If I just use `asyncio.create_task` or FastAPI `BackgroundTasks`, it's very easy to implement without adding new dependencies.

Let's refine the plan:
1. **Add Timing Logs**: Update `backend/api.py` and `backend/inspection_pipeline.py` to log exact timings.
2. **Refactor Backend to Async**: Implement the lightweight job queue in `backend/api.py`.
3. **Refactor Frontend**: Update `inspectionApi.ts`, `useInspection.ts`, and `LiveInspection.tsx` to poll and show progress.
