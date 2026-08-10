/**
 * Custom hook for running inspections against the FastAPI backend.
 * Manages loading state, progress stages, results, and error handling.
 *
 * The /inspect endpoint is now async (returns a job_id immediately).
 * This hook polls /inspect/status/{job_id} until the job is complete.
 */

import { useCallback, useState } from 'react';
import { detectImage, inspectImage } from '../api/inspectionApi';
import type { InspectionResult } from '../types/inspection';

type InspectionMode = 'detect' | 'inspect';

/**
 * Human-readable progress stages surfaced to the UI.
 * null means not running.
 */
export type InspectionStage =
  | null
  | 'Uploading image...'
  | 'YOLO detecting...'
  | 'Analyzing with VLM...'
  | 'Complete';

interface UseInspectionReturn {
  result: InspectionResult | null;
  isLoading: boolean;
  stage: InspectionStage;
  error: string | null;
  run: (file: File, mode?: InspectionMode) => Promise<void>;
  reset: () => void;
}

export function useInspection(): UseInspectionReturn {
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<InspectionStage>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (file: File, mode: InspectionMode = 'inspect') => {
      setIsLoading(true);
      setStage('Uploading image...');
      setError(null);
      try {
        if (mode === 'detect') {
          setStage('YOLO detecting...');
          const res = await detectImage(file);
          setResult(res);
        } else {
          const res = await inspectImage(
            file,
            {
              confidence_gate: 0.4,
            },
            (progressStage) => {
              setStage(progressStage as InspectionStage);
            }
          );
          setResult(res);
        }
        setStage('Complete');
      } catch (err) {
        let msg = 'Inspection failed';
        if (err instanceof Error) {
          msg = err.message;
        }
        // Provide more helpful messages for common failures
        if (msg.includes('timeout') || msg.includes('Timeout')) {
          msg = 'The inspection timed out. The VLM API may be slow or unreachable. Try again or switch to YOLO-only mode.';
        } else if (msg.includes('Network Error') || msg.includes('ECONNREFUSED')) {
          msg = 'Cannot connect to the backend. Make sure the FastAPI server is running at http://localhost:8000.';
        } else if (msg.includes('404')) {
          msg = 'Backend endpoint not found. Make sure you are running the latest version of api.py.';
        }
        setError(msg);
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setStage(null);
  }, []);

  return { result, isLoading, stage, error, run, reset };
}
