/**
 * Custom hook for running inspections against the FastAPI backend.
<<<<<<< HEAD
 * Manages loading state, results, and error handling.
=======
 * Manages loading state, progress stages, results, and error handling.
 *
 * The /inspect endpoint is now async (returns a job_id immediately).
 * This hook polls /inspect/status/{job_id} until the job is complete.
>>>>>>> be35e77e0b3359cd9193412b00f8f0385cac407d
 */

import { useCallback, useState } from 'react';
import { detectImage, inspectImage } from '../api/inspectionApi';
import type { InspectionResult } from '../types/inspection';

type InspectionMode = 'detect' | 'inspect';

<<<<<<< HEAD
interface UseInspectionReturn {
  result: InspectionResult | null;
  isLoading: boolean;
=======
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
>>>>>>> be35e77e0b3359cd9193412b00f8f0385cac407d
  error: string | null;
  run: (file: File, mode?: InspectionMode, vlmBackend?: string) => Promise<void>;
  reset: () => void;
}

export function useInspection(): UseInspectionReturn {
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
<<<<<<< HEAD
=======
  const [stage, setStage] = useState<InspectionStage>(null);
>>>>>>> be35e77e0b3359cd9193412b00f8f0385cac407d
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (file: File, mode: InspectionMode = 'inspect', vlmBackend?: string) => {
      setIsLoading(true);
<<<<<<< HEAD
      setError(null);
      try {
        const res =
          mode === 'detect'
            ? await detectImage(file)
            : await inspectImage(file, {
                vlm_backend: vlmBackend as any,
                confidence_gate: 0.4,
              });
        setResult(res);
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Inspection failed';
=======
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
              vlm_backend: vlmBackend as any,
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
>>>>>>> be35e77e0b3359cd9193412b00f8f0385cac407d
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
<<<<<<< HEAD
  }, []);

  return { result, isLoading, error, run, reset };
}

=======
    setStage(null);
  }, []);

  return { result, isLoading, stage, error, run, reset };
}
>>>>>>> be35e77e0b3359cd9193412b00f8f0385cac407d
