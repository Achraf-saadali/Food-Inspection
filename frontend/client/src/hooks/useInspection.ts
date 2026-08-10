/**
 * Custom hook for running inspections against the FastAPI backend.
 * Manages loading state, results, and error handling.
 */

import { useCallback, useState } from 'react';
import { detectImage, inspectImage } from '../api/inspectionApi';
import type { InspectionResult } from '../types/inspection';

type InspectionMode = 'detect' | 'inspect';

interface UseInspectionReturn {
  result: InspectionResult | null;
  isLoading: boolean;
  error: string | null;
  run: (file: File, mode?: InspectionMode, vlmBackend?: string) => Promise<void>;
  reset: () => void;
}

export function useInspection(): UseInspectionReturn {
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (file: File, mode: InspectionMode = 'inspect', vlmBackend?: string) => {
      setIsLoading(true);
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
  }, []);

  return { result, isLoading, error, run, reset };
}

