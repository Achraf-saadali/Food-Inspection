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
  | 'Téléversement de l’image…'
  | 'Détection YOLO…'
  | 'Analyse avec le VLM…'
  | 'Terminé';

interface UseInspectionReturn {
  result: InspectionResult | null;
  isLoading: boolean;
  stage: InspectionStage;
  error: string | null;
  run: (file: File, mode?: InspectionMode, signal?: AbortSignal) => Promise<InspectionResult | null>;
  reset: () => void;
}

export function useInspection(): UseInspectionReturn {
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<InspectionStage>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (file: File, mode: InspectionMode = 'inspect', signal?: AbortSignal) => {
      setIsLoading(true);
      setStage('Téléversement de l’image…');
      setError(null);
      try {
        let completedResult: InspectionResult;
        if (mode === 'detect') {
          setStage('Détection YOLO…');
          const res = await detectImage(file, signal);
          setResult(res);
          completedResult = res;
        } else {
          const res = await inspectImage(
            file,
            {
              confidence_gate: 0.35,
              signal,
            },
            (progressStage) => {
              setStage(progressStage as InspectionStage);
            }
          );
          setResult(res);
          completedResult = res;
        }
        setStage('Terminé');
        return completedResult;
      } catch (err) {
        if (signal?.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
          return null;
        }
        let msg = 'L’inspection a échoué.';
        if (err instanceof Error) {
          msg = err.message;
        }
        // Provide more helpful messages for common failures
        if (msg.includes('timeout') || msg.includes('Timeout')) {
          msg = 'L’inspection a dépassé le délai. L’API VLM est peut-être lente ou inaccessible. Réessayez ou utilisez le mode YOLO seul.';
        } else if (msg.includes('Network Error') || msg.includes('ECONNREFUSED')) {
          msg = 'Impossible de joindre le serveur. Vérifiez que FastAPI fonctionne sur http://localhost:8000.';
        } else if (msg.includes('404')) {
          msg = 'Point d’accès serveur introuvable. Lancez uvicorn backend.api:app depuis la racine du dépôt.';
        }
        setError(msg);
        return null;
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
