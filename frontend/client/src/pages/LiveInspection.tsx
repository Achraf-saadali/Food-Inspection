/**
 * Live Inspection page — image upload, inference, bounding box overlay, results.
 * Design: Industrial Precision — scan-line animation, status-coded results
 *
 * Updated to show per-stage progress (Uploading → YOLO → VLM → Complete)
 * and to handle the new async job-based /inspect endpoint.
 */

import { useRef, useState, useCallback } from 'react';
import { Upload, ScanLine, X, Zap, Shield, AlertTriangle, Settings2, Loader2, CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useInspection, type InspectionStage } from '../hooks/useInspection';
import { DetectionCard } from '../components/inspection/DetectionCard';
import { BoundingBoxOverlay } from '../components/inspection/BoundingBoxOverlay';
import { StatusBadge } from '../components/inspection/StatusBadge';
import { formatTimestamp } from '../utils/inspection';
import type { InspectionStatus } from '../types/inspection';

type Mode = 'detect' | 'inspect';
type VlmBackend = 'qwen-api' | 'gpt4o' | 'openrouter';

// ─── Stage progress indicator ─────────────────────────────────────────────────

const DETECT_STAGES: InspectionStage[] = ['Uploading image...', 'YOLO detecting...', 'Complete'];
const INSPECT_STAGES: InspectionStage[] = [
  'Uploading image...',
  'YOLO detecting...',
  'Analyzing with VLM...',
  'Complete',
];

function StageProgressBar({ stage, mode }: { stage: InspectionStage; mode: Mode }) {
  const stages = mode === 'detect' ? DETECT_STAGES : INSPECT_STAGES;
  const currentIdx = stage ? stages.indexOf(stage) : -1;

  return (
    <div className="flex items-center gap-2 w-full">
      {stages.map((s, idx) => {
        const isDone = currentIdx > idx;
        const isActive = currentIdx === idx;
        return (
          <div key={s} className="flex-1 flex flex-col items-center gap-1">
            <div
              className={cn(
                'w-full h-1 rounded-full transition-all duration-500',
                isDone
                  ? 'bg-primary'
                  : isActive
                  ? 'bg-primary/60 animate-pulse'
                  : 'bg-border'
              )}
            />
            <span
              className={cn(
                'text-[10px] font-mono text-center leading-tight',
                isDone
                  ? 'text-primary'
                  : isActive
                  ? 'text-primary/80'
                  : 'text-muted-foreground/50'
              )}
            >
              {s}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function LiveInspection() {
  const { result, isLoading, stage, error, run, reset } = useInspection();
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [mode, setMode] = useState<Mode>('inspect');
  const [vlmBackend, setVlmBackend] = useState<VlmBackend>('qwen-api');
  const [isDragging, setIsDragging] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('image/')) return;
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      reset();
      await run(file, mode, mode === 'inspect' ? vlmBackend : undefined);
    },
    [mode, vlmBackend, run, reset]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleImgLoad = () => {
    if (imgRef.current) {
      setImgSize({ w: imgRef.current.clientWidth, h: imgRef.current.clientHeight });
    }
  };

  const overallStatus: InspectionStatus = !result
    ? 'skipped'
    : result.items.some((i) => i.quality.status === 'defect')
    ? 'defect'
    : result.items.some((i) => i.quality.status === 'uncertain')
    ? 'uncertain'
    : result.items.length === 0
    ? 'skipped'
    : 'ok';

  // Derive a short status label for the loading button
  const loadingLabel = stage && stage !== 'Complete' ? stage : 'Processing...';

  return (
    <div className="page-enter p-6 space-y-6">
      {/* Header */}
      <div className="border-b border-border bg-card px-0 py-4 flex items-center justify-between amber-accent-line -mx-6 px-6 mb-0">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-primary" />
          <div>
            <h1 className="text-xl font-bold text-foreground font-mono tracking-tight">LIVE INSPECTION</h1>
            <p className="text-xs text-muted-foreground font-mono">UPLOAD IMAGE · YOLO + VLM QUALITY REASONING</p>
          </div>
        </div>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 border text-xs font-mono transition-colors hmi-panel',
            showSettings ? 'border-primary/40 bg-primary/10 text-primary' : 'border-border bg-card text-muted-foreground hover:text-foreground'
          )}
        >
          <Settings2 className="w-3.5 h-3.5" />
          SETTINGS
        </button>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="rounded border border-border bg-card p-4 space-y-4">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Pipeline Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted-foreground mb-2 block">Inference Mode</label>
              <div className="flex gap-2">
                {(['detect', 'inspect'] as Mode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={cn(
                      'flex-1 py-2 px-3 rounded border text-xs font-mono transition-colors',
                      mode === m ? 'border-primary/40 bg-primary/10 text-primary' : 'border-border bg-secondary text-muted-foreground hover:text-foreground'
                    )}
                  >
                    {m === 'detect' ? '⚡ YOLO Only' : '🧠 YOLO + VLM'}
                  </button>
                ))}
              </div>
            </div>
            {mode === 'inspect' && (
              <div>
                <label className="text-xs text-muted-foreground mb-2 block">VLM Backend</label>
                <select
                  value={vlmBackend}
                  onChange={(e) => setVlmBackend(e.target.value as VlmBackend)}
                  className="w-full py-2 px-3 rounded border border-border bg-secondary text-xs font-mono text-foreground"
                >
                  <option value="qwen-api">Qwen-VL (DashScope)</option>
                  <option value="gpt4o">GPT-4o (OpenAI)</option>
                  <option value="openrouter">Gemini Flash (OpenRouter)</option>
                </select>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: image upload + overlay */}
        <div className="lg:col-span-3 space-y-4">
          {/* Drop zone */}
          <div
            className={cn(
              'relative rounded border-2 border-dashed transition-all duration-200 overflow-hidden',
              isDragging ? 'border-primary bg-primary/5' : 'border-border bg-card',
              imageUrl ? 'aspect-video' : 'aspect-video flex items-center justify-center'
            )}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => !imageUrl && fileInputRef.current?.click()}
          >
            {imageUrl ? (
              <>
                <img
                  ref={imgRef}
                  src={imageUrl}
                  alt="Inspection target"
                  className="w-full h-full object-contain"
                  onLoad={handleImgLoad}
                />
                {/* Scan line animation while loading */}
                {isLoading && (
                  <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <div
                      className="absolute left-0 right-0 h-0.5 bg-primary/70 scan-line"
                      style={{ boxShadow: '0 0 8px 2px rgba(245,158,11,0.5)' }}
                    />
                    <div className="absolute inset-0 bg-background/20" />
                  </div>
                )}
                {/* Bounding box overlay */}
                {result && imgSize.w > 0 && (
                  <BoundingBoxOverlay items={result.items} width={imgSize.w} height={imgSize.h} />
                )}
                {/* Clear button */}
                <button
                  onClick={(e) => { e.stopPropagation(); setImageUrl(null); reset(); }}
                  className="absolute top-2 right-2 w-7 h-7 rounded bg-background/80 border border-border flex items-center justify-center hover:bg-secondary transition-colors"
                >
                  <X className="w-3.5 h-3.5 text-muted-foreground" />
                </button>
              </>
            ) : (
              <div className="text-center space-y-3 p-8 cursor-pointer">
                <div className="w-14 h-14 rounded-full bg-secondary flex items-center justify-center mx-auto">
                  <Upload className="w-6 h-6 text-muted-foreground" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                    Drop image here or click to upload
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">JPEG, PNG, WebP supported</p>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 rounded border border-primary/40 bg-primary/10 text-primary text-xs font-mono hover:bg-primary/20 transition-colors"
                >
                  Browse Files
                </button>
              </div>
            )}
          </div>
          <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />

          {/* Stage progress bar — shown while loading */}
          {isLoading && stage && (
            <div className="rounded border border-border bg-card p-3">
              <StageProgressBar stage={stage} mode={mode} />
            </div>
          )}

          {/* Run button */}
          {imageUrl && (
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="w-full py-2.5 rounded border border-primary/40 bg-primary/10 text-primary text-sm font-mono hover:bg-primary/20 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {loadingLabel}
                </>
              ) : stage === 'Complete' ? (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  Upload New Image
                </>
              ) : (
                <>
                  <ScanLine className="w-4 h-4" />
                  Upload New Image
                </>
              )}
            </button>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-start gap-3 p-4 rounded border border-[#ef444440] bg-[#ef444410]">
              <AlertTriangle className="w-4 h-4 text-[#ef4444] shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-[#ef4444]">Inspection Failed</p>
                <p className="text-xs text-muted-foreground mt-1 font-mono">{error}</p>
                {error.includes('connect') && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Ensure the FastAPI backend is running at{' '}
                    <code className="font-mono text-primary">http://localhost:8000</code>
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right: results panel */}
        <div className="lg:col-span-2 space-y-4">
          {/* Result summary */}
          {result ? (
            <div className="rounded border border-border bg-card p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Inspection Result</h3>
                <StatusBadge status={overallStatus} size="md" pulse={overallStatus === 'defect'} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-muted-foreground">Frame ID</p>
                  <p className="text-sm font-mono text-foreground">#{result.frame_id}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Detections</p>
                  <p className="text-sm font-mono text-foreground">{result.num_detections}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Image Size</p>
                  <p className="text-sm font-mono text-foreground">{result.image_size.width}×{result.image_size.height}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Timestamp</p>
                  <p className="text-xs font-mono text-foreground">{formatTimestamp(result.timestamp)}</p>
                </div>
              </div>
              {/* Mode indicator */}
              <div className="flex items-center gap-2 pt-1 border-t border-border">
                {mode === 'inspect' ? (
                  <><Shield className="w-3.5 h-3.5 text-primary" /><span className="text-xs font-mono text-muted-foreground">YOLO + VLM · {vlmBackend}</span></>
                ) : (
                  <><Zap className="w-3.5 h-3.5 text-primary" /><span className="text-xs font-mono text-muted-foreground">YOLO Detection Only</span></>
                )}
              </div>
            </div>
          ) : (
            <div className="rounded border border-border bg-card p-6 text-center">
              <ScanLine className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">
                {isLoading
                  ? stage || 'Analyzing image...'
                  : 'Upload an image to begin inspection'}
              </p>
              {isLoading && stage && (
                <p className="text-xs text-muted-foreground/60 mt-1 font-mono">
                  {stage === 'Analyzing with VLM...'
                    ? 'VLM reasoning may take 5–15 seconds per detection...'
                    : stage === 'YOLO detecting...'
                    ? 'Running YOLO object detection...'
                    : ''}
                </p>
              )}
            </div>
          )}

          {/* Detection items */}
          {result && result.items.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Detected Items ({result.items.length})
              </h3>
              {result.items.map((item, i) => (
                <DetectionCard key={i} item={item} index={i} />
              ))}
            </div>
          )}

          {result && result.items.length === 0 && (
            <div className="rounded border border-border bg-card p-6 text-center">
              <p className="text-sm text-muted-foreground">No objects detected in this image.</p>
              <p className="text-xs text-muted-foreground mt-1 font-mono">
                Try lowering the confidence threshold or use a clearer image.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
