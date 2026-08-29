import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Camera, Loader2, ScanLine, Settings2, Square, Upload, Video, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { detectImage } from '../api/inspectionApi';
import { useInspection, type InspectionStage } from '../hooks/useInspection';
import { BoundingBoxOverlay } from '../components/inspection/BoundingBoxOverlay';
import { DetectionCard } from '../components/inspection/DetectionCard';
import { StatusBadge } from '../components/inspection/StatusBadge';
import { formatTimestamp } from '../utils/inspection';
import type { InspectionItem, InspectionResult, InspectionStatus } from '../types/inspection';

type Mode = 'detect' | 'inspect';
type SourceMode = 'camera' | 'upload';
type Target = { id: string; item: InspectionItem };
const LIVE_INTERVAL_MS = Number(import.meta.env.VITE_LIVE_INSPECTION_INTERVAL_MS || 750);

function iou(a: [number, number, number, number], b: [number, number, number, number]) {
  const left = Math.max(a[0], b[0]); const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]); const bottom = Math.min(a[3], b[3]);
  const overlap = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return overlap / Math.max(areaA + areaB - overlap, 1);
}

function centerDistance(a: [number, number, number, number], b: [number, number, number, number]) {
  const distance = Math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2);
  return distance / Math.max((Math.hypot(a[2] - a[0], a[3] - a[1]) + Math.hypot(b[2] - b[0], b[3] - b[1])) / 2, 1);
}

function wait(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { window.clearTimeout(timer); reject(new DOMException('Cancelled', 'AbortError')); }, { once: true });
  });
}

function Progress({ stage, mode }: { stage: InspectionStage; mode: Mode }) {
  const stages = mode === 'inspect' ? ['Uploading image...', 'YOLO detecting...', 'Analyzing with VLM...', 'Complete'] : ['Uploading image...', 'YOLO detecting...', 'Complete'];
  return <div className="flex gap-2">{stages.map((item) => <div key={item} className={cn('flex-1 text-center text-[10px] font-mono border-t-2 pt-1', stage === item ? 'border-primary text-primary' : 'border-border text-muted-foreground')}>{item}</div>)}</div>;
}

export default function LiveInspection() {
  const { result, isLoading, stage, error, run, reset } = useInspection();
  const [source, setSource] = useState<SourceMode>('camera'); const [mode, setMode] = useState<Mode>('inspect');
  const [cameraActive, setCameraActive] = useState(false); const [running, setRunning] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null); const [showSettings, setShowSettings] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null); const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 }); const [liveItems, setLiveItems] = useState<InspectionItem[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null); const canvasRef = useRef<HTMLCanvasElement>(null); const streamRef = useRef<MediaStream | null>(null);
  const fileRef = useRef<HTMLInputElement>(null); const imageRef = useRef<HTMLImageElement>(null); const frameRef = useRef<HTMLDivElement>(null);
  const runningRef = useRef(false); const abortRef = useRef<AbortController | null>(null); const targetsRef = useRef<Target[]>([]);

  const stopInspection = useCallback(() => { runningRef.current = false; abortRef.current?.abort(); abortRef.current = null; setRunning(false); }, []);
  const stopCamera = useCallback(() => { stopInspection(); streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null; if (videoRef.current) videoRef.current.srcObject = null; setCameraActive(false); setLiveItems([]); }, [stopInspection]);

  const matchTargets = useCallback((items: InspectionItem[], keepQuality: boolean) => {
    const used = new Set<string>(); const next: Target[] = [];
    items.forEach((item) => {
      const candidates = targetsRef.current.filter((target) => target.item.detection.label === item.detection.label && !used.has(target.id)).map((target) => ({ target, overlap: iou(target.item.detection.bbox_xyxy, item.detection.bbox_xyxy), distance: centerDistance(target.item.detection.bbox_xyxy, item.detection.bbox_xyxy) })).filter((candidate) => candidate.overlap >= 0.05 || candidate.distance <= 1.5).sort((a, b) => b.overlap - a.overlap);
      const match = candidates[0]?.target; const id = match?.id || `${item.detection.label}-${targetsRef.current.length + next.length + 1}`;
      used.add(id); next.push({ id, item: keepQuality && match ? { ...item, quality: match.item.quality } : item });
    });
    targetsRef.current = next; return next.map((target) => target.item);
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const constraints = { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
      let stream: MediaStream; try { stream = await navigator.mediaDevices.getUserMedia(constraints); } catch { stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); }
      streamRef.current = stream; if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); } setCameraActive(true);
    } catch (err) { setCameraError(err instanceof DOMException && err.name === 'NotAllowedError' ? 'Camera permission denied. Please allow camera access in your browser settings.' : 'Camera unavailable. Check that a camera is connected.'); }
  }, []);

  const capture = () => { const video = videoRef.current; const canvas = canvasRef.current; if (!video || !canvas || !video.videoWidth || !video.videoHeight) return null; canvas.width = video.videoWidth; canvas.height = video.videoHeight; canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height); return new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.8)); };

  const startInspection = useCallback(async () => {
    if (!cameraActive || runningRef.current) return; runningRef.current = true; targetsRef.current = []; setLiveItems([]); setRunning(true); const controller = new AbortController(); abortRef.current = controller;
    try { while (runningRef.current && !controller.signal.aborted) {
      const blob = await capture(); if (!blob) { await wait(250, controller.signal); continue; }
      const frame = new File([blob], `camera-frame-${Date.now()}.jpg`, { type: 'image/jpeg' });
      if (mode === 'inspect') { const detected = await detectImage(frame, controller.signal); setFrameSize(detected.image_size); setLiveItems(matchTargets(detected.items, true)); }
      const completed = await run(frame, mode, controller.signal); if (!completed) break;
      if (mode === 'inspect') { setFrameSize(completed.image_size); setLiveItems(matchTargets(completed.items, false)); } else setLiveItems(completed.items);
      await wait(LIVE_INTERVAL_MS, controller.signal);
    } } catch (err) { if (!(err instanceof DOMException && err.name === 'AbortError')) setCameraError('Live inspection stopped unexpectedly.'); }
    finally { runningRef.current = false; abortRef.current = null; setRunning(false); }
  }, [cameraActive, mode, matchTargets, run]);

  const handleFile = useCallback(async (file: File) => { if (!file.type.startsWith('image/')) return; stopInspection(); if (imageUrl) URL.revokeObjectURL(imageUrl); setImageUrl(URL.createObjectURL(file)); reset(); await run(file, mode); }, [imageUrl, mode, reset, run, stopInspection]);
  useEffect(() => () => { stopCamera(); }, [stopCamera]);
  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl); }, [imageUrl]);
  useEffect(() => { if (!frameRef.current) return; const observer = new ResizeObserver(([entry]) => setFrameSize((size) => ({ ...size, width: entry.contentRect.width, height: entry.contentRect.height }))); observer.observe(frameRef.current); return () => observer.disconnect(); }, [source]);
  const overall: InspectionStatus = !result ? 'skipped' : result.items.some((item) => item.quality.status === 'defect') ? 'defect' : result.items.some((item) => item.quality.status === 'uncertain') ? 'uncertain' : result.items.length ? 'ok' : 'skipped';

  return <div className="page-enter p-6 space-y-6"><header className="border-b border-border bg-card py-4 flex justify-between amber-accent-line"><div className="flex gap-3"><div className="w-1 h-6 bg-primary" /><div><h1 className="text-xl font-bold font-mono">LIVE INSPECTION</h1><p className="text-xs text-muted-foreground font-mono">CAMERA · YOLO + VLM QUALITY REASONING</p></div></div><button onClick={() => setShowSettings(!showSettings)} className="border border-border px-3 py-1.5 text-xs font-mono flex gap-2 items-center"><Settings2 className="w-3.5 h-3.5" /> SETTINGS</button></header>
    {showSettings && <div className="border border-border bg-card p-4"><label className="text-xs text-muted-foreground">Inference mode</label><div className="flex gap-2 mt-2">{(['detect', 'inspect'] as Mode[]).map((item) => <button key={item} onClick={() => setMode(item)} className={cn('border px-3 py-2 text-xs font-mono', mode === item ? 'border-primary text-primary' : 'border-border text-muted-foreground')}>{item === 'detect' ? 'YOLO Only' : 'YOLO + VLM'}</button>)}</div></div>}
    <div className="flex gap-2 border-b border-border"><button onClick={() => { stopInspection(); setSource('camera'); }} className={cn('px-4 py-2 text-xs font-mono border-b-2 flex gap-2', source === 'camera' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground')}><Camera className="w-3.5 h-3.5" /> CAMERA</button><button onClick={() => { stopInspection(); setSource('upload'); }} className={cn('px-4 py-2 text-xs font-mono border-b-2 flex gap-2', source === 'upload' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground')}><Upload className="w-3.5 h-3.5" /> UPLOAD IMAGE</button></div>
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6"><main className="lg:col-span-3 space-y-4">{source === 'camera' ? <><div ref={frameRef} className="relative aspect-video bg-black border border-border overflow-hidden"><video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-contain" />{!cameraActive && <div className="absolute inset-0 flex flex-col items-center justify-center gap-3"><Video className="w-10 h-10 text-muted-foreground" /><button onClick={startCamera} className="border border-primary/40 bg-primary/10 px-4 py-2 text-primary text-xs font-mono">START CAMERA</button></div>}{liveItems.length > 0 && frameSize.width > 0 && <BoundingBoxOverlay items={liveItems} width={frameSize.width} height={frameSize.height} sourceWidth={frameSize.width} sourceHeight={frameSize.height} />}{isLoading && <div className="absolute top-0 left-0 right-0 h-0.5 bg-primary scan-line" />}</div><canvas ref={canvasRef} className="hidden" /><div className="flex flex-wrap gap-2"><button onClick={startCamera} disabled={cameraActive} className="border border-primary/40 px-3 py-2 text-primary text-xs font-mono"><Camera className="w-3.5 h-3.5 inline mr-2" />START CAMERA</button><button onClick={startInspection} disabled={!cameraActive || running} className="border border-primary/40 px-3 py-2 text-primary text-xs font-mono"><ScanLine className="w-3.5 h-3.5 inline mr-2" />START INSPECTION</button><button onClick={stopInspection} disabled={!running} className="border border-border px-3 py-2 text-muted-foreground text-xs font-mono"><Square className="w-3.5 h-3.5 inline mr-2" />STOP INSPECTION</button><button onClick={stopCamera} disabled={!cameraActive} className="border border-red-400/40 px-3 py-2 text-red-400 text-xs font-mono"><X className="w-3.5 h-3.5 inline mr-2" />STOP CAMERA</button></div></> : <><div ref={frameRef} className="relative aspect-video border-2 border-dashed border-border overflow-hidden" onDrop={(event) => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (file) void handleFile(file); }} onDragOver={(event) => event.preventDefault()}>{imageUrl ? <><img ref={imageRef} src={imageUrl} alt="Inspection target" className="w-full h-full object-contain" onLoad={() => imageRef.current && setImageSize({ width: imageRef.current.naturalWidth, height: imageRef.current.naturalHeight })} />{result && <BoundingBoxOverlay items={result.items} width={frameSize.width} height={frameSize.height} sourceWidth={imageSize.width} sourceHeight={imageSize.height} />}<button onClick={() => { URL.revokeObjectURL(imageUrl); setImageUrl(null); reset(); }} className="absolute top-2 right-2 border border-border bg-background p-2"><X className="w-3.5 h-3.5" /></button></> : <button onClick={() => fileRef.current?.click()} className="absolute inset-0 text-sm text-muted-foreground">Drop image here or click to upload</button>}</div><input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void handleFile(file); }} /><button onClick={() => fileRef.current?.click()} disabled={isLoading} className="w-full border border-primary/40 py-2.5 text-primary text-sm font-mono">{isLoading ? <><Loader2 className="w-4 h-4 inline mr-2 animate-spin" />{stage}</> : 'UPLOAD NEW IMAGE'}</button></>}{cameraError && <div className="border border-red-400/40 bg-red-400/10 p-4 text-xs text-red-400"><AlertTriangle className="w-4 h-4 inline mr-2" />{cameraError}</div>}{error && <div className="border border-red-400/40 bg-red-400/10 p-4 text-xs text-red-400"><AlertTriangle className="w-4 h-4 inline mr-2" />{error}</div>}{isLoading && stage && <div className="border border-border bg-card p-3"><Progress stage={stage} mode={mode} /></div>}</main>
      <aside className="lg:col-span-2 space-y-4">{result ? <div className="border border-border bg-card p-4"><div className="flex justify-between"><h3 className="text-xs font-semibold text-muted-foreground uppercase">Inspection Result</h3><StatusBadge status={overall} size="md" /></div><div className="grid grid-cols-2 gap-3 mt-3 text-xs"><div>Frame <strong className="font-mono">#{result.frame_id}</strong></div><div>Detections <strong className="font-mono">{result.num_detections}</strong></div><div>Size <strong className="font-mono">{result.image_size.width}×{result.image_size.height}</strong></div><div>Time <strong className="font-mono">{formatTimestamp(result.timestamp)}</strong></div></div></div> : <div className="border border-border bg-card p-6 text-center text-sm text-muted-foreground">Start an inspection to see results.</div>}{result?.items.map((item, index) => <DetectionCard key={`${result.frame_id}-${index}`} item={item} index={index} />)}</aside></div>
  </div>;
}