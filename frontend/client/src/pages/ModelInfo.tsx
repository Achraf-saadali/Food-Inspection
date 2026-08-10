/**
 * Model Information page — YOLOv9c + VLM pipeline details.
 * Design: Industrial Precision — technical specs, pipeline diagram, class list
 */

import { useEffect, useState } from 'react';
import { Cpu, Layers, Target, Zap, Shield, Brain, ChevronRight } from 'lucide-react';
import { getModelInfo } from '../api/inspectionApi';
import type { ModelInfo } from '../types/inspection';
import { ConfidenceBar } from '../components/inspection/ConfidenceBar';

const FOOD_CLASSES = [
  'apple', 'artichoke', 'asparagus', 'avocado', 'banana', 'bean', 'beet',
  'bell pepper/capsicum', 'broccoli', 'brussels sprout', 'cabbage', 'carrot',
  'cauliflower', 'celery', 'cherry', 'chili pepper', 'coconut', 'corn',
  'cucumber', 'eggplant', 'fig', 'garlic', 'ginger', 'grape', 'grapefruit',
  'green bean', 'kale', 'kiwi', 'leek', 'lemon', 'lettuce', 'lime',
  'mango', 'melon', 'mushroom', 'onion', 'orange/orange fruit', 'papaya',
  'peach', 'pear', 'peas', 'pickle', 'pineapple', 'plum', 'pomegranate',
  'potato', 'pumpkin', 'radish', 'raspberry', 'spinach', 'squash',
  'strawberry', 'sweet potato', 'tomato', 'turnip', 'watermelon',
  'zucchini/courgette',
];

const PIPELINE_STAGES = [
  {
    step: '01', label: 'Image Input', icon: Layers,
    desc: 'JPEG/PNG image submitted via multipart/form-data upload or captured from webcam stream.',
    color: '#60a5fa',
  },
  {
    step: '02', label: 'YOLO Detection', icon: Target,
    desc: 'YOLOv9c runs inference at 640×640. Returns bounding boxes, class labels, and confidence scores for all detected food items.',
    color: '#f59e0b',
  },
  {
    step: '03', label: 'Crop & Profile', icon: Cpu,
    desc: 'Each detection above the confidence gate is cropped. A class-specific quality profile (e.g., ripeness + bruising for tomatoes) is retrieved.',
    color: '#a78bfa',
  },
  {
    step: '04', label: 'VLM Reasoning', icon: Brain,
    desc: 'The crop and a structured prompt are sent to the VLM (GPT-4o, Qwen-VL, or Gemini). Returns JSON with status, quality metrics, defects, and required action.',
    color: '#34d399',
  },
  {
    step: '05', label: 'Unified Output', icon: Shield,
    desc: 'Results are merged into an InspectionResult object containing all detections with full quality assessments. Logged to JSONL and returned via API.',
    color: '#22c55e',
  },
];

function SpecRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-border last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-mono text-foreground">{value}</span>
    </div>
  );
}

export default function ModelInfo() {
  const [info, setInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    getModelInfo().then(setInfo);
  }, []);

  if (!info) return (
    <div className="flex items-center justify-center h-64">
      <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  );

  return (
    <div className="page-enter p-6 space-y-6">
      {/* Header */}
      <div className="border-b border-border bg-card -mx-6 px-6 py-4 flex items-center amber-accent-line mb-0">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-primary" />
          <div>
            <h1 className="text-xl font-bold text-foreground font-mono tracking-tight">MODEL INFORMATION</h1>
            <p className="text-xs text-muted-foreground font-mono">TECHNICAL SPECS · AI PIPELINE ARCHITECTURE</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: specs */}
        <div className="space-y-4">
          {/* Model card */}
          <div className="rounded border border-border bg-card p-5">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded bg-primary/10 border border-primary/30 flex items-center justify-center">
                <Cpu className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-foreground" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{info.name}</h2>
                <p className="text-xs font-mono text-muted-foreground">v{info.version}</p>
              </div>
            </div>
            <SpecRow label="Architecture" value={info.architecture} />
            <SpecRow label="Input Resolution" value={info.input_size} />
            <SpecRow label="Classes" value={info.num_classes} />
            <SpecRow label="Training Epochs" value={info.training_epochs} />
          </div>

          {/* Performance metrics */}
          <div className="rounded border border-border bg-card p-5">
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">Performance Metrics</h3>
            <div className="space-y-3">
              <ConfidenceBar value={info.map50 / 100} label={`mAP@50: ${info.map50}%`} showPercent={false} />
              <ConfidenceBar value={info.precision / 100} label={`Precision: ${info.precision}%`} showPercent={false} />
              <ConfidenceBar value={info.recall / 100} label={`Recall: ${info.recall}%`} showPercent={false} />
            </div>
            <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
              Metrics reflect the 63-class food detection task. The model demonstrates strong localization capabilities on the LVIS Fruits & Vegetables dataset (~8,200 images).
            </p>
          </div>

          {/* VLM backends */}
          <div className="rounded border border-border bg-card p-5">
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">VLM Backends</h3>
            <div className="space-y-2">
              {info.vlm_backends.map((b) => (
                <div key={b} className="flex items-center gap-2 py-1.5">
                  <Brain className="w-3.5 h-3.5 text-primary shrink-0" />
                  <span className="text-xs font-mono text-foreground">{b}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: pipeline + classes */}
        <div className="lg:col-span-2 space-y-4">
          {/* Pipeline diagram */}
          <div className="rounded border border-border bg-card p-5">
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-5">AI Pipeline Architecture</h3>
            <div className="space-y-3">
              {PIPELINE_STAGES.map((stage, i) => (
                <div key={stage.step} className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <div
                      className="w-9 h-9 rounded flex items-center justify-center shrink-0 border"
                      style={{ backgroundColor: `${stage.color}15`, borderColor: `${stage.color}40` }}
                    >
                      <stage.icon className="w-4 h-4" style={{ color: stage.color }} />
                    </div>
                    {i < PIPELINE_STAGES.length - 1 && (
                      <div className="w-px flex-1 mt-1 mb-1" style={{ backgroundColor: `${stage.color}30`, minHeight: 16 }} />
                    )}
                  </div>
                  <div className="pb-4">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-mono text-muted-foreground">{stage.step}</span>
                      <h4 className="text-sm font-semibold text-foreground" style={{ fontFamily: "'Space Grotesk', sans-serif", color: stage.color }}>
                        {stage.label}
                      </h4>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{stage.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detected classes */}
          <div className="rounded border border-border bg-card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Detectable Classes ({FOOD_CLASSES.length})
              </h3>
              <span className="text-xs font-mono text-muted-foreground">YOLOv9c · LVIS dataset</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {FOOD_CLASSES.map((cls) => (
                <span
                  key={cls}
                  className="text-xs font-mono px-2 py-0.5 rounded border border-border bg-secondary text-muted-foreground hover:text-foreground hover:border-primary/40 transition-colors capitalize"
                >
                  {cls}
                </span>
              ))}
            </div>
          </div>

          {/* API reference */}
          <div className="rounded border border-border bg-card p-5">
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">API Reference</h3>
            <div className="space-y-3">
              {[
                { method: 'POST', path: '/detect', desc: 'YOLO detection only — fast, no VLM cost', color: '#60a5fa' },
                { method: 'POST', path: '/inspect', desc: 'Full pipeline: YOLO + VLM quality reasoning', color: '#22c55e' },
                { method: 'GET', path: '/health', desc: 'Backend health check', color: '#f59e0b' },
              ].map((ep) => (
                <div key={ep.path} className="flex items-start gap-3 p-3 rounded bg-secondary/50 border border-border">
                  <span
                    className="text-xs font-mono font-bold px-1.5 py-0.5 rounded shrink-0"
                    style={{ backgroundColor: `${ep.color}20`, color: ep.color }}
                  >
                    {ep.method}
                  </span>
                  <div>
                    <p className="text-xs font-mono text-foreground">{ep.path}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{ep.desc}</p>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-3 font-mono">
              Backend: <span className="text-primary">uvicorn api:app --host 0.0.0.0 --port 8000</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
