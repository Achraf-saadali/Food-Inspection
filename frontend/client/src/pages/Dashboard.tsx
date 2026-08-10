/**
 * Dashboard page — overview of the inspection system.
 * Design: Industrial Precision — HMI control panel aesthetic, status-first hierarchy
 * Status colors dominate; amber schematic accents; instrument-grade charts
 */

import { useEffect, useState } from 'react';
import { Link } from 'wouter';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from 'recharts';
import type { PieLabelRenderProps } from 'recharts';
import {
  ScanLine, CheckCircle, AlertTriangle, HelpCircle, Activity,
  TrendingUp, Clock, Layers, ArrowRight, Wifi, WifiOff,
} from 'lucide-react';
import { getInspectionStats, getInspectionHistory, checkHealth } from '../api/inspectionApi';
import type { InspectionStats, InspectionResult, HealthStatus } from '../types/inspection';
import { formatTimestamp, getStatusColor } from '../utils/inspection';
import { StatusBadge } from '../components/inspection/StatusBadge';

const STATUS_COLORS = {
  ok: '#22c55e',
  defect: '#ef4444',
  uncertain: '#f59e0b',
  skipped: '#6b7280',
};

function MetricCard({
  label, value, sub, icon: Icon, color, href, large,
}: {
  label: string; value: string | number; sub?: string;
  icon: React.ElementType; color: string; href?: string; large?: boolean;
}) {
  const content = (
    <div
      className="relative overflow-hidden border bg-card hover:bg-secondary/30 transition-colors group amber-accent-line hmi-panel"
      style={{ borderColor: `${color}35`, borderLeftColor: color, borderLeftWidth: 3 }}
    >
      {/* Background grid */}
      <div className="absolute inset-0 hmi-grid opacity-30 pointer-events-none" />
      <div className="relative p-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-widest mb-2 font-mono">{label}</p>
            <p className={`font-bold text-foreground font-mono ${large ? 'text-4xl' : 'text-3xl'}`} style={{ color }}>
              {value}
            </p>
            {sub && <p className="text-xs text-muted-foreground mt-1.5 font-mono">{sub}</p>}
          </div>
          <div className="w-11 h-11 flex items-center justify-center border" style={{ backgroundColor: `${color}12`, borderColor: `${color}30` }}>
            <Icon className="w-5 h-5" style={{ color }} />
          </div>
        </div>
        {href && (
          <div className="flex items-center gap-1 mt-4 text-xs font-mono" style={{ color }}>
            <span>VIEW DETAILS</span>
            <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
          </div>
        )}
      </div>
    </div>
  );
  return href ? <Link href={href}>{content}</Link> : content;
}

export default function Dashboard() {
  const [stats, setStats] = useState<InspectionStats | null>(null);
  const [history, setHistory] = useState<InspectionResult[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getInspectionStats(), getInspectionHistory(10), checkHealth()]).then(
      ([s, h, hlt]) => {
        setStats(s);
        setHistory(h.slice(0, 8));
        setHealth(hlt);
        setLoading(false);
      }
    );
  }, []);

  if (loading || !stats) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center space-y-3">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-xs text-muted-foreground font-mono tracking-widest">LOADING SYSTEM DATA...</p>
        </div>
      </div>
    );
  }

  const pieData = [
    { name: 'PASS', value: stats.ok_count, color: STATUS_COLORS.ok },
    { name: 'DEFECT', value: stats.defect_count, color: STATUS_COLORS.defect },
    { name: 'UNCERTAIN', value: stats.uncertain_count, color: STATUS_COLORS.uncertain },
    { name: 'SKIPPED', value: stats.skipped_count, color: STATUS_COLORS.skipped },
  ].filter((d) => d.value > 0);

  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: PieLabelRenderProps) => {
    if (typeof percent !== 'number' || percent < 0.06) return null;
    const RADIAN = Math.PI / 180;
    const radius = (Number(innerRadius) + Number(outerRadius)) / 2;
    const x = Number(cx) + radius * Math.cos(-midAngle * RADIAN);
    const y = Number(cy) + radius * Math.sin(-midAngle * RADIAN);
    return (
      <text x={x} y={y} fill="#000" textAnchor="middle" dominantBaseline="central" fontSize={10} fontFamily="'JetBrains Mono', monospace" fontWeight="700">
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  const defectRate = stats.total_detections > 0
    ? ((stats.defect_count / stats.total_detections) * 100).toFixed(1)
    : '0.0';
  const passRate = stats.total_detections > 0
    ? ((stats.ok_count / stats.total_detections) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="page-enter">
      {/* Header bar — HMI style */}
      <div className="border-b border-border bg-card px-6 py-4 flex items-center justify-between amber-accent-line">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-1 h-6 bg-primary" />
            <h1 className="text-xl font-bold text-foreground font-mono tracking-tight">
              INSPECTION OVERVIEW
            </h1>
          </div>
          <p className="text-xs text-muted-foreground mt-1 font-mono ml-4">
            YOLOv9c · 63 CLASSES · VLM QUALITY REASONING
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 border hmi-panel" style={{ borderColor: health?.status === 'ok' ? '#22c55e40' : '#ef444440', backgroundColor: health?.status === 'ok' ? '#22c55e10' : '#ef444410' }}>
            {health?.status === 'ok'
              ? <Wifi className="w-3.5 h-3.5 text-[#22c55e]" />
              : <WifiOff className="w-3.5 h-3.5 text-[#ef4444]" />
            }
            <span className="text-xs font-mono" style={{ color: health?.status === 'ok' ? '#22c55e' : '#ef4444' }}>
              {health?.status === 'ok' ? 'API ONLINE' : 'API OFFLINE'}
            </span>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* Primary status metrics — status-first, dominant */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <MetricCard
            label="PASS" value={stats.ok_count}
            sub={`${passRate}% pass rate`}
            icon={CheckCircle} color={STATUS_COLORS.ok} href="/reports" large
          />
          <MetricCard
            label="DEFECT" value={stats.defect_count}
            sub={`${defectRate}% defect rate`}
            icon={AlertTriangle} color={STATUS_COLORS.defect} href="/reports" large
          />
          <MetricCard
            label="UNCERTAIN" value={stats.uncertain_count}
            sub="requires review"
            icon={HelpCircle} color={STATUS_COLORS.uncertain}
          />
          <MetricCard
            label="TOTAL SCANS" value={stats.total_inspections}
            sub={`${stats.total_detections} objects · avg ${(stats.avg_confidence * 100).toFixed(1)}% conf`}
            icon={ScanLine} color="#f59e0b"
          />
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          {/* Status distribution */}
          <div className="border bg-card hmi-panel amber-accent-line relative overflow-hidden" style={{ borderColor: 'var(--border)' }}>
            <div className="absolute inset-0 hmi-grid opacity-20 pointer-events-none" />
            <div className="relative p-5">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-1 h-4 bg-primary" />
                <h2 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-widest">
                  STATUS DISTRIBUTION
                </h2>
              </div>
              <ResponsiveContainer width="100%" height={190}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%" cy="50%"
                    innerRadius={52} outerRadius={82}
                    paddingAngle={2}
                    dataKey="value"
                    labelLine={false}
                    label={renderCustomLabel}
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} stroke={entry.color} strokeWidth={1} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f1117', border: '1px solid #2a2f3e', borderRadius: 2, fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}
                    itemStyle={{ color: '#94a3b8' }}
                  />
                  <Legend
                    iconType="square" iconSize={8}
                    wrapperStyle={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", paddingTop: 8 }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Top classes bar chart */}
          <div className="lg:col-span-2 border bg-card hmi-panel amber-accent-line relative overflow-hidden" style={{ borderColor: 'var(--border)' }}>
            <div className="absolute inset-0 hmi-grid opacity-20 pointer-events-none" />
            <div className="relative p-5">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-1 h-4 bg-primary" />
                <h2 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-widest">
                  TOP DETECTED CLASSES
                </h2>
              </div>
              <ResponsiveContainer width="100%" height={190}>
                <BarChart data={stats.top_classes} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
                  <XAxis
                    type="number"
                    tick={{ fontSize: 10, fill: '#4b5563', fontFamily: "'JetBrains Mono', monospace" }}
                    axisLine={{ stroke: '#2a2f3e' }} tickLine={false}
                    domain={[0, 'dataMax + 1']}
                  />
                  <YAxis
                    type="category" dataKey="label" width={120}
                    tick={{ fontSize: 10, fill: '#9ca3af', fontFamily: "'JetBrains Mono', monospace" }}
                    axisLine={false} tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f1117', border: '1px solid #2a2f3e', borderRadius: 2, fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}
                    cursor={{ fill: '#f59e0b0a' }}
                    formatter={(v) => [`${v} detections`, 'Count']}
                  />
                  <Bar dataKey="count" radius={[0, 2, 2, 0]} maxBarSize={16}>
                    {stats.top_classes.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? '#f59e0b' : i < 3 ? '#d97706' : '#92400e'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Recent inspections */}
        <div className="border bg-card hmi-panel amber-accent-line relative overflow-hidden" style={{ borderColor: 'var(--border)' }}>
          <div className="absolute inset-0 hmi-grid opacity-10 pointer-events-none" />
          <div className="relative">
            <div className="flex items-center justify-between px-5 py-3.5 border-b border-border">
              <div className="flex items-center gap-2">
                <div className="w-1 h-4 bg-primary" />
                <h2 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-widest">
                  RECENT INSPECTIONS
                </h2>
              </div>
              <Link href="/reports">
                <span className="text-xs text-primary hover:underline flex items-center gap-1 font-mono">
                  VIEW ALL <ArrowRight className="w-3 h-3" />
                </span>
              </Link>
            </div>
            <div className="divide-y divide-border">
              {history.map((result) => {
                const hasDefect = result.items.some((i) => i.quality.status === 'defect');
                const overallStatus = hasDefect ? 'defect' : result.items.some((i) => i.quality.status === 'uncertain') ? 'uncertain' : result.items.length === 0 ? 'skipped' : 'ok';
                const statusColor = getStatusColor(overallStatus);
                return (
                  <div
                    key={result.frame_id}
                    className="flex items-center gap-4 px-5 py-3 hover:bg-secondary/30 transition-colors"
                    style={{ borderLeftColor: statusColor, borderLeftWidth: 2 }}
                  >
                    <div className="w-7 h-7 flex items-center justify-center bg-secondary shrink-0" style={{ borderRadius: 2 }}>
                      <Activity className="w-3.5 h-3.5 text-muted-foreground" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-mono text-foreground">FRAME #{result.frame_id}</span>
                        <span className="text-xs text-muted-foreground font-mono">·</span>
                        <span className="text-xs text-muted-foreground font-mono">{result.num_detections} OBJ</span>
                      </div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <Clock className="w-3 h-3 text-muted-foreground" />
                        <span className="text-xs font-mono text-muted-foreground">{formatTimestamp(result.timestamp)}</span>
                      </div>
                    </div>
                    <StatusBadge status={overallStatus} size="sm" pulse={overallStatus === 'defect'} />
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Quick actions */}
        <div className="grid grid-cols-2 gap-3">
          <Link href="/inspection">
            <div className="border hmi-panel p-5 hover:bg-primary/10 transition-colors cursor-pointer group relative overflow-hidden" style={{ borderColor: '#f59e0b40', backgroundColor: '#f59e0b08' }}>
              <div className="absolute inset-0 hmi-grid opacity-20 pointer-events-none" />
              <div className="relative flex items-center gap-3">
                <ScanLine className="w-6 h-6 text-primary" />
                <div>
                  <p className="text-sm font-bold text-foreground font-mono">RUN INSPECTION</p>
                  <p className="text-xs text-muted-foreground font-mono mt-0.5">Upload image · YOLO + VLM</p>
                </div>
                <ArrowRight className="w-4 h-4 text-primary ml-auto group-hover:translate-x-0.5 transition-transform" />
              </div>
            </div>
          </Link>
          <Link href="/reports">
            <div className="border hmi-panel p-5 hover:bg-secondary/30 transition-colors cursor-pointer group relative overflow-hidden" style={{ borderColor: 'var(--border)' }}>
              <div className="absolute inset-0 hmi-grid opacity-20 pointer-events-none" />
              <div className="relative flex items-center gap-3">
                <TrendingUp className="w-6 h-6 text-muted-foreground" />
                <div>
                  <p className="text-sm font-bold text-foreground font-mono">VIEW REPORTS</p>
                  <p className="text-xs text-muted-foreground font-mono mt-0.5">Browse inspection history</p>
                </div>
                <ArrowRight className="w-4 h-4 text-muted-foreground ml-auto group-hover:translate-x-0.5 transition-transform" />
              </div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
