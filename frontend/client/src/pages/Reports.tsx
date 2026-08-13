/**
 * Reports page — real SQLite-backed inspection history and quality trends.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Clock,
  Download,
  Filter,
  HelpCircle,
  Layers,
  Search,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  downloadReportExport,
  getInspectionHistory,
  getInspectionStats,
} from '../api/inspectionApi';
import type { InspectionResult, InspectionStats, InspectionStatus } from '../types/inspection';
import { formatTimestamp } from '../utils/inspection';
import { StatusBadge } from '../components/inspection/StatusBadge';
import { DetectionCard } from '../components/inspection/DetectionCard';

type SortKey = 'timestamp' | 'frame_id' | 'num_detections';
type SortDir = 'asc' | 'desc';

function getOverallStatus(result: InspectionResult): InspectionStatus {
  if (result.items.some((item) => item.quality.status === 'defect')) return 'defect';
  if (result.items.some((item) => item.quality.status === 'uncertain')) return 'uncertain';
  if (result.items.length === 0 || result.items.every((item) => item.quality.status === 'skipped')) return 'skipped';
  return 'ok';
}

function recordKey(result: InspectionResult): string {
  return result.report_id || `${result.frame_id}-${result.timestamp}`;
}

export default function Reports() {
  const [history, setHistory] = useState<InspectionResult[]>([]);
  const [stats, setStats] = useState<InspectionStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<InspectionStatus | 'all'>('all');
  const [sortKey, setSortKey] = useState<SortKey>('timestamp');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [jsonView, setJsonView] = useState<string | null>(null);

  async function loadReports() {
    setLoading(true);
    setError(null);
    try {
      const [reports, summary] = await Promise.all([
        getInspectionHistory(500),
        getInspectionStats(),
      ]);
      setHistory(reports);
      setStats(summary);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Could not load saved inspections.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadReports();
  }, []);

  const filtered = useMemo(() => {
    let data = [...history];
    if (statusFilter !== 'all') {
      data = data.filter((result) => getOverallStatus(result) === statusFilter);
    }
    if (search.trim()) {
      const query = search.toLowerCase();
      data = data.filter((result) =>
        String(result.frame_id).includes(query)
        || result.source.toLowerCase().includes(query)
        || result.items.some((item) =>
          item.detection.label.toLowerCase().includes(query)
          || item.quality.defects.some((defect) => defect.toLowerCase().includes(query))
          || item.quality.commentary.toLowerCase().includes(query)
        )
      );
    }
    data.sort((left, right) => {
      const leftValue = sortKey === 'timestamp'
        ? new Date(left.timestamp).getTime()
        : left[sortKey];
      const rightValue = sortKey === 'timestamp'
        ? new Date(right.timestamp).getTime()
        : right[sortKey];
      return sortDir === 'asc' ? Number(leftValue) - Number(rightValue) : Number(rightValue) - Number(leftValue);
    });
    return data;
  }, [history, search, statusFilter, sortDir, sortKey]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(key);
      setSortDir('desc');
    }
  }

  function downloadJson(result: InspectionResult) {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `inspection-${recordKey(result)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function SortIcon({ sort }: { sort: SortKey }) {
    if (sortKey !== sort) return <ChevronDown className="w-3 h-3 opacity-30" />;
    return sortDir === 'asc'
      ? <ChevronUp className="w-3 h-3 text-primary" />
      : <ChevronDown className="w-3 h-3 text-primary" />;
  }

  const averageScore = stats?.avg_quality_score == null
    ? '—'
    : `${(stats.avg_quality_score * 100).toFixed(0)}%`;

  return (
    <div className="page-enter p-6 space-y-6">
      <div className="border-b border-border bg-card -mx-6 px-6 py-4 flex flex-wrap gap-4 items-center justify-between amber-accent-line">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-primary" />
          <div>
            <h1 className="text-xl font-bold text-foreground font-mono tracking-tight">INSPECTION REPORTS</h1>
            <p className="text-xs text-muted-foreground font-mono">SAVED RESULTS · QUALITY TRENDS · EXPORT</p>
          </div>
        </div>
        <button
          onClick={() => void downloadReportExport()}
          disabled={!stats?.total_inspections}
          className="inline-flex items-center gap-2 px-3 py-2 border border-primary/40 bg-primary/10 text-primary text-xs font-mono hover:bg-primary/20 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Download className="w-3.5 h-3.5" /> EXPORT SAVED REPORTS
        </button>
      </div>

      {error && (
        <div className="border border-[#ef444440] bg-[#ef444410] px-4 py-3 text-sm text-[#ef4444]">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="border border-border bg-card p-4">
          <p className="text-xs font-mono uppercase text-muted-foreground">Saved inspections</p>
          <p className="text-2xl mt-1 font-bold font-mono text-foreground">{stats?.total_inspections ?? 0}</p>
        </div>
        <div className="border border-[#22c55e35] bg-card p-4">
          <p className="text-xs font-mono uppercase text-muted-foreground">Accepted items</p>
          <p className="text-2xl mt-1 font-bold font-mono text-[#22c55e]">{stats?.ok_count ?? 0}</p>
        </div>
        <div className="border border-[#ef444435] bg-card p-4">
          <p className="text-xs font-mono uppercase text-muted-foreground">Defects found</p>
          <p className="text-2xl mt-1 font-bold font-mono text-[#ef4444]">{stats?.defect_count ?? 0}</p>
        </div>
        <div className="border border-[#f59e0b35] bg-card p-4">
          <p className="text-xs font-mono uppercase text-muted-foreground">Average quality score</p>
          <p className="text-2xl mt-1 font-bold font-mono text-primary">{averageScore}</p>
        </div>
      </div>

      {stats && stats.defect_breakdown.length > 0 && (
        <div className="border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-[#ef4444]" />
            <h2 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-widest">Most common detected issues</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {stats.defect_breakdown.map(({ defect, count }) => (
              <span key={defect} className="px-2.5 py-1 text-xs font-mono border border-[#ef444440] bg-[#ef444410] text-[#ef4444]">
                {defect.replace(/_/g, ' ')} · {count}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-52">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search filename, item, defect, or commentary..."
            className="w-full pl-9 pr-4 py-2 rounded border border-border bg-card text-sm text-foreground placeholder:text-muted-foreground font-mono focus:outline-none focus:border-primary/50"
          />
        </div>
        <div className="flex items-center gap-1 border border-border rounded bg-card p-1">
          <Filter className="w-3.5 h-3.5 text-muted-foreground ml-1.5" />
          {(['all', 'ok', 'defect', 'uncertain', 'skipped'] as const).map((status) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={cn(
                'px-2.5 py-1 rounded text-xs font-mono transition-colors',
                statusFilter === status ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {status.toUpperCase()}
            </button>
          ))}
        </div>
        <span className="text-xs font-mono text-muted-foreground">{filtered.length} saved records</span>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32">
          <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="rounded border border-border bg-card overflow-hidden">
          <div className="grid grid-cols-12 gap-2 px-4 py-2.5 border-b border-border bg-secondary/50">
            <button className="col-span-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('frame_id')}>
              Frame <SortIcon sort="frame_id" />
            </button>
            <button className="col-span-3 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('timestamp')}>
              Timestamp <SortIcon sort="timestamp" />
            </button>
            <button className="col-span-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('num_detections')}>
              Objects <SortIcon sort="num_detections" />
            </button>
            <div className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Classes</div>
            <div className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Status</div>
            <div className="col-span-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Actions</div>
          </div>

          <div className="divide-y divide-border">
            {filtered.length === 0 && (
              <div className="py-12 text-center space-y-2 text-sm text-muted-foreground">
                <Layers className="w-5 h-5 mx-auto opacity-50" />
                <p>No saved inspections match your filters.</p>
                <p className="text-xs">Run a full inspection to begin building report history.</p>
              </div>
            )}
            {filtered.map((result) => {
              const key = recordKey(result);
              const overall = getOverallStatus(result);
              const labels = Array.from(new Set(result.items.map((item) => item.detection.label))).slice(0, 3);
              const isExpanded = expanded === key;
              const isJsonView = jsonView === key;

              return (
                <div key={key}>
                  <div
                    className="grid grid-cols-12 gap-2 px-4 py-3 hover:bg-secondary/30 transition-colors cursor-pointer"
                    onClick={() => setExpanded(isExpanded ? null : key)}
                  >
                    <div className="col-span-2 flex items-center"><span className="text-sm font-mono text-foreground">#{result.frame_id}</span></div>
                    <div className="col-span-3 flex items-center gap-1.5"><Clock className="w-3 h-3 text-muted-foreground shrink-0" /><span className="text-xs font-mono text-muted-foreground truncate">{formatTimestamp(result.timestamp)}</span></div>
                    <div className="col-span-2 flex items-center gap-1.5"><Layers className="w-3 h-3 text-muted-foreground" /><span className="text-sm font-mono text-foreground">{result.num_detections}</span></div>
                    <div className="col-span-2 flex items-center flex-wrap gap-1">
                      {labels.map((label) => <span key={label} className="text-xs font-mono px-1.5 py-0.5 rounded bg-secondary text-muted-foreground capitalize truncate max-w-20">{label}</span>)}
                      {result.items.length > 3 && <span className="text-xs text-muted-foreground">+{result.items.length - 3}</span>}
                    </div>
                    <div className="col-span-2 flex items-center"><StatusBadge status={overall} size="sm" /></div>
                    <div className="col-span-1 flex items-center gap-1.5">
                      <button onClick={(event) => { event.stopPropagation(); setJsonView(isJsonView ? null : key); }} className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground text-xs font-mono" title="View JSON">{'{}'}</button>
                      <button onClick={(event) => { event.stopPropagation(); downloadJson(result); }} className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground" title="Download JSON"><Download className="w-3.5 h-3.5" /></button>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="px-4 pb-4 space-y-3 bg-secondary/20 border-t border-border">
                      <p className="text-xs text-muted-foreground uppercase tracking-wider pt-3">Farmer-ready inspection results</p>
                      {result.items.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No food items were detected in this image.</p>
                      ) : result.items.map((item, index) => <DetectionCard key={index} item={item} index={index} />)}
                    </div>
                  )}

                  {isJsonView && (
                    <div className="px-4 pb-4 bg-secondary/20 border-t border-border">
                      <pre className="text-xs font-mono text-foreground overflow-auto max-h-64 mt-3 p-3 rounded bg-background border border-border">{JSON.stringify(result, null, 2)}</pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
