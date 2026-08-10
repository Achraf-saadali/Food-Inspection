/**
 * Reports page — inspection history with search/filter.
 * Design: Industrial Precision — table + filter bar, JSON viewer
 */

import { useEffect, useState, useMemo } from 'react';
import { Search, Filter, Download, ChevronDown, ChevronUp, Clock, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getInspectionHistory } from '../api/inspectionApi';
import type { InspectionResult, InspectionStatus } from '../types/inspection';
import { formatTimestamp, getStatusColor } from '../utils/inspection';
import { StatusBadge } from '../components/inspection/StatusBadge';
import { DetectionCard } from '../components/inspection/DetectionCard';

type SortKey = 'timestamp' | 'frame_id' | 'num_detections';
type SortDir = 'asc' | 'desc';

export default function Reports() {
  const [history, setHistory] = useState<InspectionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<InspectionStatus | 'all'>('all');
  const [sortKey, setSortKey] = useState<SortKey>('timestamp');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [expanded, setExpanded] = useState<number | null>(null);
  const [jsonView, setJsonView] = useState<number | null>(null);

  useEffect(() => {
    getInspectionHistory(50).then((h) => {
      setHistory(h);
      setLoading(false);
    });
  }, []);

  function getOverallStatus(result: InspectionResult): InspectionStatus {
    if (result.items.some((i) => i.quality.status === 'defect')) return 'defect';
    if (result.items.some((i) => i.quality.status === 'uncertain')) return 'uncertain';
    if (result.items.length === 0) return 'skipped';
    return 'ok';
  }

  const filtered = useMemo(() => {
    let data = [...history];
    if (statusFilter !== 'all') {
      data = data.filter((r) => getOverallStatus(r) === statusFilter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      data = data.filter(
        (r) =>
          String(r.frame_id).includes(q) ||
          r.items.some((i) => i.detection.label.toLowerCase().includes(q))
      );
    }
    data.sort((a, b) => {
      let av: number, bv: number;
      if (sortKey === 'timestamp') {
        av = new Date(a.timestamp).getTime();
        bv = new Date(b.timestamp).getTime();
      } else {
        av = a[sortKey] as number;
        bv = b[sortKey] as number;
      }
      return sortDir === 'asc' ? av - bv : bv - av;
    });
    return data;
  }, [history, search, statusFilter, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  }

  function SortIcon({ k }: { k: SortKey }) {
    if (sortKey !== k) return <ChevronDown className="w-3 h-3 opacity-30" />;
    return sortDir === 'asc' ? <ChevronUp className="w-3 h-3 text-primary" /> : <ChevronDown className="w-3 h-3 text-primary" />;
  }

  function downloadJson(result: InspectionResult) {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inspection_frame_${result.frame_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="page-enter p-6 space-y-6">
      {/* Header */}
      <div className="border-b border-border bg-card -mx-6 px-6 py-4 flex items-center amber-accent-line mb-0">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-primary" />
          <div>
            <h1 className="text-xl font-bold text-foreground font-mono tracking-tight">INSPECTION REPORTS</h1>
            <p className="text-xs text-muted-foreground font-mono">BROWSE · SEARCH · EXPORT HISTORY</p>
          </div>
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by frame ID or class..."
            className="w-full pl-9 pr-4 py-2 rounded border border-border bg-card text-sm text-foreground placeholder:text-muted-foreground font-mono focus:outline-none focus:border-primary/50"
          />
        </div>
        <div className="flex items-center gap-1 border border-border rounded bg-card p-1">
          <Filter className="w-3.5 h-3.5 text-muted-foreground ml-1.5" />
          {(['all', 'ok', 'defect', 'uncertain', 'skipped'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={cn(
                'px-2.5 py-1 rounded text-xs font-mono transition-colors',
                statusFilter === s ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {s.toUpperCase()}
            </button>
          ))}
        </div>
        <span className="text-xs font-mono text-muted-foreground">{filtered.length} records</span>
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex items-center justify-center h-32">
          <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="rounded border border-border bg-card overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-12 gap-2 px-4 py-2.5 border-b border-border bg-secondary/50">
            <button className="col-span-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('frame_id')}>
              Frame <SortIcon k="frame_id" />
            </button>
            <button className="col-span-3 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('timestamp')}>
              Timestamp <SortIcon k="timestamp" />
            </button>
            <button className="col-span-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('num_detections')}>
              Objects <SortIcon k="num_detections" />
            </button>
            <div className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Classes</div>
            <div className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Status</div>
            <div className="col-span-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Actions</div>
          </div>

          {/* Rows */}
          <div className="divide-y divide-border">
            {filtered.length === 0 && (
              <div className="py-12 text-center text-sm text-muted-foreground">No records match your filters.</div>
            )}
            {filtered.map((result) => {
              const overall = getOverallStatus(result);
                    const labels = Array.from(new Set(result.items.map((i) => i.detection.label))).slice(0, 3);
              const isExpanded = expanded === result.frame_id;
              const isJsonView = jsonView === result.frame_id;

              return (
                <div key={result.frame_id}>
                  <div
                    className="grid grid-cols-12 gap-2 px-4 py-3 hover:bg-secondary/30 transition-colors cursor-pointer"
                    onClick={() => setExpanded(isExpanded ? null : result.frame_id)}
                  >
                    <div className="col-span-2 flex items-center">
                      <span className="text-sm font-mono text-foreground">#{result.frame_id}</span>
                    </div>
                    <div className="col-span-3 flex items-center gap-1.5">
                      <Clock className="w-3 h-3 text-muted-foreground shrink-0" />
                      <span className="text-xs font-mono text-muted-foreground truncate">{formatTimestamp(result.timestamp)}</span>
                    </div>
                    <div className="col-span-2 flex items-center gap-1.5">
                      <Layers className="w-3 h-3 text-muted-foreground" />
                      <span className="text-sm font-mono text-foreground">{result.num_detections}</span>
                    </div>
                    <div className="col-span-2 flex items-center flex-wrap gap-1">
                      {labels.map((l) => (
                        <span key={l} className="text-xs font-mono px-1.5 py-0.5 rounded bg-secondary text-muted-foreground capitalize truncate max-w-20">{l}</span>
                      ))}
                      {result.items.length > 3 && <span className="text-xs text-muted-foreground">+{result.items.length - 3}</span>}
                    </div>
                    <div className="col-span-2 flex items-center">
                      <StatusBadge status={overall} size="sm" />
                    </div>
                    <div className="col-span-1 flex items-center gap-1.5">
                      <button
                        onClick={(e) => { e.stopPropagation(); setJsonView(isJsonView ? null : result.frame_id); }}
                        className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors text-xs font-mono"
                        title="View JSON"
                      >
                        {'{}'}
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); downloadJson(result); }}
                        className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                        title="Download JSON"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Expanded detection cards */}
                  {isExpanded && result.items.length > 0 && (
                    <div className="px-4 pb-4 space-y-2 bg-secondary/20 border-t border-border">
                      <p className="text-xs text-muted-foreground uppercase tracking-wider pt-3 mb-2">Detection Details</p>
                      {result.items.map((item, i) => (
                        <DetectionCard key={i} item={item} index={i} />
                      ))}
                    </div>
                  )}

                  {/* JSON view */}
                  {isJsonView && (
                    <div className="px-4 pb-4 bg-secondary/20 border-t border-border">
                      <pre className="text-xs font-mono text-foreground overflow-auto max-h-64 mt-3 p-3 rounded bg-background border border-border">
                        {JSON.stringify(result, null, 2)}
                      </pre>
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
