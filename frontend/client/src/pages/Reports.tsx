/**
 * Farmer-facing inspection reports.
 * Technical inspection details remain available below the actionable summary.
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
  Layers,
  Search,
  ShieldAlert,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  downloadReportExport,
  getFarmerReport,
  getInspectionHistory,
  getInspectionStats,
} from '../api/inspectionApi';
import type {
  FarmerReport,
  InspectionResult,
  InspectionStats,
  InspectionStatus,
} from '../types/inspection';
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

function decisionStyles(severity: FarmerReport['decision']['severity']): string {
  if (severity === 'success') return 'border-[#22c55e55] bg-[#22c55e10] text-[#22c55e]';
  if (severity === 'attention') return 'border-[#ef444455] bg-[#ef444410] text-[#ef4444]';
  if (severity === 'warning') return 'border-[#f59e0b55] bg-[#f59e0b10] text-[#f59e0b]';
  return 'border-[#60a5fa55] bg-[#60a5fa10] text-[#60a5fa]';
}

function objectStyles(result: string): string {
  if (result === 'Acceptable') return 'border-[#22c55e55] bg-[#22c55e10] text-[#22c55e]';
  if (result === 'À isoler') return 'border-[#ef444455] bg-[#ef444410] text-[#ef4444]';
  return 'border-[#f59e0b55] bg-[#f59e0b10] text-[#f59e0b]';
}

function FarmerReportView({ report }: { report: FarmerReport }) {
  const { decision, summary } = report;

  return (
    <div className="space-y-4">
      <div className={cn('border px-4 py-3', decisionStyles(decision.severity))}>
        <div className="flex items-start gap-3">
          {decision.severity === 'success'
            ? <CheckCircle className="w-5 h-5 mt-0.5 shrink-0" />
            : <ShieldAlert className="w-5 h-5 mt-0.5 shrink-0" />}
          <div>
            <p className="text-xs uppercase tracking-widest font-mono opacity-80">Résultat du contrôle</p>
            <p className="text-lg font-semibold mt-1">{decision.label}</p>
            <p className="text-sm mt-1 text-foreground">{decision.explanation}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        {[
          ['Détectés', summary.detected, 'text-foreground'],
          ['Acceptables', summary.acceptable, 'text-[#22c55e]'],
          ['À isoler', summary.to_isolate, 'text-[#ef4444]'],
          ['À vérifier', summary.to_review, 'text-[#f59e0b]'],
          ['Non évalués', summary.not_assessed, 'text-muted-foreground'],
        ].map(([label, value, color]) => (
          <div key={String(label)} className="border border-border bg-card px-3 py-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
            <p className={cn('text-xl font-bold font-mono mt-1', color)}>{value}</p>
          </div>
        ))}
      </div>

      {report.actions.length > 0 && (
        <div className="border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-primary" />
            <h3 className="text-xs uppercase tracking-widest font-mono text-muted-foreground">Actions recommandées</h3>
          </div>
          <div className="space-y-2">
            {report.actions.map((action) => (
              <div key={`${action.priority}-${action.type}`} className="flex items-start gap-3 text-sm">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-primary/15 text-primary text-xs font-mono shrink-0">{action.priority}</span>
                <span className="text-foreground">{action.label}</span>
                {action.object_ids.length > 0 && (
                  <span className="text-xs text-muted-foreground font-mono">Objets : {action.object_ids.join(', ')}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {report.objects.length > 0 && (
        <div className="border border-border bg-card p-4">
          <h3 className="text-xs uppercase tracking-widest font-mono text-muted-foreground mb-3">Résultat par produit</h3>
          <div className="space-y-2">
            {report.objects.map((object) => (
              <div key={object.object_id} className="border border-border/70 px-3 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-muted-foreground">#{object.object_id}</span>
                    <span className="font-medium text-foreground">{object.object_type}</span>
                    <span className={cn('px-2 py-0.5 border text-[11px] font-mono', objectStyles(object.result))}>{object.result}</span>
                  </div>
                  <span className="text-xs text-muted-foreground">Fiabilité : {object.reliability}</span>
                </div>
                <p className="text-sm text-foreground mt-2">{object.explanation}</p>
                <p className="text-xs text-primary mt-1">Action : {object.action}</p>
                {object.problem.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {object.problem.map((problem) => (
                      <span key={problem} className="px-2 py-0.5 text-[11px] border border-[#ef444455] bg-[#ef444410] text-[#ef4444]">{problem}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {report.warnings.length > 0 && (
        <div className="border border-[#f59e0b40] bg-[#f59e0b08] px-4 py-3">
          <p className="text-xs uppercase tracking-widest text-[#f59e0b] font-mono mb-2">À garder en tête</p>
          <ul className="space-y-1">
            {report.warnings.map((warning) => <li key={warning} className="text-xs text-muted-foreground">• {warning}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function Reports() {
  const [history, setHistory] = useState<InspectionResult[]>([]);
  const [stats, setStats] = useState<InspectionStats | null>(null);
  const [farmerReports, setFarmerReports] = useState<Record<string, FarmerReport>>({});
  const [farmerLoading, setFarmerLoading] = useState<Record<string, boolean>>({});
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
      const [reports, summary] = await Promise.all([getInspectionHistory(500), getInspectionStats()]);
      setHistory(reports);
      setStats(summary);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Impossible de charger les inspections enregistrées.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void loadReports(); }, []);

  async function loadFarmerReport(result: InspectionResult) {
    const key = recordKey(result);
    if (farmerReports[key] || farmerLoading[key] || !result.report_id) return;
    setFarmerLoading((current) => ({ ...current, [key]: true }));
    try {
      const response = await getFarmerReport(result.report_id);
      setFarmerReports((current) => ({ ...current, [key]: response.farmer_report }));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Impossible de charger le rapport farmer.');
    } finally {
      setFarmerLoading((current) => ({ ...current, [key]: false }));
    }
  }

  const filtered = useMemo(() => {
    let data = [...history];
    if (statusFilter !== 'all') data = data.filter((result) => getOverallStatus(result) === statusFilter);
    if (search.trim()) {
      const query = search.toLowerCase();
      data = data.filter((result) => String(result.frame_id).includes(query)
        || result.source.toLowerCase().includes(query)
        || result.items.some((item) => item.detection.label.toLowerCase().includes(query)
          || item.quality.defects.some((defect) => defect.toLowerCase().includes(query))
          || item.quality.commentary.toLowerCase().includes(query)));
    }
    data.sort((left, right) => {
      const leftValue = sortKey === 'timestamp' ? new Date(left.timestamp).getTime() : left[sortKey];
      const rightValue = sortKey === 'timestamp' ? new Date(right.timestamp).getTime() : right[sortKey];
      return sortDir === 'asc' ? Number(leftValue) - Number(rightValue) : Number(rightValue) - Number(leftValue);
    });
    return data;
  }, [history, search, statusFilter, sortDir, sortKey]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
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
    return sortDir === 'asc' ? <ChevronUp className="w-3 h-3 text-primary" /> : <ChevronDown className="w-3 h-3 text-primary" />;
  }

  const averageScore = stats?.avg_quality_score == null ? '—' : `${(stats.avg_quality_score * 100).toFixed(0)}%`;

  return (
    <div className="page-enter p-6 space-y-6">
      <div className="border-b border-border bg-card -mx-6 px-6 py-4 flex flex-wrap gap-4 items-center justify-between amber-accent-line">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-primary" />
          <div>
            <h1 className="text-xl font-bold text-foreground tracking-tight">RAPPORTS D’INSPECTION</h1>
            <p className="text-xs text-muted-foreground">DÉCISION DU LOT · ACTIONS · HISTORIQUE</p>
          </div>
        </div>
        <button onClick={() => void downloadReportExport()} disabled={!stats?.total_inspections} className="inline-flex items-center gap-2 px-3 py-2 border border-primary/40 bg-primary/10 text-primary text-xs hover:bg-primary/20 disabled:opacity-40 disabled:cursor-not-allowed">
          <Download className="w-3.5 h-3.5" /> EXPORTER LES RAPPORTS
        </button>
      </div>

      {error && <div className="border border-[#ef444440] bg-[#ef444410] px-4 py-3 text-sm text-[#ef4444]">{error}</div>}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="border border-border bg-card p-4"><p className="text-xs uppercase text-muted-foreground">Inspections enregistrées</p><p className="text-2xl mt-1 font-bold text-foreground">{stats?.total_inspections ?? 0}</p></div>
        <div className="border border-[#22c55e35] bg-card p-4"><p className="text-xs uppercase text-muted-foreground">Produits acceptables</p><p className="text-2xl mt-1 font-bold text-[#22c55e]">{stats?.ok_count ?? 0}</p></div>
        <div className="border border-[#ef444435] bg-card p-4"><p className="text-xs uppercase text-muted-foreground">Défauts trouvés</p><p className="text-2xl mt-1 font-bold text-[#ef4444]">{stats?.defect_count ?? 0}</p></div>
        <div className="border border-[#f59e0b35] bg-card p-4"><p className="text-xs uppercase text-muted-foreground">Score qualité moyen</p><p className="text-2xl mt-1 font-bold text-primary">{averageScore}</p></div>
      </div>

      {stats && stats.defect_breakdown.length > 0 && (
        <div className="border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-3"><AlertTriangle className="w-4 h-4 text-[#ef4444]" /><h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">Problèmes les plus fréquents</h2></div>
          <div className="flex flex-wrap gap-2">{stats.defect_breakdown.map(({ defect, count }) => <span key={defect} className="px-2.5 py-1 text-xs border border-[#ef444440] bg-[#ef444410] text-[#ef4444]">{defect.replace(/_/g, ' ')} · {count}</span>)}</div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-52"><Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Rechercher un fichier, produit ou défaut..." className="w-full pl-9 pr-4 py-2 border border-border bg-card text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50" /></div>
        <div className="flex items-center gap-1 border border-border bg-card p-1"><Filter className="w-3.5 h-3.5 text-muted-foreground ml-1.5" />{(['all', 'ok', 'defect', 'uncertain', 'skipped'] as const).map((status) => <button key={status} onClick={() => setStatusFilter(status)} className={cn('px-2.5 py-1 text-xs transition-colors', statusFilter === status ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground')}>{status === 'all' ? 'TOUS' : status.toUpperCase()}</button>)}</div>
        <span className="text-xs text-muted-foreground">{filtered.length} rapport(s)</span>
      </div>

      {loading ? <div className="flex items-center justify-center h-32"><div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" /></div> : (
        <div className="border border-border bg-card overflow-hidden">
          <div className="grid grid-cols-12 gap-2 px-4 py-2.5 border-b border-border bg-secondary/50">
            <button className="col-span-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('frame_id')}>Frame <SortIcon sort="frame_id" /></button>
            <button className="col-span-3 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('timestamp')}>Date <SortIcon sort="timestamp" /></button>
            <button className="col-span-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground" onClick={() => toggleSort('num_detections')}>Produits <SortIcon sort="num_detections" /></button>
            <div className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Types</div><div className="col-span-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">État</div><div className="col-span-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Actions</div>
          </div>
          <div className="divide-y divide-border">
            {filtered.length === 0 && <div className="py-12 text-center space-y-2 text-sm text-muted-foreground"><Layers className="w-5 h-5 mx-auto opacity-50" /><p>Aucun rapport ne correspond aux filtres.</p><p className="text-xs">Lancez une inspection complète pour créer un historique.</p></div>}
            {filtered.map((result) => {
              const key = recordKey(result);
              const overall = getOverallStatus(result);
              const labels = Array.from(new Set(result.items.map((item) => item.detection.label))).slice(0, 3);
              const isExpanded = expanded === key;
              const isJsonView = jsonView === key;
              const farmerReport = farmerReports[key];
              return (
                <div key={key}>
                  <div className="grid grid-cols-12 gap-2 px-4 py-3 hover:bg-secondary/30 transition-colors cursor-pointer" onClick={() => { setExpanded(isExpanded ? null : key); if (!isExpanded) void loadFarmerReport(result); }}>
                    <div className="col-span-2 flex items-center"><span className="text-sm font-mono text-foreground">#{result.frame_id}</span></div>
                    <div className="col-span-3 flex items-center gap-1.5"><Clock className="w-3 h-3 text-muted-foreground shrink-0" /><span className="text-xs text-muted-foreground truncate">{formatTimestamp(result.timestamp)}</span></div>
                    <div className="col-span-2 flex items-center gap-1.5"><Layers className="w-3 h-3 text-muted-foreground" /><span className="text-sm text-foreground">{result.num_detections}</span></div>
                    <div className="col-span-2 flex items-center flex-wrap gap-1">{labels.map((label) => <span key={label} className="text-xs px-1.5 py-0.5 bg-secondary text-muted-foreground capitalize truncate max-w-20">{label}</span>)}{result.items.length > 3 && <span className="text-xs text-muted-foreground">+{result.items.length - 3}</span>}</div>
                    <div className="col-span-2 flex items-center"><StatusBadge status={overall} size="sm" /></div>
                    <div className="col-span-1 flex items-center gap-1.5"><button onClick={(event) => { event.stopPropagation(); setJsonView(isJsonView ? null : key); }} className="p-1 text-muted-foreground hover:text-foreground text-xs" title="Voir le JSON">{'{}'}</button><button onClick={(event) => { event.stopPropagation(); downloadJson(result); }} className="p-1 text-muted-foreground hover:text-foreground" title="Télécharger le JSON"><Download className="w-3.5 h-3.5" /></button></div>
                  </div>
                  {isExpanded && <div className="px-4 pb-4 space-y-4 bg-secondary/20 border-t border-border">
                    <div className="pt-3"><p className="text-xs text-muted-foreground uppercase tracking-wider mb-3">Rapport farmer</p>{farmerLoading[key] ? <div className="py-8 text-center text-sm text-muted-foreground">Génération du rapport...</div> : farmerReport ? <FarmerReportView report={farmerReport} /> : <div className="py-8 text-center text-sm text-muted-foreground">Rapport farmer indisponible pour cette inspection.</div>}</div>
                    <details className="border border-border bg-card"><summary className="cursor-pointer px-3 py-2 text-xs uppercase tracking-wider text-muted-foreground">Détails techniques</summary><div className="p-3 space-y-3">{result.items.length === 0 ? <p className="text-sm text-muted-foreground">Aucun produit détecté dans cette image.</p> : result.items.map((item, index) => <DetectionCard key={index} item={item} index={index} />)}</div></details>
                  </div>}
                  {isJsonView && <div className="px-4 pb-4 bg-secondary/20 border-t border-border"><pre className="text-xs text-foreground overflow-auto max-h-64 mt-3 p-3 bg-background border border-border">{JSON.stringify(result, null, 2)}</pre></div>}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
