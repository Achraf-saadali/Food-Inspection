/**
 * Persistent left sidebar navigation.
 * Design: Industrial Precision — icon + label nav with amber accent on active state
 */

import { Link, useLocation } from 'wouter';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  ScanLine,
  FileText,
  Cpu,
  ChevronRight,
  Activity,
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/inspection', label: 'Live Inspection', icon: ScanLine },
  { path: '/reports', label: 'Reports', icon: FileText },
  { path: '/model', label: 'Model Info', icon: Cpu },
];

export function Sidebar() {
  const [location] = useLocation();

  return (
    <aside className="flex flex-col w-60 min-h-screen border-r border-border bg-sidebar shrink-0 relative">
      {/* Amber top accent line */}
      <div className="absolute top-0 left-0 right-0 h-0.5 bg-primary" />
      {/* Brand */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-border mt-0.5">
        <div className="w-8 h-8 flex items-center justify-center bg-primary/10 border border-primary/30" style={{ borderRadius: 2 }}>
          <Activity className="w-5 h-5 text-primary" />
        </div>
        <div>
          <p className="text-sm font-bold text-foreground font-mono tracking-tight">
            FoodScan AI
          </p>
          <p className="text-xs text-primary/70 font-mono">INSPECTION PLATFORM</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-0.5">
        <p className="px-3 py-1.5 text-xs font-mono text-muted-foreground uppercase tracking-widest mb-2">
          Navigation
        </p>
        {NAV_ITEMS.map(({ path, label, icon: Icon }) => {
          const isActive = path === '/' ? location === '/' : location.startsWith(path);
          return (
            <Link key={path} href={path}>
              <div
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 text-sm transition-all duration-150 group font-mono',
                  isActive
                    ? 'bg-primary/10 text-primary font-medium border border-primary/25 border-l-2'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                )}
                style={{ borderRadius: 2 }}
              >
                <Icon className={cn('w-4 h-4 shrink-0', isActive ? 'text-primary' : '')} />
                <span className="flex-1">{label}</span>
                {isActive && <ChevronRight className="w-3 h-3 text-primary" />}
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#22c55e] animate-pulse" />
          <span className="text-xs text-muted-foreground font-mono">SYSTEM ONLINE</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1 font-mono">YOLOv9c · 63 CLASSES</p>
      </div>
    </aside>
  );
}
