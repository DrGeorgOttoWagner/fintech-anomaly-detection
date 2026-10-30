import clsx from 'clsx'
import {
  Activity,
  BarChart3,
  Check,
  Database,
  FileText,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  ListTree,
  ScanSearch,
  Sparkles,
  Telescope,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import type { Experiment } from '../lib/api'
import { fmtInt } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

export const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, group: 'Research' },
  { to: '/datasets', label: 'Datasets', icon: Database, group: 'Research' },
  { to: '/schema', label: 'Schema Mapping', icon: ListTree, group: 'Research' },
  { to: '/explore', label: 'Exploratory Analysis', icon: Telescope, group: 'Research' },
  { to: '/studio', label: 'Experiment Studio', icon: FlaskConical, group: 'Modelling' },
  { to: '/benchmark', label: 'ML Benchmark', icon: BarChart3, group: 'Modelling' },
  { to: '/evaluation', label: 'Model Evaluation', icon: Gauge, group: 'Modelling' },
  { to: '/anomalies', label: 'Anomaly Explorer', icon: ScanSearch, group: 'Modelling' },
  { to: '/inference', label: 'Score New Data', icon: Sparkles, group: 'Modelling' },
  { to: '/reports', label: 'Research Reports', icon: FileText, group: 'Output' },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <svg viewBox="0 0 32 32" className="size-8" aria-hidden>
        <rect width="32" height="32" rx="8" fill="#0f1a1f" stroke="#1f3a3a" />
        <path d="M6 21 L11 14 L15 18 L20 9 L26 15" fill="none" stroke="#2dd4bf" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="20" cy="9" r="2.8" fill="#fb7185" />
      </svg>
      <div className="leading-tight">
        <div className="font-display text-[15px] font-bold tracking-tight">FTAD</div>
        <div className="text-[10.5px] text-dim">FinTech Anomaly Detection</div>
      </div>
    </div>
  )
}

function Workflow() {
  const { dataset, schema } = useSession()
  const { data: exps } = useApi<Experiment[]>(dataset ? `/api/experiments?dataset_id=${dataset.id}` : null, { cache: false, poll: 8000 })
  const { data: reports } = useApi<unknown[]>(dataset ? `/api/reports?dataset_id=${dataset.id}` : null, { cache: false, poll: 15000 })
  const done = exps?.some((e) => e.status.startsWith('completed'))
  const steps = [
    { label: 'Upload', ok: !!dataset, to: '/datasets' },
    { label: 'Schema', ok: !!schema, to: '/schema' },
    { label: 'Explore', ok: !!schema, to: '/explore' },
    { label: 'Experiment', ok: !!done, to: '/studio' },
    { label: 'Evaluate', ok: !!done, to: '/benchmark' },
    { label: 'Report', ok: !!reports?.length, to: '/reports' },
  ]
  return (
    <ol className="hidden items-center gap-1 xl:flex" aria-label="Research workflow">
      {steps.map((s, i) => (
        <li key={s.label} className="flex items-center gap-1">
          <NavLink to={s.to} className={clsx('flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors hover:bg-panel-2', s.ok ? 'text-text' : 'text-dim')}>
            <span className={clsx('flex size-4 items-center justify-center rounded-full border text-[9px]', s.ok ? 'border-accent bg-accent/15 text-accent' : 'border-line-2')}>
              {s.ok ? <Check className="size-2.5" strokeWidth={3} /> : i + 1}
            </span>
            {s.label}
          </NavLink>
          {i < steps.length - 1 && <span className="h-px w-3 bg-line-2" />}
        </li>
      ))}
    </ol>
  )
}

function DatasetChip() {
  const { dataset, schema } = useSession()
  if (!dataset)
    return (
      <NavLink to="/datasets" className="flex items-center gap-2 rounded-lg border border-dashed border-line-2 px-3 py-1.5 text-xs text-muted hover:border-accent hover:text-text">
        <Database className="size-3.5" /> No dataset loaded
      </NavLink>
    )
  return (
    <NavLink to="/datasets" className="flex min-w-0 items-center gap-2.5 rounded-lg border border-line-2 bg-panel-2 px-3 py-1.5 hover:border-dim" data-testid="active-dataset">
      <span className="relative flex size-2">
        <span className="absolute inline-flex size-full rounded-full bg-accent opacity-40" />
        <span className="relative inline-flex size-2 rounded-full bg-accent" />
      </span>
      <span className="min-w-0">
        <span className="block max-w-[220px] truncate text-xs font-medium">{dataset.original_filename}</span>
        <span className="num block text-[10.5px] text-dim">
          {fmtInt(dataset.n_rows)} × {dataset.n_cols} · {dataset.sha256.slice(0, 8)} · {schema ? `schema ${schema.hash.slice(0, 6)}` : 'schema pending'}
        </span>
      </span>
    </NavLink>
  )
}

export function Layout({ children }: { children: ReactNode }) {
  const loc = useLocation()
  const groups = [...new Set(NAV.map((n) => n.group))]
  const current = NAV.find((n) => (n.to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(n.to)))
  return (
    <div className="flex h-full">
      <aside className="hidden w-[236px] shrink-0 flex-col border-r border-line bg-[#0b0f15] md:flex">
        <div className="flex h-14 items-center border-b border-line px-3">
          <Logo />
        </div>
        <nav className="flex-1 overflow-y-auto px-3 py-4" aria-label="Primary">
          {groups.map((g) => (
            <div key={g} className="mb-5">
              <div className="mb-1.5 px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-dim">{g}</div>
              {NAV.filter((n) => n.group === g).map((n) => (
                <NavLink
                  key={n.to}
                  to={n.to}
                  end={n.to === '/'}
                  className={({ isActive }) =>
                    clsx(
                      'group mb-0.5 flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] transition-colors',
                      isActive ? 'bg-panel-3 text-text ring-1 ring-line-2' : 'text-muted hover:bg-panel-2 hover:text-text',
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      <n.icon className={clsx('size-4', isActive ? 'text-accent' : 'text-dim group-hover:text-muted')} />
                      {n.label}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="border-t border-line px-4 py-3 text-[11px] leading-relaxed text-dim">
          <div className="flex items-center gap-1.5 text-muted"><Activity className="size-3 text-ok" /> Local-only · 127.0.0.1</div>
          Data never leaves this machine.
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-line bg-bg/80 px-5 backdrop-blur">
          <div className="flex min-w-0 items-center gap-3">
            <div className="md:hidden"><Logo /></div>
            <div className="hidden text-sm text-dim md:block">
              FTAD <span className="mx-1.5 text-line-2">/</span>
              <span className="text-text">{current?.label ?? 'Page'}</span>
            </div>
          </div>
          <Workflow />
          <DatasetChip />
        </header>
        <nav className="flex gap-1 overflow-x-auto border-b border-line px-3 py-2 md:hidden" aria-label="Primary mobile">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => clsx('whitespace-nowrap rounded-md px-2.5 py-1 text-xs', isActive ? 'bg-panel-3 text-text' : 'text-muted')}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <main className="grid-bg flex-1 overflow-y-auto">
          <div key={loc.pathname} className="fade-in mx-auto w-full max-w-[1480px] px-5 py-7 md:px-8">{children}</div>
        </main>
      </div>
    </div>
  )
}
