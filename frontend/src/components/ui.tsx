import clsx from 'clsx'
import { AlertTriangle, ArrowDown, ArrowUp, ChevronsUpDown, Info, Loader2, XCircle } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'

export function Card({ title, subtitle, actions, children, className, pad = true }: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children?: ReactNode
  className?: string
  pad?: boolean
}) {
  return (
    <section className={clsx('rounded-xl border border-line bg-panel/90 shadow-[0_1px_0_0_rgba(255,255,255,0.02)_inset]', className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
          <div className="min-w-0">
            {title && <h3 className="font-display text-[15px] font-semibold tracking-tight text-text">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs leading-relaxed text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx(pad && 'p-5')}>{children}</div>
    </section>
  )
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        {eyebrow && <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-accent">{eyebrow}</div>}
        <h1 className="font-display text-[26px] font-semibold leading-tight tracking-tight">{title}</h1>
        {description && <p className="mt-1.5 text-sm leading-relaxed text-muted">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Stat({ label, value, sub, tone, mono = true }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'pos' | 'accent' | 'warn'; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-panel-2/70 px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-dim">{label}</div>
      <div className={clsx('mt-1 text-[22px] font-semibold leading-tight', mono && 'num', tone === 'pos' && 'text-pos', tone === 'accent' && 'text-accent', tone === 'warn' && 'text-warn')}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
    </div>
  )
}

export function Badge({ children, tone = 'default', title }: { children: ReactNode; tone?: 'default' | 'accent' | 'pos' | 'warn' | 'ok' | 'bad' | 'dim'; title?: string }) {
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-none',
        tone === 'default' && 'border-line-2 bg-panel-3 text-muted',
        tone === 'accent' && 'border-accent/30 bg-accent/10 text-accent',
        tone === 'pos' && 'border-pos/30 bg-pos/10 text-pos',
        tone === 'warn' && 'border-warn/30 bg-warn/10 text-warn',
        tone === 'ok' && 'border-ok/30 bg-ok/10 text-ok',
        tone === 'bad' && 'border-bad/30 bg-bad/10 text-bad',
        tone === 'dim' && 'border-line bg-transparent text-dim',
      )}
    >
      {children}
    </span>
  )
}

type BtnProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' | 'danger'; size?: 'sm' | 'md'; loading?: boolean }

export function Button({ variant = 'secondary', size = 'md', loading, className, children, disabled, ...rest }: BtnProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={clsx(
        'inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-45',
        size === 'md' ? 'h-9 px-3.5 text-sm' : 'h-7 px-2.5 text-xs',
        variant === 'primary' && 'bg-accent text-[#04201c] hover:bg-[#5eead4]',
        variant === 'secondary' && 'border border-line-2 bg-panel-2 text-text hover:border-dim hover:bg-panel-3',
        variant === 'ghost' && 'text-muted hover:bg-panel-2 hover:text-text',
        variant === 'danger' && 'border border-bad/40 bg-bad/10 text-bad hover:bg-bad/20',
        className,
      )}
    >
      {loading && <Loader2 className="size-4 animate-spin" />}
      {children}
    </button>
  )
}

export function Segmented<T extends string>({ value, options, onChange, size = 'md' }: { value: T; options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[]; onChange: (v: T) => void; size?: 'sm' | 'md' }) {
  return (
    <div className="inline-flex rounded-lg border border-line-2 bg-panel-2 p-0.5" role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          title={o.title}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          className={clsx(
            'rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
            size === 'md' ? 'px-3 py-1.5 text-[13px]' : 'px-2 py-1 text-xs',
            value === o.value ? 'bg-panel-3 text-text shadow-sm ring-1 ring-line-2' : 'text-muted hover:text-text',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Tabs<T extends string>({ value, tabs, onChange }: { value: T; tabs: { value: T; label: ReactNode }[]; onChange: (v: T) => void }) {
  return (
    <div className="mb-5 flex gap-1 overflow-x-auto border-b border-line" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.value}
          role="tab"
          aria-selected={value === t.value}
          onClick={() => onChange(t.value)}
          className={clsx(
            '-mb-px whitespace-nowrap border-b-2 px-3.5 py-2.5 text-sm font-medium transition-colors',
            value === t.value ? 'border-accent text-text' : 'border-transparent text-muted hover:text-text',
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

export function Select({ value, onChange, options, className, ariaLabel, disabled }: { value: string; onChange: (v: string) => void; options: { value: string; label: string; disabled?: boolean }[]; className?: string; ariaLabel?: string; disabled?: boolean }) {
  return (
    <select
      aria-label={ariaLabel}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className={clsx('h-9 rounded-lg border border-line-2 bg-panel-2 px-2.5 text-sm text-text outline-none focus:border-accent disabled:opacity-50', className)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-muted">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] leading-snug text-dim">{hint}</span>}
    </label>
  )
}

export function NumberInput({ value, onChange, min, max, step, className, ariaLabel }: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; className?: string; ariaLabel?: string }) {
  return (
    <input
      type="number"
      aria-label={ariaLabel}
      value={Number.isFinite(value) ? value : ''}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(e.target.value === '' ? NaN : Number(e.target.value))}
      className={clsx('num h-9 w-full rounded-lg border border-line-2 bg-panel-2 px-2.5 text-sm outline-none focus:border-accent', className)}
    />
  )
}

export function Toggle({ checked, onChange, label, hint, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; hint?: ReactNode; disabled?: boolean }) {
  return (
    <label className={clsx('flex cursor-pointer items-start gap-3', disabled && 'cursor-not-allowed opacity-50')}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={clsx('relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors', checked ? 'bg-accent' : 'bg-line-2')}
      >
        <span className={clsx('absolute top-0.5 size-4 rounded-full bg-white transition-all', checked ? 'left-[18px]' : 'left-0.5')} />
      </button>
      <span>
        <span className="block text-sm text-text">{label}</span>
        {hint && <span className="block text-xs text-dim">{hint}</span>}
      </span>
    </label>
  )
}

export function Callout({ tone = 'info', title, children }: { tone?: 'info' | 'warn' | 'error' | 'ok'; title?: ReactNode; children?: ReactNode }) {
  const Icon = tone === 'error' ? XCircle : tone === 'warn' ? AlertTriangle : Info
  return (
    <div
      className={clsx(
        'flex gap-3 rounded-lg border px-4 py-3 text-sm leading-relaxed',
        tone === 'info' && 'border-accent/25 bg-accent/[0.06] text-[#b7ece4]',
        tone === 'warn' && 'border-warn/30 bg-warn/[0.07] text-[#f6dca9]',
        tone === 'error' && 'border-bad/35 bg-bad/[0.08] text-[#fecaca]',
        tone === 'ok' && 'border-ok/30 bg-ok/[0.06] text-[#bbf7d0]',
      )}
    >
      <Icon className="mt-0.5 size-4 shrink-0 opacity-80" />
      <div className="min-w-0">
        {title && <div className="font-medium text-text">{title}</div>}
        {children}
      </div>
    </div>
  )
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2.5 py-8 text-sm text-muted">
      <Loader2 className="size-4 animate-spin text-accent" /> {label}
    </div>
  )
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null
  const msg = error instanceof Error ? error.message : String(error)
  return <Callout tone="error" title="Request failed">{msg}</Callout>
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line-2 bg-panel/50 px-6 py-14 text-center">
      {icon && <div className="mb-3 text-dim">{icon}</div>}
      <div className="font-display text-lg font-semibold">{title}</div>
      {children && <div className="mt-1.5 max-w-md text-sm text-muted">{children}</div>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

export function NeedSchema({ hasDataset }: { hasDataset: boolean }) {
  return hasDataset ? (
    <EmptyState title="Confirm the schema mapping first" action={<Link to="/schema"><Button variant="primary">Open schema mapping</Button></Link>}>
      Analyses use the column roles you confirm. Nothing is computed until the target, features and other roles are set.
    </EmptyState>
  ) : (
    <EmptyState title="No dataset loaded" action={<Link to="/datasets"><Button variant="primary">Upload a dataset</Button></Link>}>
      FTAD starts empty. Upload a CSV or Parquet file to begin.
    </EmptyState>
  )
}

export interface Column<T> {
  key: string
  header: ReactNode
  render?: (row: T) => ReactNode
  value?: (row: T) => number | string | null | undefined
  align?: 'left' | 'right'
  sortable?: boolean
  className?: string
}

export function DataTable<T>({ rows, columns, initialSort, filterable, filterPlaceholder, rowKey, onRowClick, selectedKey, maxHeight, dense }: {
  rows: T[]
  columns: Column<T>[]
  initialSort?: { key: string; dir: 'asc' | 'desc' }
  filterable?: (row: T) => string
  filterPlaceholder?: string
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  selectedKey?: string | null
  maxHeight?: number
  dense?: boolean
}) {
  const [sort, setSort] = useState(initialSort ?? null)
  const [q, setQ] = useState('')
  const view = useMemo(() => {
    let out = rows
    if (filterable && q) {
      const needle = q.toLowerCase()
      out = out.filter((r) => filterable(r).toLowerCase().includes(needle))
    }
    if (sort) {
      const col = columns.find((c) => c.key === sort.key)
      if (col?.value) {
        const get = col.value
        out = [...out].sort((a, b) => {
          const va = get(a), vb = get(b)
          if (va == null && vb == null) return 0
          if (va == null) return 1
          if (vb == null) return -1
          const c = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))
          return sort.dir === 'asc' ? c : -c
        })
      }
    }
    return out
  }, [rows, q, sort, columns, filterable])

  return (
    <div>
      {filterable && (
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={filterPlaceholder ?? 'Filter…'}
          className="mb-3 h-8 w-full max-w-xs rounded-lg border border-line-2 bg-panel-2 px-3 text-sm outline-none placeholder:text-dim focus:border-accent"
        />
      )}
      <div className="overflow-auto rounded-lg border border-line" style={maxHeight ? { maxHeight } : undefined}>
        <table className="w-full border-collapse text-[13px]">
          <thead className="sticky top-0 z-10 bg-panel-2">
            <tr>
              {columns.map((c) => {
                const active = sort?.key === c.key
                const canSort = c.sortable !== false && !!c.value
                return (
                  <th
                    key={c.key}
                    onClick={() => canSort && setSort({ key: c.key, dir: active && sort?.dir === 'desc' ? 'asc' : 'desc' })}
                    className={clsx(
                      'whitespace-nowrap border-b border-line px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-dim',
                      c.align === 'right' ? 'text-right' : 'text-left',
                      canSort && 'cursor-pointer select-none hover:text-text',
                    )}
                  >
                    <span className="inline-flex items-center gap-1">
                      {c.header}
                      {canSort && (active ? sort?.dir === 'desc' ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" /> : <ChevronsUpDown className="size-3 opacity-40" />)}
                    </span>
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {view.map((r) => {
              const k = rowKey(r)
              return (
                <tr
                  key={k}
                  onClick={onRowClick ? () => onRowClick(r) : undefined}
                  className={clsx('border-b border-line/70 last:border-0', onRowClick && 'cursor-pointer hover:bg-panel-2', selectedKey === k && 'bg-accent/[0.07]')}
                >
                  {columns.map((c) => (
                    <td key={c.key} className={clsx(dense ? 'px-3 py-1.5' : 'px-3 py-2', c.align === 'right' && 'num text-right', c.className)}>
                      {c.render ? c.render(r) : String(c.value?.(r) ?? '—')}
                    </td>
                  ))}
                </tr>
              )
            })}
            {view.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="px-3 py-6 text-center text-sm text-dim">No rows</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function KeyValue({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(120px,auto)_1fr] gap-x-4 gap-y-1.5 text-sm">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="min-w-0 break-words text-text">{v}</dd>
        </div>
      ))}
    </dl>
  )
}
