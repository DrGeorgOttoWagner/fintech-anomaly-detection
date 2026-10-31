import clsx from 'clsx'
import { Check, RotateCcw, Sparkles } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Badge, Button, Callout, Card, EmptyState, ErrorBox, Field, PageHeader, Select, Spinner } from '../components/ui'
import { api, type Any, type Role, type SchemaConfig } from '../lib/api'
import { fmtInt, fmtPct, ROLE_LABEL, ROLE_SHORT } from '../lib/format'
import { useApi, useDebounced } from '../lib/hooks'
import { useSession } from '../lib/session'

const ROLES: Role[] = ['target', 'numeric', 'categorical', 'pca', 'time', 'transaction_id', 'group_id', 'ignore']
const ROLE_TONE: Record<Role, string> = {
  target: 'border-pos/40 bg-pos/10 text-pos',
  numeric: 'border-accent/30 bg-accent/10 text-accent',
  categorical: 'border-[#a78bfa]/30 bg-[#a78bfa]/10 text-[#c4b5fd]',
  pca: 'border-accent-2/30 bg-accent-2/10 text-accent-2',
  time: 'border-warn/30 bg-warn/10 text-warn',
  transaction_id: 'border-line-2 bg-panel-3 text-muted',
  group_id: 'border-line-2 bg-panel-3 text-muted',
  ignore: 'border-line bg-transparent text-dim',
}

export default function Schema() {
  const { dataset, schema, refresh } = useSession()
  const nav = useNavigate()
  const inspect = useApi<Any>(dataset ? `/api/datasets/${dataset.id}/inspect` : null)
  const [cfg, setCfg] = useState<SchemaConfig | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveErr, setSaveErr] = useState<unknown>(null)
  const [saved, setSaved] = useState(false)
  const [q, setQ] = useState('')
  const [bulk, setBulk] = useState<Role>('numeric')
  const [selected, setSelected] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!inspect.data) return
    setCfg(schema?.config ? { ...schema.config } : { ...inspect.data.suggested })
    setSaved(!!schema)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspect.data, dataset?.id])

  const targetValues = useApi<Any>(dataset && cfg?.target ? `/api/datasets/${dataset.id}/values?column=${encodeURIComponent(cfg.target)}` : null)
  const debounced = useDebounced(cfg, 250)
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[]; warnings: string[] } | null>(null)
  useEffect(() => {
    if (!dataset || !debounced) return
    let alive = true
    api.post(`/api/datasets/${dataset.id}/schema/validate`, debounced).then((v) => alive && setValidation(v)).catch(() => {})
    return () => { alive = false }
  }, [debounced, dataset])

  const profiles: Any[] = inspect.data?.profiles ?? []
  const shown = useMemo(() => profiles.filter((p) => p.name.toLowerCase().includes(q.toLowerCase())), [profiles, q])

  if (!dataset) return <EmptyState title="No dataset loaded" action={<Button variant="primary" onClick={() => nav('/datasets')}>Upload a dataset</Button>}>Upload a file to map its columns.</EmptyState>
  if (inspect.error) return <ErrorBox error={inspect.error} />
  if (!inspect.data || !cfg) return <Spinner label="Inspecting columns…" />

  const setRole = (col: string, role: Role) => {
    setSaved(false)
    setCfg((c) => {
      if (!c) return c
      const roles = { ...c.roles }
      const next = { ...c }
      if (role === 'target') {
        for (const [k, r] of Object.entries(roles)) if (r === 'target') roles[k] = 'ignore'
        next.target = col
        next.positive_label = null
      } else if (c.target === col) {
        next.target = null
        next.positive_label = null
      }
      for (const [key, r] of [['time_column', 'time'], ['id_column', 'transaction_id'], ['group_column', 'group_id']] as const) {
        if (role === r) {
          if (c[key] && c[key] !== col) roles[c[key] as string] = 'ignore'
          ;(next as Any)[key] = col
          if (key === 'time_column') next.time_kind = profiles.find((p) => p.name === col)?.kind === 'datetime' ? 'timestamp' : (c.time_kind ?? 'ordinal')
        } else if (c[key] === col) {
          ;(next as Any)[key] = null
          if (key === 'time_column') next.time_kind = null
        }
      }
      roles[col] = role
      if (next.amount_column === col && !['numeric', 'pca', 'ignore'].includes(role)) next.amount_column = null
      next.roles = roles
      return next
    })
  }

  const counts = ROLES.map((r) => [r, Object.values(cfg.roles).filter((x) => x === r).length] as const)
  const numericCols = profiles.filter((p) => p.kind === 'numeric').map((p) => p.name)

  const save = async () => {
    setSaving(true)
    setSaveErr(null)
    try {
      await api.put(`/api/datasets/${dataset.id}/schema`, cfg)
      await refresh()
      setSaved(true)
    } catch (e) {
      setSaveErr(e)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Step 2 · Schema"
        title="Schema Mapping"
        description="Assign an analytical role to every column. FTAD suggests roles from names, types and value patterns — nothing is enforced; you decide the target, the positive label, features, identifiers and time handling."
        actions={
          <>
            <Button variant="ghost" onClick={() => { setCfg({ ...inspect.data.suggested }); setSaved(false) }}><RotateCcw className="size-4" />Reset to suggestion</Button>
            <Button variant="primary" onClick={save} loading={saving} disabled={!validation?.valid} data-testid="confirm-schema">
              <Check className="size-4" />{saved ? 'Schema confirmed' : 'Confirm schema'}
            </Button>
          </>
        }
      />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card title="Columns" subtitle={`${profiles.length} columns detected · ${fmtInt(dataset.n_rows)} rows`} pad={false}
          actions={
            <div className="flex items-center gap-2">
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter columns…" className="h-8 w-40 rounded-lg border border-line-2 bg-panel-2 px-2.5 text-xs outline-none focus:border-accent" />
            </div>
          }>
          {selected.size > 0 && (
            <div className="flex items-center gap-2 border-b border-line bg-panel-2 px-4 py-2 text-xs">
              <span className="text-muted">{selected.size} selected → set role</span>
              <Select value={bulk} onChange={(v) => setBulk(v as Role)} options={ROLES.filter((r) => !['target', 'time', 'transaction_id', 'group_id'].includes(r)).map((r) => ({ value: r, label: ROLE_LABEL[r] }))} className="h-7 text-xs" />
              <Button size="sm" onClick={() => { selected.forEach((c) => setRole(c, bulk)); setSelected(new Set()) }}>Apply</Button>
              <Button size="sm" variant="ghost" onClick={() => setSelected(new Set())}>Clear</Button>
            </div>
          )}
          <div className="max-h-[640px] overflow-auto">
            <table className="w-full text-[13px]">
              <thead className="sticky top-0 z-10 bg-panel-2 text-[11px] uppercase tracking-wider text-dim">
                <tr>
                  <th className="w-8 border-b border-line px-3 py-2"><input type="checkbox" aria-label="Select all" checked={selected.size === shown.length && shown.length > 0} onChange={(e) => setSelected(e.target.checked ? new Set(shown.map((p) => p.name)) : new Set())} /></th>
                  <th className="border-b border-line px-3 py-2 text-left">Column</th>
                  <th className="border-b border-line px-3 py-2 text-left">Type</th>
                  <th className="border-b border-line px-3 py-2 text-right">Unique</th>
                  <th className="border-b border-line px-3 py-2 text-right">Missing</th>
                  <th className="border-b border-line px-3 py-2 text-left">Detected</th>
                  <th className="border-b border-line px-3 py-2 text-left">Role</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((p) => {
                  const role = cfg.roles[p.name]
                  const suggested = inspect.data.suggested.roles[p.name]
                  return (
                    <tr key={p.name} className="border-b border-line/60 hover:bg-panel-2/60">
                      <td className="px-3 py-2"><input type="checkbox" aria-label={`Select ${p.name}`} checked={selected.has(p.name)} onChange={(e) => { const s = new Set(selected); if (e.target.checked) s.add(p.name); else s.delete(p.name); setSelected(s) }} /></td>
                      <td className="px-3 py-2">
                        <div className="font-medium">{p.name}</div>
                        <div className="max-w-[220px] truncate text-[11px] text-dim" title={p.sample_values.join(', ')}>e.g. {p.sample_values.slice(0, 3).join(', ')}</div>
                      </td>
                      <td className="px-3 py-2 text-xs text-muted">{p.kind}<div className="text-[11px] text-dim">{p.dtype}</div></td>
                      <td className="num px-3 py-2 text-right text-xs">{fmtInt(p.n_unique)}</td>
                      <td className="num px-3 py-2 text-right text-xs">{p.n_missing ? <span className="text-warn">{fmtPct(p.missing_pct, 1)}</span> : '0'}</td>
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap gap-1">
                          {p.potential_target && <Badge tone="pos">target?</Badge>}
                          {p.potential_pca && <Badge tone="accent">PCA pattern</Badge>}
                          {p.potential_time && <Badge tone="warn">time</Badge>}
                          {p.potential_amount && <Badge>amount</Badge>}
                          {p.potential_identifier && <Badge tone="dim">identifier</Badge>}
                          {p.potential_group && <Badge tone="dim">entity</Badge>}
                          {p.is_constant && <Badge tone="warn">constant</Badge>}
                        </div>
                      </td>
                      <td className="px-3 py-2">
                        <select aria-label={`Role of ${p.name}`} data-testid={`role-${p.name}`} value={role} onChange={(e) => setRole(p.name, e.target.value as Role)}
                          className={clsx('h-7 rounded-md border px-1.5 text-xs font-medium outline-none', ROLE_TONE[role])}>
                          {ROLES.map((r) => <option key={r} value={r} className="bg-panel text-text">{ROLE_SHORT[r]}</option>)}
                        </select>
                        {suggested !== role && <div className="mt-0.5 text-[10px] text-dim">suggested: {ROLE_SHORT[suggested]}</div>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>

        <div className="space-y-5">
          <Card title="Target & labels">
            <div className="space-y-4">
              <Field label="Target column">
                <Select value={cfg.target ?? ''} onChange={(v) => v ? setRole(v, 'target') : cfg.target && setRole(cfg.target, 'ignore')}
                  options={[{ value: '', label: '— none (unsupervised EDA only) —' }, ...profiles.filter((p) => p.n_unique >= 2 && p.n_unique <= 50).map((p) => ({ value: p.name, label: p.name }))]} className="w-full" ariaLabel="Target column" />
              </Field>
              {cfg.target && (
                <Field label="Positive class / anomaly label" hint="Records with this value are the class of interest (e.g. fraud). All other values form the negative class.">
                  <div className="flex flex-wrap gap-1.5" data-testid="positive-labels">
                    {(targetValues.data?.values ?? []).map((v: Any) => (
                      <button key={v.value} onClick={() => { setCfg({ ...cfg, positive_label: v.value }); setSaved(false) }}
                        className={clsx('rounded-lg border px-2.5 py-1.5 text-left text-xs', cfg.positive_label === v.value ? 'border-pos bg-pos/10 text-text' : 'border-line-2 bg-panel-2 text-muted hover:border-dim')}>
                        <span className="num font-semibold">{v.value}</span> <span className="num text-dim">· {fmtInt(v.count)}</span>
                      </button>
                    ))}
                  </div>
                </Field>
              )}
              <div className="grid grid-cols-2 gap-3">
                <Field label="Positive class name"><input value={cfg.positive_name} onChange={(e) => { setCfg({ ...cfg, positive_name: e.target.value || 'Positive' }); setSaved(false) }} className="h-9 w-full rounded-lg border border-line-2 bg-panel-2 px-2.5 text-sm outline-none focus:border-accent" /></Field>
                <Field label="Negative class name"><input value={cfg.negative_name} onChange={(e) => { setCfg({ ...cfg, negative_name: e.target.value || 'Negative' }); setSaved(false) }} className="h-9 w-full rounded-lg border border-line-2 bg-panel-2 px-2.5 text-sm outline-none focus:border-accent" /></Field>
              </div>
            </div>
          </Card>
          <Card title="Time, amount & identifiers">
            <div className="space-y-4">
              <Field label="Time column interpretation" hint={cfg.time_kind === 'elapsed_seconds' ? 'Seconds elapsed since the first record — not a calendar timestamp.' : cfg.time_kind === 'timestamp' ? 'Parsed as calendar date/time values.' : 'Ordered numeric values with unknown unit.'}>
                <Select value={cfg.time_kind ?? ''} disabled={!cfg.time_column} onChange={(v) => { setCfg({ ...cfg, time_kind: (v || null) as SchemaConfig['time_kind'] }); setSaved(false) }}
                  options={[{ value: '', label: cfg.time_column ? 'Select…' : 'No time column' }, { value: 'elapsed_seconds', label: 'Elapsed seconds' }, { value: 'timestamp', label: 'Calendar timestamp' }, { value: 'ordinal', label: 'Ordinal / unknown unit' }]} className="w-full" />
              </Field>
              <Field label="Amount / value column for value analysis">
                <Select value={cfg.amount_column ?? ''} onChange={(v) => { setCfg({ ...cfg, amount_column: v || null }); setSaved(false) }}
                  options={[{ value: '', label: '— none —' }, ...numericCols.filter((c) => c !== cfg.target).map((c) => ({ value: c, label: c }))]} className="w-full" />
              </Field>
              <div className="text-xs text-muted">
                Time: <b className="text-text">{cfg.time_column ?? '—'}</b> · Transaction ID: <b className="text-text">{cfg.id_column ?? '—'}</b> · Entity: <b className="text-text">{cfg.group_column ?? '—'}</b>
              </div>
              <Field label="Domain terminology">
                <Select value={cfg.domain} onChange={(v) => { setCfg({ ...cfg, domain: v as SchemaConfig['domain'] }); setSaved(false) }} options={[{ value: 'fraud', label: 'Fraud detection (adds an analysis-coverage appendix to reports)' }, { value: 'generic', label: 'Generic anomaly / classification' }]} className="w-full" />
              </Field>
            </div>
          </Card>
          <Card title="Role summary">
            <div className="flex flex-wrap gap-1.5">
              {counts.filter(([, n]) => n).map(([r, n]) => <span key={r} className={clsx('rounded-md border px-2 py-1 text-xs', ROLE_TONE[r])}>{ROLE_LABEL[r]} · <span className="num">{n}</span></span>)}
            </div>
            <p className="mt-3 text-xs leading-relaxed text-dim">“Existing PCA feature” marks columns that arrived already transformed. They are modelled like numeric features but never interpreted. Optional PCA performed by FTAD is configured separately in the Experiment Studio and is fitted on training data only.</p>
          </Card>
          {(inspect.data.suggested.notes?.length ?? 0) > 0 && (
            <Callout tone="info" title={<span className="flex items-center gap-1.5"><Sparkles className="size-3.5" />Detection notes</span>}>
              <ul className="mt-1 list-disc space-y-1 pl-4 text-xs">{inspect.data.suggested.notes.map((n: string) => <li key={n}>{n}</li>)}</ul>
            </Callout>
          )}
          {validation && validation.errors.length > 0 && <Callout tone="error" title="Fix before confirming"><ul className="list-disc pl-4 text-xs">{validation.errors.map((e) => <li key={e}>{e}</li>)}</ul></Callout>}
          {validation && validation.warnings.length > 0 && <Callout tone="warn" title="Warnings"><ul className="list-disc pl-4 text-xs">{validation.warnings.map((e) => <li key={e}>{e}</li>)}</ul></Callout>}
          <ErrorBox error={saveErr} />
          {saved && (
            <Callout tone="ok" title="Schema confirmed">
              <span className="text-xs">All analyses now use this mapping.</span>
              <div className="mt-2"><Button size="sm" variant="primary" onClick={() => nav('/explore')}>Explore the data</Button></div>
            </Callout>
          )}
        </div>
      </div>
    </div>
  )
}
