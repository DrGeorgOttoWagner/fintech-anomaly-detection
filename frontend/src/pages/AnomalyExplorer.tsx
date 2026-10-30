import clsx from 'clsx'
import { Download, ScanSearch } from 'lucide-react'
import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Badge, Button, Card, EmptyState, ErrorBox, NeedSchema, PageHeader, Segmented, Select, Spinner } from '../components/ui'
import type { Any, Experiment } from '../lib/api'
import { fmtInt, fmtNum, IMBALANCE_LABEL, ROLE_LABEL, SCORE_TYPE_LABEL } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

const OUTCOMES = [
  { k: 'TP', label: 'True positive', tone: 'ok' },
  { k: 'FP', label: 'False positive', tone: 'warn' },
  { k: 'FN', label: 'False negative', tone: 'pos' },
  { k: 'TN', label: 'True negative', tone: 'dim' },
] as const

function RecordDetail({ datasetId, row }: { datasetId: string; row: number }) {
  const { data, error } = useApi<Any>(`/api/datasets/${datasetId}/rows/${row}`)
  if (error) return <ErrorBox error={error} />
  if (!data) return <Spinner />
  return (
    <div className="max-h-[560px] overflow-auto">
      <table className="w-full text-[12.5px]">
        <tbody>
          {data.values.map((v: Any) => (
            <tr key={v.column} className="border-b border-line/60">
              <td className="py-1.5 pr-3 text-muted">{v.column}<div className="text-[10px] text-dim">{ROLE_LABEL[v.role] ?? ''}</div></td>
              <td className="num py-1.5 text-right">{v.value === null ? '—' : typeof v.value === 'number' ? fmtNum(v.value, 5) : String(v.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-dim">Original values of row {fmtInt(row)} from the uploaded file. Anonymised components are shown as numbers only; no meaning is attributed to them.</p>
    </div>
  )
}

export default function AnomalyExplorer() {
  const { dataset, schema } = useSession()
  const [params, setParams] = useSearchParams()
  const exps = useApi<Experiment[]>(dataset ? `/api/experiments?dataset_id=${dataset.id}` : null, { cache: false })
  const completed = (exps.data ?? []).filter((e) => e.status.startsWith('completed'))
  const [expId, setExpId] = useState<string | null>(null)
  const runParam = params.get('run')
  const allRuns = useApi<Any>(dataset && schema ? `/api/datasets/${dataset.id}/leaderboard?h=${schema.hash}` : null, { cache: false })
  const runs: Any[] = (allRuns.data?.groups ?? []).flatMap((g: Any) => g.runs)
  const runId = runParam ?? runs.find((r) => !expId || r.experiment_id === expId)?.run_id ?? null
  const [partition, setPartition] = useState<'test' | 'validation'>('test')
  const [outcome, setOutcome] = useState<string[]>([])
  const [sort, setSort] = useState<'score_desc' | 'score_asc'>('score_desc')
  const [offset, setOffset] = useState(0)
  const [selected, setSelected] = useState<number | null>(null)
  const limit = 50
  const q = runId ? `/api/runs/${runId}/records?partition=${partition}&sort=${sort}&offset=${offset}&limit=${limit}${outcome.length ? `&outcome=${outcome.join(',')}` : ''}` : null
  const rec = useApi<Any>(q)

  if (!dataset || !schema) return <><PageHeader eyebrow="Records" title="Anomaly Explorer" /><NeedSchema hasDataset={!!dataset} /></>
  const data = rec.data && rec.data.dataset_id === dataset.id ? rec.data : null
  const pos = schema.config.positive_name

  return (
    <div>
      <PageHeader eyebrow="Records" title="Anomaly Explorer" description="Record-level predictions of a trained model on the validation or test partition: actual label, score, applied threshold and outcome. Click a record to see its feature values." />
      {exps.data && completed.length === 0 ? (
        <EmptyState icon={<ScanSearch className="size-7" />} title="No model predictions yet" action={<Link to="/studio"><Button variant="primary">Run an experiment</Button></Link>}>Predictions exist only after a model has been trained.</EmptyState>
      ) : (
        <>
          <div className="mb-5 flex flex-wrap items-center gap-3">
            <Select value={expId ?? ''} onChange={(v) => { setExpId(v || null); setParams({}); setOffset(0) }} options={[{ value: '', label: 'All experiments' }, ...completed.map((e) => ({ value: e.id, label: e.name }))]} ariaLabel="Experiment" className="max-w-[300px]" />
            <Select value={runId ?? ''} onChange={(v) => { setParams({ run: v }); setOffset(0); setSelected(null) }} ariaLabel="Model run"
              options={runs.filter((r) => !expId || r.experiment_id === expId).map((r) => ({ value: r.run_id, label: `${r.algorithm_name} · ${IMBALANCE_LABEL[r.imbalance]} · ${r.experiment_name.slice(0, 28)}` }))} className="max-w-[420px]" />
            <Segmented value={partition} onChange={(v) => { setPartition(v); setOffset(0); setSelected(null) }} options={[{ value: 'test', label: 'Test' }, { value: 'validation', label: 'Validation' }]} />
            <Segmented value={sort} onChange={(v) => { setSort(v); setOffset(0) }} options={[{ value: 'score_desc', label: 'Highest score' }, { value: 'score_asc', label: 'Lowest score' }]} />
          </div>
          {rec.error && <ErrorBox error={rec.error} />}
          {!data ? <Spinner /> : (
            <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
              <Card title={<span>Records · {partition}</span>}
                subtitle={<span>Threshold <span className="num text-text">{fmtNum(data.threshold, 5)}</span> ({data.threshold_basis}) · {SCORE_TYPE_LABEL[data.score_type]}</span>}
                actions={<a href={`/api/runs/${runId}/records.csv?partition=${partition}${outcome.length ? `&outcome=${outcome.join(',')}` : ''}`} download><Button size="sm"><Download className="size-3.5" />Export CSV</Button></a>}
                pad={false}>
                <div className="flex flex-wrap gap-2 border-b border-line px-4 py-3">
                  {OUTCOMES.map((o) => {
                    const on = outcome.includes(o.k)
                    return (
                      <button key={o.k} data-testid={`filter-${o.k}`} onClick={() => { setOutcome(on ? outcome.filter((x) => x !== o.k) : [...outcome, o.k]); setOffset(0) }}
                        className={clsx('flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs', on ? 'border-accent/60 bg-accent/10' : 'border-line-2 bg-panel-2 hover:border-dim')}>
                        <Badge tone={o.tone}>{o.k}</Badge><span className="text-muted">{o.label}</span><span className="num text-text">{fmtInt(data.counts[o.k])}</span>
                      </button>
                    )
                  })}
                </div>
                <div className="overflow-auto">
                  <table className="w-full text-[13px]">
                    <thead className="bg-panel-2 text-[11px] uppercase tracking-wider text-dim">
                      <tr>
                        <th className="px-3 py-2 text-left">Row</th>
                        {data.id_column && <th className="px-3 py-2 text-left">{data.id_column}</th>}
                        <th className="px-3 py-2 text-left">Actual</th>
                        <th className="px-3 py-2 text-left">Predicted</th>
                        <th className="px-3 py-2 text-right">Score</th>
                        <th className="px-3 py-2 text-left">Outcome</th>
                        <th className="px-3 py-2 text-left">Correct</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.records.map((r: Any) => (
                        <tr key={r.row} onClick={() => setSelected(r.row)} className={clsx('cursor-pointer border-b border-line/60 hover:bg-panel-2', selected === r.row && 'bg-accent/[0.07]')}>
                          <td className="num px-3 py-1.5">{fmtInt(r.row)}</td>
                          {data.id_column && <td className="num px-3 py-1.5">{r.id}</td>}
                          <td className={clsx('px-3 py-1.5', r.actual && 'text-pos')}>{r.actual ? data.positive_name : data.negative_name}</td>
                          <td className={clsx('px-3 py-1.5', r.predicted && 'text-pos')}>{r.predicted ? data.positive_name : data.negative_name}</td>
                          <td className="num px-3 py-1.5 text-right">{fmtNum(r.score, 5)}</td>
                          <td className="px-3 py-1.5"><Badge tone={OUTCOMES.find((o) => o.k === r.outcome)!.tone}>{r.outcome}</Badge></td>
                          <td className="px-3 py-1.5 text-xs">{r.correct ? <span className="text-ok">yes</span> : <span className="text-bad">no</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="flex items-center justify-between px-4 py-3 text-xs text-muted">
                  <span className="num">{fmtInt(offset + 1)}–{fmtInt(Math.min(offset + limit, data.total))} of {fmtInt(data.total)}</span>
                  <div className="flex gap-1.5">
                    <Button size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>Prev</Button>
                    <Button size="sm" disabled={offset + limit >= data.total} onClick={() => setOffset(offset + limit)}>Next</Button>
                  </div>
                </div>
              </Card>
              <Card title={selected != null ? `Record ${fmtInt(selected)}` : 'Record details'} subtitle={selected == null ? `Select a record. FN = missed ${pos.toLowerCase()}, FP = false alarm.` : undefined}>
                {selected != null ? <RecordDetail datasetId={dataset.id} row={selected} /> : <p className="text-sm text-dim">No record selected.</p>}
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  )
}
