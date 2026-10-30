import { ArrowRight, BarChart3, FileText, FlaskConical, ListTree, Lock, ScanSearch, Telescope } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { UploadPanel } from '../components/UploadPanel'
import { Badge, Button, Card, PageHeader, Spinner, Stat } from '../components/ui'
import type { Any, Envelope, Experiment } from '../lib/api'
import { fmtDate, fmtInt, fmtMetric, fmtPct, IMBALANCE_LABEL } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

const CAPS = [
  { icon: ListTree, title: 'Universal schema mapping', text: 'Any column names, any number of PCA or numeric features, categorical data, timestamps and entity IDs.' },
  { icon: Telescope, title: 'Full-data EDA', text: 'Class imbalance, value and time analysis, correlations and separation indicators computed on every record.' },
  { icon: FlaskConical, title: 'Leakage-safe experiments', text: 'Train/validation/test protocols, pipelines fitted on training data, resampling inside training only.' },
  { icon: BarChart3, title: 'Honest benchmarking', text: 'Thresholds frozen on validation, a single test estimate, PR-AUC against the prevalence baseline.' },
  { icon: ScanSearch, title: 'Record-level insight', text: 'Inspect true/false positives and negatives, export predictions, score new data with compatibility checks.' },
  { icon: FileText, title: 'Reproducible reports', text: 'Markdown/HTML reports, CSV tables, PNG/SVG figures and manifests with dataset fingerprints and seeds.' },
]

function DatasetSummary() {
  const { dataset, schema } = useSession()
  const ov = useApi<Envelope>(dataset && schema ? `/api/datasets/${dataset.id}/eda/overview?h=${schema.hash}` : null)
  const exps = useApi<Experiment[]>(dataset ? `/api/experiments?dataset_id=${dataset.id}` : null, { cache: false })
  const lb = useApi<Any>(dataset && schema ? `/api/datasets/${dataset.id}/leaderboard?h=${schema.hash}` : null, { cache: false })
  const r = ov.data?.dataset_id === dataset?.id ? ov.data?.result : null
  const cb = r?.class_balance
  const best = lb.data?.groups?.[0]?.runs?.find((x: Any) => x.run_id === lb.data.groups[0].best_run_id)
  return (
    <div className="space-y-6">
      <Card title={dataset!.original_filename} subtitle={<span className="num">SHA-256 {dataset!.sha256.slice(0, 24)}… · uploaded {fmtDate(dataset!.created_at)}</span>}
        actions={!schema ? <Link to="/schema"><Button variant="primary" size="sm">Map schema <ArrowRight className="size-3.5" /></Button></Link> : <Link to="/explore"><Button size="sm">Explore <ArrowRight className="size-3.5" /></Button></Link>}>
        {!schema ? (
          <p className="text-sm text-muted">{fmtInt(dataset!.n_rows)} rows × {dataset!.n_cols} columns uploaded. Confirm the schema mapping to compute statistics — nothing is computed from assumptions.</p>
        ) : !r ? <Spinner label="Computing dataset overview…" /> : (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="Records" value={fmtInt(r.n_records)} />
            <Stat label="Features" value={fmtInt(r.n_features)} sub={`${r.n_pca_features} precomputed PCA`} />
            <Stat label={`${cb?.positive_name ?? 'Positive'} records`} value={cb ? fmtInt(cb.n_positive) : '—'} tone="pos" />
            <Stat label="Prevalence" value={cb ? fmtPct(cb.prevalence, 3) : '—'} sub={cb ? `1 in ${fmtInt(cb.one_in_n)}` : undefined} />
            <Stat label="Majority baseline acc." value={cb ? fmtPct(cb.majority_baseline_accuracy, 2) : '—'} tone="warn" />
            <Stat label="Duplicates" value={fmtInt(r.duplicate_rows)} />
          </div>
        )}
      </Card>
      {schema && (
        <div className="grid gap-6 xl:grid-cols-2">
          <Card title="Best model so far" subtitle="Ranked by validation PR-AUC; test is a single final estimate.">
            {best ? (
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <div className="font-display text-lg font-semibold">{best.algorithm_name}</div>
                  <div className="text-xs text-muted">{IMBALANCE_LABEL[best.imbalance]} · {best.experiment_name}</div>
                </div>
                <div className="num grid grid-cols-3 gap-4 text-right text-sm">
                  <div><div className="text-[11px] text-dim">val PR-AUC</div>{fmtMetric(best.validation.pr_auc)}</div>
                  <div><div className="text-[11px] text-dim">test PR-AUC</div>{fmtMetric(best.test.pr_auc)}</div>
                  <div><div className="text-[11px] text-dim">test recall</div>{fmtMetric(best.test.recall)}</div>
                </div>
                <Link to="/benchmark" className="text-sm text-accent hover:underline">Leaderboard →</Link>
              </div>
            ) : <div className="flex items-center justify-between text-sm text-muted">No trained models yet. <Link to="/studio"><Button size="sm" variant="primary">Run a Quick Benchmark</Button></Link></div>}
          </Card>
          <Card title="Recent experiments" pad={false}>
            <div className="divide-y divide-line">
              {(exps.data ?? []).slice(0, 5).map((e) => (
                <Link key={e.id} to={`/benchmark?experiment=${e.id}`} className="flex items-center justify-between px-5 py-2.5 text-sm hover:bg-panel-2">
                  <span className="truncate">{e.name}</span>
                  <span className="flex items-center gap-2 text-xs text-dim"><Badge tone={e.status === 'completed' ? 'ok' : 'default'}>{e.status}</Badge>{e.n_completed}/{e.n_runs}</span>
                </Link>
              ))}
              {exps.data?.length === 0 && <p className="px-5 py-4 text-sm text-dim">None yet.</p>}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

export default function Overview() {
  const { dataset, refresh } = useSession()
  const nav = useNavigate()
  return (
    <div>
      <PageHeader eyebrow="Financial anomaly detection & ML research platform" title="FinTech Anomaly Detection"
        description="A local research cockpit for imbalanced financial data: upload a dataset, map its schema, explore it rigorously, benchmark models under a leakage-safe protocol and export reproducible evidence." />
      {!dataset ? (
        <div className="space-y-8">
          <UploadPanel large onUploaded={async () => { await refresh(); nav('/schema') }} />
          <div className="flex items-center justify-center gap-2 text-xs text-dim"><Lock className="size-3.5" /> No dataset is preloaded. Files are processed by the backend on 127.0.0.1 and never transmitted elsewhere.</div>
        </div>
      ) : <DatasetSummary />}
      <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {CAPS.map((c) => (
          <div key={c.title} className="rounded-xl border border-line bg-panel/70 p-5">
            <c.icon className="mb-3 size-5 text-accent" />
            <div className="font-display font-semibold">{c.title}</div>
            <p className="mt-1 text-sm leading-relaxed text-muted">{c.text}</p>
          </div>
        ))}
      </div>
      <p className="mt-6 text-center text-[11px] text-dim">FTAD v1.0 · Open-source research software. Results depend on the data you upload and are never fabricated or precomputed.</p>
    </div>
  )
}
