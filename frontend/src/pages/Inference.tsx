import { Download, ShieldAlert, ShieldCheck, ShieldX } from 'lucide-react'
import { useState } from 'react'
import { UploadPanel } from '../components/UploadPanel'
import { Badge, Button, Callout, Card, DataTable, ErrorBox, PageHeader, Select, Spinner, Stat, Toggle } from '../components/ui'
import { api, ApiError, type Any, type Dataset } from '../lib/api'
import { fmtInt, fmtNum, fmtPct, IMBALANCE_LABEL, SCORE_TYPE_LABEL } from '../lib/format'
import { useApi } from '../lib/hooks'

export default function Inference() {
  const datasets = useApi<Dataset[]>('/api/datasets', { cache: false })
  const [modelDs, setModelDs] = useState<string>('')
  const trainDs = modelDs || datasets.data?.find((d) => d.n_experiments)?.id || ''
  const lb = useApi<Any>(trainDs ? `/api/datasets/${trainDs}/leaderboard` : null, { cache: false })
  const runs: Any[] = (lb.data?.groups ?? []).flatMap((g: Any) => g.runs)
  const [runId, setRunId] = useState('')
  const run = runId || runs[0]?.run_id || ''
  const [target, setTarget] = useState('')
  const [check, setCheck] = useState<Any>(null)
  const [ack, setAck] = useState(false)
  const [result, setResult] = useState<Any>(null)
  const [err, setErr] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)

  const doCheck = async (ds = target) => {
    setErr(null); setResult(null); setCheck(null); setAck(false)
    if (!run || !ds) return
    try { setCheck(await api.post('/api/inference/check', { run_id: run, dataset_id: ds })) } catch (e) { setErr(e) }
  }
  const doRun = async () => {
    setBusy(true); setErr(null)
    try { setResult(await api.post('/api/inference/run', { run_id: run, dataset_id: target, acknowledge_warnings: ack })) } catch (e) {
      setErr(e instanceof ApiError && e.detail?.errors ? new Error(`${e.detail.message} ${e.detail.errors.join(' ')}`) : e)
    } finally { setBusy(false) }
  }
  const Icon = check?.status === 'compatible' ? ShieldCheck : check?.status === 'warnings' ? ShieldAlert : ShieldX

  return (
    <div>
      <PageHeader eyebrow="Deploy" title="Score New Data" description="Apply a previously trained model — with exactly the preprocessing it was fitted with — to another dataset. Nothing is retrained. Compatibility is checked first; incompatible inputs are rejected." />
      <div className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card title="1 · Trained model">
            {!datasets.data ? <Spinner /> : (
              <div className="space-y-3">
                <Select className="w-full" value={trainDs} onChange={(v) => { setModelDs(v); setRunId(''); setCheck(null) }} ariaLabel="Training dataset" options={datasets.data.filter((d) => d.n_experiments).map((d) => ({ value: d.id, label: `Trained on ${d.original_filename}` }))} />
                <Select className="w-full" value={run} onChange={(v) => { setRunId(v); setCheck(null) }} ariaLabel="Model" options={runs.map((r) => ({ value: r.run_id, label: `${r.algorithm_name} · ${IMBALANCE_LABEL[r.imbalance]} · val PR-AUC ${fmtNum(r.validation.pr_auc, 3)}` }))} />
                {runs.length === 0 && <p className="text-xs text-dim">No completed runs available yet.</p>}
              </div>
            )}
          </Card>
          <Card title="2 · Dataset to score" subtitle="Choose any uploaded dataset or upload a new (typically unlabeled) file.">
            <Select className="mb-4 w-full" value={target} onChange={(v) => { setTarget(v); doCheck(v) }} ariaLabel="Dataset to score" options={[{ value: '', label: 'Select a dataset…' }, ...(datasets.data ?? []).map((d) => ({ value: d.id, label: `${d.original_filename} (${fmtInt(d.n_rows)} rows)` }))]} />
            <UploadPanel compact onUploaded={(r) => { datasets.reload(); setTarget(r.dataset.id); doCheck(r.dataset.id) }} />
            <p className="mt-2 text-[11px] text-dim">Uploading here also makes the file the active dataset session.</p>
          </Card>
        </div>
        <div className="space-y-6">
          <ErrorBox error={err} />
          {check && (
            <Card title={<span className="flex items-center gap-2"><Icon className={check.status === 'compatible' ? 'size-5 text-ok' : check.status === 'warnings' ? 'size-5 text-warn' : 'size-5 text-bad'} />Compatibility: {check.status}</span>}
              subtitle={`${check.model.algorithm_name} · ${check.n_features} required features · ${SCORE_TYPE_LABEL[check.model.score_type]} · threshold ${fmtNum(check.model.threshold, 5)}`}>
              <div className="space-y-2">
                {check.errors.map((e: string) => <Callout key={e} tone="error">{e}</Callout>)}
                {check.warnings.map((e: string) => <Callout key={e} tone="warn">{e}</Callout>)}
                {check.info.map((e: string) => <Callout key={e} tone="info">{e}</Callout>)}
              </div>
              {check.pca_drift?.length > 0 && (
                <div className="mt-4">
                  <DataTable<Any> rows={check.pca_drift} rowKey={(r) => r.feature} dense maxHeight={220}
                    columns={[{ key: 'f', header: 'Component', value: (r) => r.feature }, { key: 's', header: 'Mean shift (SD)', align: 'right', value: (r) => r.mean_shift_sd, render: (r) => fmtNum(r.mean_shift_sd, 3) }, { key: 'r', header: 'Std ratio', align: 'right', value: (r) => r.std_ratio, render: (r) => fmtNum(r.std_ratio, 3) }, { key: 'x', header: '', render: (r) => r.flagged ? <Badge tone="warn">shift</Badge> : <Badge tone="ok">ok</Badge> }]} />
                </div>
              )}
              {check.status !== 'incompatible' && (
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                  {check.status === 'warnings' ? <Toggle checked={ack} onChange={setAck} label="I have reviewed the warnings and confirm the input provenance" /> : <span />}
                  <Button variant="primary" onClick={doRun} loading={busy} disabled={check.status === 'warnings' && !ack}>Score dataset</Button>
                </div>
              )}
            </Card>
          )}
          {result && (
            <Card title="Predictions" subtitle={result.note} actions={<a href={`/api/inference/${result.prediction_id}.csv`} download><Button size="sm" variant="primary"><Download className="size-3.5" />Download CSV</Button></a>}>
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Records scored" value={fmtInt(result.n)} />
                <Stat label="Flagged" value={fmtInt(result.n_flagged)} tone="pos" />
                <Stat label="Flag rate" value={fmtPct(result.flagged_rate, 3)} />
              </div>
              <div className="mt-4">
                <DataTable<Any> rows={result.top.map((r: Any, i: number) => ({ ...r, i }))} rowKey={(r) => String(r.i)} dense maxHeight={360}
                  columns={result.columns.map((c: string) => ({ key: c, header: c, align: typeof result.top[0]?.[c] === 'number' ? 'right' : 'left', value: (r: Any) => r[c], render: (r: Any) => typeof r[c] === 'number' ? fmtNum(r[c], 5) : String(r[c]) }))} />
                <p className="mt-2 text-[11px] text-dim">Top 50 records by score. Uncalibrated model probabilities are labelled as such; calibrated probabilities appear only if the experiment fitted a calibrator.</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
