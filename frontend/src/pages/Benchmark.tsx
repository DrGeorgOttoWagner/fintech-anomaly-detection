import { Trophy } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Curves, GroupedBars, SERIES, type CurveSeries } from '../components/charts'
import { Badge, Callout, Card, DataTable, EmptyState, ErrorBox, NeedSchema, PageHeader, Segmented, Select, Spinner } from '../components/ui'
import { api, type Any } from '../lib/api'
import { fmtInt, fmtMetric, fmtNum, IMBALANCE_LABEL } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

const OBJECTIVES = [
  { value: 'pr_auc', label: 'PR-AUC' },
  { value: 'f1', label: 'F1' },
  { value: 'recall', label: 'Recall' },
  { value: 'precision', label: 'Precision' },
  { value: 'roc_auc', label: 'ROC-AUC' },
  { value: 'mcc', label: 'MCC' },
  { value: 'balanced_accuracy', label: 'Balanced accuracy' },
]

function useRunDetails(ids: string[]) {
  const [details, setDetails] = useState<Record<string, Any>>({})
  useEffect(() => {
    let alive = true
    ids.filter((id) => !details[id]).forEach((id) => api.get(`/api/runs/${id}`).then((d) => alive && setDetails((cur) => ({ ...cur, [id]: d }))).catch(() => {}))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids.join()])
  return details
}

export default function Benchmark() {
  const { dataset, schema } = useSession()
  const [params, setParams] = useSearchParams()
  const [objective, setObjective] = useState('pr_auc')
  const [partition, setPartition] = useState<'validation' | 'test'>('validation')
  const lb = useApi<Any>(dataset && schema ? `/api/datasets/${dataset.id}/leaderboard?objective=${objective}&h=${schema.hash}` : null, { cache: false })
  const groups: Any[] = lb.data?.dataset_id === dataset?.id ? lb.data?.groups ?? [] : []
  const expParam = params.get('experiment')
  const groupIdx = Math.max(0, groups.findIndex((g) => g.experiments.some((e: Any) => e.id === expParam)))
  const group = groups[groupIdx]
  const [chosen, setChosen] = useState<string[]>([])
  const runs: Any[] = group?.runs ?? []
  const compareIds = chosen.length ? chosen : runs.slice(0, 5).map((r) => r.run_id)
  const details = useRunDetails(compareIds)

  const curves = useMemo(() => {
    const pr: CurveSeries[] = [], roc: CurveSeries[] = []
    compareIds.forEach((id, i) => {
      const d = details[id]
      const c = d?.detail?.curves?.[partition]
      if (!c?.pr) return
      const name = `${d.algorithm_name} · ${IMBALANCE_LABEL[d.imbalance]}`
      pr.push({ name, color: SERIES[i % SERIES.length], points: c.pr.recall.map((x: number, j: number) => ({ x, y: c.pr.precision[j] })) })
      roc.push({ name, color: SERIES[i % SERIES.length], points: c.roc.fpr.map((x: number, j: number) => ({ x, y: c.roc.tpr[j] })) })
    })
    return { pr, roc }
  }, [details, compareIds, partition])

  if (!dataset || !schema) return <><PageHeader eyebrow="Step 5 · Compare" title="ML Benchmark" /><NeedSchema hasDataset={!!dataset} /></>
  const best = runs.find((r) => r.run_id === group?.best_run_id)
  const metric = (r: Any, k: string) => r[partition]?.[k]

  return (
    <div>
      <PageHeader eyebrow="Step 5 · Compare" title="ML Benchmark"
        description="Leaderboard of completed runs on the active dataset and schema. Runs are grouped by evaluation protocol (identical split indices), so every model in a table was scored on exactly the same records." />
      {lb.error && <ErrorBox error={lb.error} />}
      {!lb.data ? <Spinner /> : groups.length === 0 ? (
        <EmptyState icon={<Trophy className="size-7" />} title="No completed runs yet" action={<Link to="/studio" className="text-sm text-accent hover:underline">Open the Experiment Studio →</Link>}>Run a Quick Benchmark to populate the leaderboard. Scores appear only after models have actually been trained and evaluated.</EmptyState>
      ) : (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <Select value={String(groupIdx)} onChange={(v) => setParams({ experiment: groups[Number(v)].experiments[0].id })} ariaLabel="Evaluation protocol"
              options={groups.map((g, i) => ({ value: String(i), label: `Protocol ${i + 1} · ${g.split.method} split · ${g.runs.length} runs · ${g.experiments.map((e: Any) => e.name).join(', ').slice(0, 60)}` }))} className="max-w-[520px]" />
            <span className="text-xs text-dim">Rank by</span>
            <Select value={objective} onChange={setObjective} options={OBJECTIVES} ariaLabel="Objective" />
            <Segmented value={partition} onChange={setPartition} options={[{ value: 'validation', label: 'Validation (selection)' }, { value: 'test', label: 'Test (final estimate)' }]} />
          </div>
          <Callout tone={partition === 'test' ? 'warn' : 'info'} title="Selection basis">
            {lb.data.selection_basis}{partition === 'test' && ' You are viewing test metrics: do not use them to pick a model.'}
          </Callout>
          {best && (
            <div className="grid gap-3 md:grid-cols-[1.4fr_repeat(4,1fr)]">
              <div className="rounded-xl border border-accent/40 bg-accent/[0.06] px-4 py-3">
                <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-accent"><Trophy className="size-3.5" /> Best by validation {lb.data.objective_label}</div>
                <div className="mt-1 font-display text-lg font-semibold">{best.algorithm_name}</div>
                <div className="text-xs text-muted">{IMBALANCE_LABEL[best.imbalance]} · {best.experiment_name}</div>
              </div>
              {(['pr_auc', 'recall', 'precision', 'f1'] as const).map((k) => (
                <div key={k} className="rounded-xl border border-line bg-panel-2/70 px-4 py-3">
                  <div className="text-[11px] uppercase tracking-wider text-dim">{k.replace('_', '-')} · val / test</div>
                  <div className="num mt-1 text-lg font-semibold">{fmtMetric(best.validation[k])}</div>
                  <div className="num text-xs text-muted">test {fmtMetric(best.test[k])}</div>
                </div>
              ))}
            </div>
          )}
          <Card title="Leaderboard" subtitle={`${runs.length} runs · split ${group.split.method} · train ${fmtInt(group.split.partitions.train.n)} / val ${fmtInt(group.split.partitions.validation.n)} / test ${fmtInt(group.split.partitions.test.n)} records · tick rows to compare`} pad={false}>
            <div className="p-4">
              <DataTable<Any>
                rows={runs}
                rowKey={(r) => r.run_id}
                initialSort={{ key: objective, dir: 'desc' }}
                dense
                columns={[
                  { key: 'sel', header: '', sortable: false, render: (r) => <input type="checkbox" aria-label={`Compare ${r.algorithm_name}`} checked={compareIds.includes(r.run_id)} onChange={(e) => setChosen(e.target.checked ? [...compareIds, r.run_id] : compareIds.filter((x) => x !== r.run_id))} /> },
                  { key: 'algo', header: 'Algorithm', value: (r) => r.algorithm_name, render: (r) => <div><div className="flex items-center gap-1.5 font-medium">{r.algorithm_name}{r.run_id === group.best_run_id && <Trophy className="size-3.5 text-accent" />}</div><div className="text-[11px] text-dim">{r.family} · {r.score_type.replace('_', ' ')}{r.mode === 'hpo' ? ' · tuned' : ''}</div></div> },
                  { key: 'imb', header: 'Imbalance', value: (r) => r.imbalance, render: (r) => <Badge>{IMBALANCE_LABEL[r.imbalance]}</Badge> },
                  { key: 'precision', header: 'Precision', align: 'right', value: (r) => metric(r, 'precision'), render: (r) => fmtMetric(metric(r, 'precision')) },
                  { key: 'recall', header: 'Recall', align: 'right', value: (r) => metric(r, 'recall'), render: (r) => fmtMetric(metric(r, 'recall')) },
                  { key: 'f1', header: 'F1', align: 'right', value: (r) => metric(r, 'f1'), render: (r) => fmtMetric(metric(r, 'f1')) },
                  { key: 'pr_auc', header: 'PR-AUC', align: 'right', value: (r) => metric(r, 'pr_auc'), render: (r) => <span className="text-accent">{fmtMetric(metric(r, 'pr_auc'))}</span> },
                  { key: 'roc_auc', header: 'ROC-AUC', align: 'right', value: (r) => metric(r, 'roc_auc'), render: (r) => fmtMetric(metric(r, 'roc_auc')) },
                  { key: 'mcc', header: 'MCC', align: 'right', value: (r) => metric(r, 'mcc'), render: (r) => fmtMetric(metric(r, 'mcc')) },
                  { key: 'balanced_accuracy', header: 'Bal. acc.', align: 'right', value: (r) => metric(r, 'balanced_accuracy'), render: (r) => fmtMetric(metric(r, 'balanced_accuracy')) },
                  { key: 'cm', header: 'TP / FP / FN', align: 'right', sortable: false, render: (r) => { const c = r[partition].confusion; return <span>{fmtInt(c.tp)} / {fmtInt(c.fp)} / {fmtInt(c.fn)}</span> } },
                  { key: 'train', header: 'Train', align: 'right', value: (r) => r.train_seconds, render: (r) => `${fmtNum(r.train_seconds, 1)} s` },
                  { key: 'inf', header: 'ms / 1k', align: 'right', value: (r) => r.inference_ms_per_1k, render: (r) => fmtNum(r.inference_ms_per_1k, 1) },
                  { key: 'open', header: '', sortable: false, render: (r) => <Link to={`/evaluation?run=${r.run_id}`} className="text-xs text-accent hover:underline">evaluate →</Link> },
                ]}
              />
              <p className="mt-2 text-[11px] text-dim">Threshold metrics use each run's validation-frozen threshold. Undefined values (e.g. precision with zero alerts) are shown as n/a.</p>
            </div>
          </Card>
          <div className="grid gap-6 xl:grid-cols-2">
            <Card title={`Precision–recall curves · ${partition}`} subtitle="Selected runs (first five by default)">
              {curves.pr.length ? <Curves series={curves.pr} xLabel="Recall" yLabel="Precision" baseline={group.split.partitions[partition].prevalence} /> : <Spinner />}
            </Card>
            <Card title={`ROC curves · ${partition}`}>
              {curves.roc.length ? <Curves series={curves.roc} xLabel="False positive rate" yLabel="True positive rate" diagonal /> : <Spinner />}
            </Card>
          </div>
          <Card title="Metric comparison" subtitle={`${partition} partition`}>
            <GroupedBars data={runs.slice(0, 16).map((r) => ({ name: `${r.algorithm_name.split(' ')[0]} · ${r.imbalance}`, pr_auc: r[partition].pr_auc, recall: r[partition].recall, precision: r[partition].precision, f1: r[partition].f1 }))}
              keys={[{ key: 'pr_auc', name: 'PR-AUC', color: SERIES[0] }, { key: 'recall', name: 'Recall', color: '#fb7185' }, { key: 'precision', name: 'Precision', color: SERIES[3] }, { key: 'f1', name: 'F1', color: SERIES[1] }]} height={320} />
          </Card>
        </div>
      )}
    </div>
  )
}
