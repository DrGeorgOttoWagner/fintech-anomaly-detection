import { Gauge } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { BarList, C, ConfusionMatrix, Curves, ThresholdChart } from '../components/charts'
import { Badge, Button, Callout, Card, EmptyState, ErrorBox, Field, KeyValue, NeedSchema, NumberInput, PageHeader, Segmented, Select, Spinner, Stat, Tabs, Toggle } from '../components/ui'
import { api, type Any, type Experiment, type Metrics } from '../lib/api'
import { fmtInt, fmtMetric, fmtNum, fmtPct, fmtSeconds, IMBALANCE_LABEL, SCORE_TYPE_LABEL } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

type Part = 'train' | 'validation' | 'test'

function MetricGrid({ m }: { m: Metrics }) {
  const items: [string, number | null][] = [
    ['Precision', m.precision], ['Recall', m.recall], ['F1', m.f1], ['PR-AUC', m.pr_auc], ['ROC-AUC', m.roc_auc],
    ['Specificity', m.specificity], ['Balanced acc.', m.balanced_accuracy], ['MCC', m.mcc], ['Accuracy', m.accuracy],
  ]
  return (
    <div>
      <div className="grid grid-cols-3 gap-2 md:grid-cols-5 xl:grid-cols-9">
        {items.map(([k, v]) => <Stat key={k} label={k} value={fmtMetric(v)} tone={k === 'PR-AUC' ? 'accent' : undefined} />)}
      </div>
      <div className="mt-2 text-[11px] text-dim">
        n = {fmtInt(m.n)} · positives {fmtInt(m.n_positive)} · alerts {fmtInt(m.alerts)} ({fmtPct(m.alert_rate, 3)}) · threshold {fmtNum(m.threshold, 5)}
        {m.pr_auc_baseline != null && <> · PR-AUC of a random ranking = {fmtNum(m.pr_auc_baseline, 5)}</>}
        {m.undefined.length > 0 && <> · undefined: {m.undefined.join('; ')}</>}
      </div>
    </div>
  )
}

function ThresholdStudio({ run, posName }: { run: Any; posName: string }) {
  const [showTest, setShowTest] = useState(false)
  const part = showTest ? 'test' : 'validation'
  const sweep = run.detail.sweep[part]
  const pts = useMemo(() => [...sweep.points].sort((a: Any, b: Any) => a.t - b.t).map((p: Any) => {
    const prec = p.tp + p.fp ? p.tp / (p.tp + p.fp) : null
    const rec = p.tp + p.fn ? p.tp / (p.tp + p.fn) : null
    const f1 = prec != null && rec != null && prec + rec ? (2 * prec * rec) / (prec + rec) : 0
    return { ...p, precision: prec, recall: rec, f1, alerts: p.tp + p.fp }
  }), [sweep])
  const frozenIdx = Math.max(0, pts.findIndex((p: Any) => p.t >= run.threshold))
  const [idx, setIdx] = useState(frozenIdx)
  useEffect(() => setIdx(frozenIdx), [frozenIdx, part])
  const p = pts[Math.min(idx, pts.length - 1)]
  const frozen = run.metrics[part]
  return (
    <Card title="Decision Threshold Studio" subtitle="Move the threshold to see the trade-off between missed cases and false alarms. The frozen threshold was chosen on validation before the test partition was scored."
      actions={<Toggle checked={showTest} onChange={setShowTest} label="Exploratory test view" />}>
      {showTest && <div className="mb-4"><Callout tone="warn" title="Exploratory, post-hoc view">Changing the threshold while looking at test results is not an unbiased evaluation. The reported test estimate is the one at the frozen threshold.</Callout></div>}
      <div className="grid gap-6 xl:grid-cols-[1.3fr_1fr]">
        <div>
          <ThresholdChart data={pts} current={p.t} frozen={run.threshold} />
          <input type="range" min={0} max={pts.length - 1} value={idx} onChange={(e) => setIdx(Number(e.target.value))} className="mt-3 w-full" aria-label="Threshold" data-testid="threshold-slider" />
          <div className="mt-1 flex justify-between text-[11px] text-dim"><span>more alerts ←</span><span className="num">t = {fmtNum(p.t, 5)} ({SCORE_TYPE_LABEL[run.score_type]})</span><span>→ fewer alerts</span></div>
        </div>
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2" data-testid="threshold-metrics">
            <Stat label="Precision" value={fmtMetric(p.precision)} />
            <Stat label="Recall" value={fmtMetric(p.recall)} tone="pos" />
            <Stat label="F1" value={fmtMetric(p.f1)} />
            <Stat label="Alerts" value={fmtInt(p.alerts)} />
            <Stat label="False positives" value={fmtInt(p.fp)} tone="warn" />
            <Stat label={`Missed ${posName.toLowerCase()}`} value={fmtInt(p.fn)} tone="pos" />
          </div>
          <div className="rounded-lg border border-line bg-panel-2/60 p-3 text-xs text-muted">
            At the <b className="text-warn">frozen</b> threshold ({fmtNum(run.threshold, 5)}; {run.threshold_basis}) on {part}: precision {fmtMetric(frozen.precision)}, recall {fmtMetric(frozen.recall)}, {fmtInt(frozen.confusion.fp)} false alarms, {fmtInt(frozen.confusion.fn)} missed.
            <Button size="sm" variant="ghost" className="ml-1" onClick={() => setIdx(frozenIdx)}>reset</Button>
          </div>
          <p className="text-[11px] leading-relaxed text-dim">Lowering the threshold catches more {posName.toLowerCase()} cases (higher recall) at the cost of more false alarms that must be reviewed; raising it does the opposite. The right operating point depends on the relative cost of each error — see the impact simulator.</p>
        </div>
      </div>
    </Card>
  )
}

function CostSimulator({ run, posName, amountColumn }: { run: Any; posName: string; amountColumn: string | null }) {
  const [a, setA] = useState({ missed_positive_cost: 100, false_positive_cost: 5, review_cost: 2, use_amount_for_missed: false })
  const [res, setRes] = useState<Any>(null)
  const [err, setErr] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const runIt = async () => {
    setBusy(true); setErr(null)
    try { setRes(await api.post(`/api/runs/${run.id}/costs`, a)) } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  const curve = res?.validation_curve ? [...res.validation_curve].sort((x: Any, y: Any) => x.t - y.t) : []
  const maxCost = Math.max(...curve.map((c: Any) => c.cost), 1)
  return (
    <Card title="Business impact simulator" subtitle="Hypothetical: all costs below are your assumptions — none are observed in the dataset.">
      <div className="grid gap-3 md:grid-cols-4">
        <Field label={`Cost per missed ${posName.toLowerCase()}`}><NumberInput value={a.missed_positive_cost} min={0} onChange={(v) => setA({ ...a, missed_positive_cost: Math.max(0, v || 0) })} /></Field>
        <Field label="Cost per false alarm"><NumberInput value={a.false_positive_cost} min={0} onChange={(v) => setA({ ...a, false_positive_cost: Math.max(0, v || 0) })} /></Field>
        <Field label="Manual review cost per alert"><NumberInput value={a.review_cost} min={0} onChange={(v) => setA({ ...a, review_cost: Math.max(0, v || 0) })} /></Field>
        <div className="flex items-end"><Button variant="primary" onClick={runIt} loading={busy} className="w-full">Simulate</Button></div>
      </div>
      {amountColumn && <div className="mt-3"><Toggle checked={a.use_amount_for_missed} onChange={(v) => setA({ ...a, use_amount_for_missed: v })} label={`Use each missed record's '${amountColumn}' as its loss`} hint="Uses observed transaction amounts instead of a flat cost per missed case." /></div>}
      <ErrorBox error={err} />
      {res?.available && (
        <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_1fr]">
          <div>
            <div className="mb-2 text-xs text-muted">Total hypothetical cost on validation by threshold (cost-optimal t = <span className="num text-text">{fmtNum(res.cost_optimal_threshold_validation, 5)}</span>)</div>
            <svg viewBox="0 0 400 140" className="w-full rounded-lg border border-line bg-panel-2/50">
              <polyline fill="none" stroke={C.accent} strokeWidth={1.6} points={curve.map((c: Any, i: number) => `${(400 * i) / Math.max(1, curve.length - 1)},${132 - (124 * c.cost) / maxCost}`).join(' ')} />
            </svg>
          </div>
          <div className="space-y-2 text-sm">
            {(['validation', 'test'] as const).map((p) => (
              <div key={p} className="rounded-lg border border-line bg-panel-2/60 p-3">
                <div className="mb-1 text-xs font-semibold capitalize text-muted">{p}</div>
                <KeyValue items={[
                  ['No detection', <span className="num">{fmtNum(res[p].no_detection_cost, 0)}</span>],
                  ['Frozen threshold', <span className="num">{fmtNum(res[p].frozen_threshold.cost, 0)} · {fmtInt(res[p].frozen_threshold.alerts)} alerts · {fmtInt(res[p].frozen_threshold.fn)} missed</span>],
                  ['Cost-optimal (from validation)', <span className="num text-accent">{fmtNum((res[p].cost_optimal ?? res[p].cost_optimal_from_validation)?.cost, 0)} · {fmtInt((res[p].cost_optimal ?? res[p].cost_optimal_from_validation)?.alerts)} alerts</span>],
                ]} />
              </div>
            ))}
            <p className="text-[11px] text-dim">{res.disclaimer} {res.note}</p>
          </div>
        </div>
      )}
    </Card>
  )
}

export default function Evaluation() {
  const { dataset, schema } = useSession()
  const [params, setParams] = useSearchParams()
  const exps = useApi<Experiment[]>(dataset ? `/api/experiments?dataset_id=${dataset.id}` : null, { cache: false })
  const completedExps = (exps.data ?? []).filter((e) => e.status.startsWith('completed'))
  const expId = params.get('experiment') ?? completedExps[0]?.id ?? null
  const exp = useApi<Experiment>(expId ? `/api/experiments/${expId}` : null)
  const runs = (exp.data?.runs ?? []).filter((r) => r.status === 'completed')
  const runId = params.get('run') ?? runs[0]?.id ?? null
  const run = useApi<Any>(runId ? `/api/runs/${runId}` : null)
  const [part, setPart] = useState<Part>('test')
  const [tab, setTab] = useState<'metrics' | 'threshold' | 'importance' | 'impact'>('metrics')

  if (!dataset || !schema) return <><PageHeader eyebrow="Step 5 · Evaluate" title="Model Evaluation" /><NeedSchema hasDataset={!!dataset} /></>
  const r = run.data && run.data.dataset_id === dataset.id ? run.data : null
  const pos = schema.config.positive_name, neg = schema.config.negative_name

  return (
    <div>
      <PageHeader eyebrow="Step 5 · Evaluate" title="Model Evaluation" description="Detailed results of one trained model: training (in-sample), validation (threshold selection) and final test metrics, curves, importance and threshold trade-offs." />
      {exps.data && completedExps.length === 0 ? (
        <EmptyState icon={<Gauge className="size-7" />} title="No evaluated models yet" action={<Link to="/studio"><Button variant="primary">Run an experiment</Button></Link>}>Model metrics are computed from actual predictions after training.</EmptyState>
      ) : (
        <>
          <div className="mb-5 flex flex-wrap gap-3">
            <Select value={expId ?? ''} onChange={(v) => setParams({ experiment: v })} options={completedExps.map((e) => ({ value: e.id, label: e.name }))} ariaLabel="Experiment" className="max-w-[420px]" />
            <Select value={runId ?? ''} onChange={(v) => setParams({ experiment: expId ?? '', run: v })} options={runs.map((x) => ({ value: x.id, label: `${x.algorithm_name} · ${IMBALANCE_LABEL[x.imbalance]}` }))} ariaLabel="Run" />
          </div>
          {run.error && <ErrorBox error={run.error} />}
          {!r ? <Spinner /> : (
            <div className="space-y-6">
              <Card>
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <div className="font-display text-xl font-semibold">{r.algorithm_name}</div>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      <Badge>{r.family}</Badge><Badge>{IMBALANCE_LABEL[r.imbalance]}</Badge>
                      <Badge tone={r.score_type === 'probability' ? 'accent' : 'warn'}>{SCORE_TYPE_LABEL[r.score_type]}</Badge>
                      {r.cv && <Badge tone="accent">tuned · CV {r.cv.scoring} {fmtNum(r.cv.mean_score, 4)} ± {fmtNum(r.cv.std_score, 4)}</Badge>}
                    </div>
                  </div>
                  <div className="text-right text-xs text-muted">
                    <div>Frozen threshold <span className="num text-text">{fmtNum(r.threshold, 5)}</span> — {r.threshold_basis}</div>
                    <div>Model default {fmtNum(r.default_threshold, 4)} · train {fmtSeconds(r.train_seconds)} on {fmtInt(r.train_rows_used)} rows · {fmtNum(r.inference_ms_per_1k, 1)} ms / 1k rows</div>
                  </div>
                </div>
                {r.notes?.length > 0 && <ul className="mt-3 list-disc space-y-0.5 pl-5 text-xs text-dim">{r.notes.map((n: string) => <li key={n}>{n}</li>)}</ul>}
                {r.score_type === 'anomaly_score' && <p className="mt-2 text-xs text-warn">Anomaly scores rank records by unusualness; they are not probabilities and labels were not used to fit this detector.</p>}
              </Card>
              <Tabs value={tab} onChange={setTab} tabs={[{ value: 'metrics', label: 'Metrics & curves' }, { value: 'threshold', label: 'Threshold studio' }, { value: 'importance', label: 'Feature importance' }, { value: 'impact', label: 'Business impact' }]} />
              {tab === 'metrics' && (
                <>
                  <Card title="Metrics by partition" actions={<Segmented value={part} onChange={setPart} options={[{ value: 'train', label: 'Training (in-sample)' }, { value: 'validation', label: 'Validation' }, { value: 'test', label: 'Final test' }]} />}>
                    {part === 'train' && <div className="mb-3"><Callout tone="warn">Training metrics are computed on the data the model was fitted on and are optimistic. They are shown to diagnose overfitting only.</Callout></div>}
                    {part === 'test' && <div className="mb-3"><Callout tone="info">Final test estimate at the threshold frozen on validation. The test partition was not used for fitting, tuning or threshold selection.</Callout></div>}
                    <MetricGrid m={r.metrics[part]} />
                    {r.cv && part !== 'train' && <p className="mt-2 text-xs text-muted">Cross-validation (training partition): {r.cv.scoring} {fmtNum(r.cv.mean_score, 4)} ± {fmtNum(r.cv.std_score, 4)} with params {JSON.stringify(r.cv.params)}</p>}
                  </Card>
                  <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
                    <Card title={`Confusion matrix · ${part}`}><ConfusionMatrix cm={r.metrics[part].confusion} posName={pos} negName={neg} /></Card>
                    {part !== 'train' && r.detail.curves[part]?.pr ? (
                      <div className="grid gap-6 lg:grid-cols-2">
                        <Card title={`Precision–recall · ${part}`}><Curves series={[{ name: r.algorithm_name, color: C.accent, points: r.detail.curves[part].pr.recall.map((x: number, i: number) => ({ x, y: r.detail.curves[part].pr.precision[i] })) }]} xLabel="Recall" yLabel="Precision" height={260} baseline={r.metrics[part].pr_auc_baseline} /></Card>
                        <Card title={`ROC · ${part}`}><Curves series={[{ name: r.algorithm_name, color: C.accent2, points: r.detail.curves[part].roc.fpr.map((x: number, i: number) => ({ x, y: r.detail.curves[part].roc.tpr[i] })) }]} xLabel="False positive rate" yLabel="True positive rate" height={260} diagonal /></Card>
                      </div>
                    ) : <Card title="Curves">{part === 'train' ? 'Curves are shown for validation and test.' : 'Curves unavailable (single-class partition).'}</Card>}
                  </div>
                  {part !== 'train' && (
                    <Card title={`Calibration · ${part}`} subtitle={r.score_type === 'probability' ? 'Mean predicted probability vs. observed positive rate (quantile bins).' : undefined}>
                      {r.detail.curves[part]?.calibration ? (
                        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
                          <Curves series={[
                            { name: 'Model', color: C.warn, points: r.detail.curves[part].calibration.mean_predicted.map((x: number, i: number) => ({ x, y: r.detail.curves[part].calibration.fraction_positive[i] })) },
                            ...(r.detail.calibration?.curve && part === 'test' ? [{ name: 'Platt-calibrated', color: C.accent, points: r.detail.calibration.curve.mean_predicted.map((x: number, i: number) => ({ x, y: r.detail.calibration.curve.fraction_positive[i] })) }] : []),
                          ]} xLabel="Mean predicted probability" yLabel="Observed positive rate" height={240} diagonal />
                          <div className="text-xs text-muted">Brier score <span className="num text-text">{fmtNum(r.detail.curves[part].calibration.brier, 5)}</span>{r.detail.calibration && part === 'test' && <> · Platt-calibrated Brier <span className="num text-accent">{fmtNum(r.detail.calibration.test_brier, 5)}</span></>}<p className="mt-2 text-dim">{r.detail.curves[part].calibration.note}</p></div>
                        </div>
                      ) : <p className="text-sm text-muted">Not applicable: this model emits {SCORE_TYPE_LABEL[r.score_type].toLowerCase()}s, which are not probabilities.</p>}
                    </Card>
                  )}
                </>
              )}
              {tab === 'threshold' && <ThresholdStudio run={r} posName={pos} />}
              {tab === 'importance' && (
                <div className="grid gap-6 xl:grid-cols-2">
                  <Card title="Model-native importance" subtitle={r.detail.importance.native?.kind ?? 'Not provided by this model type.'}>
                    {r.detail.importance.native ? <BarList items={r.detail.importance.native.features.slice(0, 20).map((f: Any) => ({ label: f.feature, value: f.value }))} signed /> : <p className="text-sm text-muted">This estimator does not expose coefficients or impurity importances.</p>}
                  </Card>
                  <Card title="Permutation importance" subtitle={r.detail.importance.permutation?.kind ?? r.detail.importance.permutation_skipped ?? 'Disabled for this experiment.'}>
                    {r.detail.importance.permutation ? <BarList items={r.detail.importance.permutation.features.slice(0, 20).map((f: Any) => ({ label: f.feature, value: f.mean, err: f.std }))} color={C.pos} /> : <p className="text-sm text-muted">Enable “Permutation importance” in the Experiment Studio to compute it.</p>}
                    <p className="mt-3 text-[11px] text-dim">Importance measures reliance of this model on a feature, not a causal effect. Precomputed PCA components carry no interpretable business meaning.</p>
                  </Card>
                </div>
              )}
              {tab === 'impact' && <CostSimulator run={r} posName={pos} amountColumn={schema.config.amount_column} />}
              <div className="text-right"><Link to={`/anomalies?run=${r.id}`} className="text-sm text-accent hover:underline">Inspect individual records in the Anomaly Explorer →</Link></div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
