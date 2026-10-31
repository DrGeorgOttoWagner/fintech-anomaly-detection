import clsx from 'clsx'
import { CheckCircle2, CircleSlash, Clock, Cpu, Loader2, Play, Square, XCircle, Zap } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { SegmentedBar } from '../components/charts'
import { Badge, Button, Callout, Card, DataTable, ErrorBox, Field, NeedSchema, NumberInput, PageHeader, Segmented, Select, Spinner, Toggle } from '../components/ui'
import { api, type Any, type Envelope, type Experiment, type ModelInfo } from '../lib/api'
import { fmtDate, fmtInt, fmtPct, fmtSeconds, IMBALANCE_LABEL, SCORE_TYPE_LABEL } from '../lib/format'
import { useApi, useDebounced } from '../lib/hooks'
import { useSession } from '../lib/session'

type Parts = { train: number; validation: number; test: number }
const KEYS: (keyof Parts)[] = ['train', 'validation', 'test']
const PART_COLOR = { train: '#2dd4bf', validation: '#38bdf8', test: '#f5b84b' }
const PRESETS: Parts[] = [{ train: 80, validation: 10, test: 10 }, { train: 70, validation: 15, test: 15 }, { train: 60, validation: 20, test: 20 }]
const MIN_PART = 5
const MODE_LABEL: Record<string, string> = { quick: 'Quick benchmark', full: 'Full benchmark', hpo: 'Hyperparameter optimisation' }

/** Change one partition and redistribute the remainder proportionally so the total stays 100%. */
export function rebalance(p: Parts, key: keyof Parts, raw: number): Parts {
  const value = Math.max(MIN_PART, Math.min(100 - 2 * MIN_PART, Math.round(raw)))
  const others = KEYS.filter((k) => k !== key)
  const rest = 100 - value
  const oldSum = others.reduce((s, k) => s + p[k], 0)
  let a = oldSum ? Math.round((p[others[0]] / oldSum) * rest) : Math.round(rest / 2)
  a = Math.max(MIN_PART, Math.min(rest - MIN_PART, a))
  return { ...p, [key]: value, [others[0]]: a, [others[1]]: rest - a } as Parts
}

function SplitDesigner({ parts, setParts, method, setMethod, seed, setSeed, preview, canChrono, canGroup }: Any) {
  return (
    <Card title="Train / validation / test" subtitle="The test partition is reserved: it is never used for tuning, threshold selection or model selection.">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Segmented value={method} onChange={setMethod} options={[
          { value: 'stratified', label: 'Stratified' },
          { value: 'random', label: 'Random' },
          { value: 'chronological', label: 'Chronological', disabled: !canChrono, title: canChrono ? undefined : 'Requires a time column' },
          { value: 'group', label: 'Group-aware', disabled: !canGroup, title: canGroup ? undefined : 'Requires a group / entity ID column' },
        ]} />
        <div className="flex items-center gap-1.5">
          {PRESETS.map((p) => (
            <Button key={p.train} size="sm" variant={p.train === parts.train && p.validation === parts.validation ? 'primary' : 'secondary'} onClick={() => setParts(p)}>
              {p.train}/{p.validation}/{p.test}
            </Button>
          ))}
        </div>
      </div>
      <SegmentedBar parts={KEYS.map((k) => ({ label: k === 'validation' ? 'Val' : k[0].toUpperCase() + k.slice(1), value: parts[k], color: PART_COLOR[k] }))} />
      <div className="mt-5 grid gap-4 md:grid-cols-3">
        {KEYS.map((k) => {
          const p = preview?.partitions?.[k]
          return (
            <div key={k} className="rounded-lg border border-line bg-panel-2/60 p-3.5">
              <div className="flex items-baseline justify-between">
                <span className="flex items-center gap-1.5 text-sm font-medium capitalize"><span className="size-2 rounded-full" style={{ background: PART_COLOR[k] }} />{k}</span>
                <span className="num text-lg font-semibold" data-testid={`pct-${k}`}>{parts[k]}%</span>
              </div>
              <input type="range" min={MIN_PART} max={100 - 2 * MIN_PART} value={parts[k]} aria-label={`${k} percentage`} data-testid={`slider-${k}`} onChange={(e) => setParts(rebalance(parts, k, Number(e.target.value)))} className="mt-2 w-full" />
              {p ? (
                <div className="num mt-2 grid grid-cols-2 gap-x-2 text-xs text-muted">
                  <span>Records</span><span className="text-right text-text">{fmtInt(p.n)}</span>
                  <span>Positives</span><span className={clsx('text-right', p.n_positive < 20 ? 'text-warn' : 'text-pos')}>{fmtInt(p.n_positive)}</span>
                  <span>Prevalence</span><span className="text-right">{fmtPct(p.prevalence, 3)}</span>
                </div>
              ) : <div className="mt-2 h-12" />}
            </div>
          )
        })}
      </div>
      <div className="mt-4 flex flex-wrap items-end gap-4">
        <div className="w-32"><Field label="Random seed"><NumberInput value={seed} onChange={(v) => setSeed(Number.isFinite(v) ? v : 0)} ariaLabel="Random seed" /></Field></div>
        <div className="flex-1 text-xs text-dim">
          {method === 'chronological' && 'Order is preserved: train precedes validation precedes test. No stratification — class shares may drift between partitions.'}
          {method === 'stratified' && 'Each partition keeps the overall class prevalence.'}
          {method === 'group' && 'All records of an entity stay in one partition; partition sizes are approximate.'}
          {method === 'random' && 'Unstratified random split — prevalence varies by chance.'}
        </div>
      </div>
      <div className="mt-3 space-y-2">
        {preview?.errors?.map((e: string) => <Callout key={e} tone="error">{e}</Callout>)}
        {preview?.warnings?.map((e: string) => <Callout key={e} tone="warn">{e}</Callout>)}
        {preview?.notes?.map((e: string) => <p key={e} className="text-xs text-dim">{e}</p>)}
      </div>
    </Card>
  )
}

function ModelPicker({ models, selected, setSelected, mode }: { models: ModelInfo[]; selected: string[]; setSelected: (v: string[]) => void; mode: string }) {
  const families = [...new Set(models.map((m) => m.family))]
  return (
    <div className="space-y-4">
      {families.map((f) => (
        <div key={f}>
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-dim">{f}</div>
          <div className="grid gap-2 sm:grid-cols-2 2xl:grid-cols-3">
            {models.filter((m) => m.family === f).map((m) => {
              const on = selected.includes(m.key)
              return (
                <button key={m.key} disabled={!m.available} data-testid={`model-${m.key}`}
                  onClick={() => mode !== 'quick' && m.available && setSelected(on ? selected.filter((k) => k !== m.key) : [...selected, m.key])}
                  className={clsx('rounded-lg border p-3 text-left transition-colors', on ? 'border-accent/60 bg-accent/[0.07]' : 'border-line bg-panel-2/50 hover:border-line-2', (!m.available || mode === 'quick') && 'cursor-default', !m.available && 'opacity-50')}>
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-[13px] font-medium">{m.name}</span>
                    {on ? <CheckCircle2 className="size-4 shrink-0 text-accent" /> : <span className="size-4 shrink-0 rounded-full border border-line-2" />}
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    <Badge tone={m.score_type === 'probability' ? 'accent' : 'default'}>{m.score_type.replace('_', ' ')}</Badge>
                    {m.cost !== 'low' && <Badge tone={m.cost === 'high' ? 'warn' : 'default'}>{m.cost} cost</Badge>}
                    {m.max_train_rows && <Badge tone="dim">cap {fmtInt(m.max_train_rows)} rows</Badge>}
                    {mode === 'hpo' && !m.hpo && <Badge tone="dim">no search</Badge>}
                    {!m.available && <Badge tone="bad" title={m.unavailable_reason ?? ''}>unavailable</Badge>}
                  </div>
                  <div className="mt-1 text-[11px] text-dim">{m.library}</div>
                </button>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

function Progress({ expId, onDone }: { expId: string; onDone: () => void }) {
  const { data: exp, reload } = useApi<Experiment>(`/api/experiments/${expId}`, { cache: false, poll: 1000 })
  const [cancelling, setCancelling] = useState(false)
  const active = exp && ['created', 'queued', 'running'].includes(exp.status)
  useEffect(() => {
    if (exp && !active) onDone()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exp?.status])
  if (!exp) return <Spinner />
  const done = exp.runs.filter((r) => ['completed', 'failed', 'skipped', 'cancelled'].includes(r.status)).length
  const icon = (s: string) =>
    s === 'completed' ? <CheckCircle2 className="size-4 text-ok" /> : s === 'running' ? <Loader2 className="size-4 animate-spin text-accent" /> : s === 'failed' ? <XCircle className="size-4 text-bad" /> : s === 'skipped' ? <CircleSlash className="size-4 text-dim" /> : <Clock className="size-4 text-dim" />
  return (
    <Card title={<span className="flex items-center gap-2">{exp.name} <Badge tone={exp.status === 'completed' ? 'ok' : exp.status.includes('fail') || exp.status === 'cancelled' ? 'bad' : 'accent'}>{exp.status}</Badge></span>}
      subtitle={<span>{MODE_LABEL[exp.mode] ?? exp.mode} · {exp.progress?.stage ?? ''}{exp.progress?.current_algorithm ? ` · ${exp.progress.current_algorithm}` : ''} · elapsed {fmtSeconds(exp.progress?.elapsed_seconds)}</span>}
      actions={active ? <Button variant="danger" size="sm" loading={cancelling} onClick={async () => { setCancelling(true); try { await api.post(`/api/experiments/${expId}/cancel`) } finally { setCancelling(false); reload() } }}><Square className="size-3.5" />Cancel</Button>
        : <Link to={`/benchmark?experiment=${exp.id}`}><Button variant="primary" size="sm">Open leaderboard</Button></Link>}
      >
      <div className="mb-3 h-1.5 overflow-hidden rounded-full bg-panel-3"><div className="h-full bg-accent transition-[width]" style={{ width: `${(100 * done) / Math.max(1, exp.runs.length)}%` }} /></div>
      {exp.error && <div className="mb-3"><Callout tone="error">{exp.error}</Callout></div>}
      <div className="space-y-1" data-testid="run-status">
        {exp.runs.map((r) => (
          <div key={r.id} className="grid grid-cols-[20px_1fr_auto] items-center gap-2 rounded-md px-2 py-1.5 text-[13px] hover:bg-panel-2">
            {icon(r.status)}
            <div className="min-w-0">
              <span className="font-medium">{r.algorithm_name}</span> <span className="text-xs text-dim">· {IMBALANCE_LABEL[r.imbalance]}</span>
              {r.error && <div className="truncate text-[11px] text-dim" title={r.error}>{r.error}</div>}
            </div>
            <div className="num text-right text-xs text-muted">
              {r.status === 'completed' && r.metrics ? <>val PR-AUC <span className="text-text">{r.metrics.validation.pr_auc?.toFixed(4) ?? 'n/a'}</span> · {fmtSeconds(r.train_seconds)}</> : r.status}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

export default function Studio() {
  const { dataset, schema } = useSession()
  const models = useApi<{ models: ModelInfo[]; quick_set: string[] }>('/api/models')
  const strategies = useApi<Any[]>('/api/imbalance-strategies')
  const history = useApi<Experiment[]>(dataset ? `/api/experiments?dataset_id=${dataset.id}` : null, { cache: false })
  const cfg = schema?.config

  const [mode, setMode] = useState<'quick' | 'full' | 'hpo'>('quick')
  const [parts, setParts] = useState<Parts>({ train: 70, validation: 15, test: 15 })
  const [method, setMethod] = useState('stratified')
  const [seed, setSeed] = useState(42)
  const [algos, setAlgos] = useState<string[]>([])
  const [imbalance, setImbalance] = useState<string[]>(['none'])
  const [ratio, setRatio] = useState(1)
  const [prep, setPrep] = useState({ numeric_imputation: 'median', scaling: 'auto', log_amount: false, pca: false, pca_variance: 0.95, feature_selection: 'none', k_best: 20 })
  const [threshold, setThreshold] = useState<{ policy: string; value: number | null }>({ policy: 'max_f1', value: null })
  const [hpo, setHpo] = useState({ n_iter: 10, cv_folds: 3, scoring: 'average_precision', max_rows: 80000 })
  const [opts, setOpts] = useState({ permutation_importance: true, calibrate: false, resource_profile: 'safe', anomaly_training: 'unsupervised' })
  const [features, setFeatures] = useState<string[] | null>(null)
  const [overrides, setOverrides] = useState('{}')
  const [name, setName] = useState('')
  const [running, setRunning] = useState<string | null>(null)
  const [submitErr, setSubmitErr] = useState<unknown>(null)
  const [submitting, setSubmitting] = useState(false)

  const quickSet = models.data?.quick_set ?? []
  useEffect(() => {
    if (mode === 'quick' && quickSet.length) setAlgos(quickSet)
    if (mode === 'hpo') setAlgos((a) => a.filter((k) => models.data?.models.find((m) => m.key === k)?.hpo || models.data?.models.find((m) => m.key === k)?.kind === 'anomaly'))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, quickSet.join()])

  const splitBody = useMemo(() => ({ method, train: parts.train / 100, validation: parts.validation / 100, test: parts.test / 100, seed }), [method, parts, seed])
  const debounced = useDebounced(splitBody, 250)
  const [preview, setPreview] = useState<Any>(null)
  useEffect(() => {
    if (!dataset || !schema?.config.target) return
    let alive = true
    api.post<Envelope>(`/api/datasets/${dataset.id}/split/preview`, debounced).then((r) => alive && r.dataset_id === dataset.id && setPreview(r.result)).catch((e) => alive && setPreview({ errors: [e.message] }))
    return () => { alive = false }
  }, [debounced, dataset, schema])

  if (!dataset || !schema || !cfg) return <><PageHeader eyebrow="Step 4 · Experiment" title="Experiment Studio" /><NeedSchema hasDataset={!!dataset} /></>
  if (!cfg.target) return <><PageHeader title="Experiment Studio" /><Callout tone="warn">The confirmed schema has no target column. Supervised benchmarking requires a labelled target.</Callout></>

  const schemaFeatures = Object.entries(cfg.roles).filter(([, r]) => ['numeric', 'categorical', 'pca'].includes(r)).map(([c]) => c)
  const chosenFeatures = features ?? schemaFeatures
  let overridesParsed: Any = null
  try { overridesParsed = JSON.parse(overrides || '{}') } catch { overridesParsed = null }

  const submit = async () => {
    setSubmitting(true)
    setSubmitErr(null)
    try {
      const body = {
        dataset_id: dataset.id,
        config: {
          name: name || `${mode === 'quick' ? 'Quick benchmark' : mode === 'full' ? 'Full benchmark' : 'Hyperparameter optimisation'} · ${new Date().toLocaleString()}`,
          mode, split: splitBody, features: features, algorithms: algos, imbalance, resample_ratio: ratio,
          preprocessing: prep, threshold: { policy: threshold.policy, value: threshold.value }, hpo, ...opts,
          hyperparameters: overridesParsed ?? {},
        },
      }
      const exp = await api.post<Experiment>('/api/experiments', body)
      setRunning(exp.id)
      history.reload()
    } catch (e) {
      setSubmitErr(e)
    } finally {
      setSubmitting(false)
    }
  }

  const nRuns = algos.reduce((s, k) => s + (models.data?.models.find((m) => m.key === k)?.kind === 'anomaly' ? 1 : imbalance.length), 0)
  const canRun = preview?.valid && algos.length > 0 && imbalance.length > 0 && overridesParsed !== null && chosenFeatures.length > 0

  return (
    <div>
      <PageHeader eyebrow="Step 4 · Experiment" title="Experiment Studio" description="Configure one evaluation protocol and run several algorithms under it. Training runs in a separate local worker process; preprocessing and resampling are fitted on training data only." />
      <div className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_440px]">
        <div className="space-y-6">
          <Card title="Execution mode">
            <div className="grid gap-3 md:grid-cols-3">
              {([
                ['quick', 'Quick Benchmark', 'Dummy, Logistic Regression, Random Forest, XGBoost and Isolation Forest with safe defaults.', Zap],
                ['full', 'Full Benchmark', 'Any selection of available algorithms and imbalance strategies under the same protocol.', Cpu],
                ['hpo', 'Hyperparameter Optimisation', 'Bounded RandomizedSearchCV inside the training partition; validation/test never used for the search.', Play],
              ] as const).map(([k, t, d, Icon]) => (
                <button key={k} onClick={() => setMode(k)} data-testid={`mode-${k}`} className={clsx('rounded-xl border p-4 text-left transition-colors', mode === k ? 'border-accent/60 bg-accent/[0.07]' : 'border-line bg-panel-2/50 hover:border-line-2')}>
                  <Icon className={clsx('mb-2 size-5', mode === k ? 'text-accent' : 'text-dim')} />
                  <div className="font-medium">{t}</div>
                  <div className="mt-1 text-xs leading-relaxed text-muted">{d}</div>
                </button>
              ))}
            </div>
          </Card>
          <SplitDesigner parts={parts} setParts={setParts} method={method} setMethod={setMethod} seed={seed} setSeed={setSeed} preview={preview} canChrono={!!cfg.time_column} canGroup={!!cfg.group_column} />
          <Card title="Algorithms" subtitle={mode === 'quick' ? 'Quick mode uses a fixed, bounded set.' : 'Unavailable libraries are shown, never silently replaced.'}>
            {models.data ? <ModelPicker models={models.data.models} selected={algos} setSelected={setAlgos} mode={mode} /> : <Spinner />}
          </Card>
          <Card title="Class-imbalance strategies" subtitle="Each selected strategy is run for every supervised algorithm (controlled comparison). Applied to training data only.">
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
              {(strategies.data ?? []).map((s) => {
                const on = imbalance.includes(s.key)
                return (
                  <button key={s.key} data-testid={`imb-${s.key}`} onClick={() => setImbalance(on ? imbalance.filter((x) => x !== s.key) : [...imbalance, s.key])}
                    className={clsx('rounded-lg border p-3 text-left', on ? 'border-accent/60 bg-accent/[0.07]' : 'border-line bg-panel-2/50 hover:border-line-2')}>
                    <div className="flex items-center justify-between text-[13px] font-medium">{s.name}{on && <CheckCircle2 className="size-4 text-accent" />}</div>
                    <div className="mt-1 text-[11.5px] leading-snug text-muted">{s.description}</div>
                  </button>
                )
              })}
            </div>
            {imbalance.some((s) => ['undersample', 'oversample', 'smote'].includes(s)) && (
              <div className="mt-4 max-w-md">
                <Field label={`Target minority : majority ratio after resampling — ${ratio.toFixed(2)}`} hint="1.0 = balanced training set. Validation and test keep the original prevalence.">
                  <input type="range" min={0.05} max={1} step={0.05} value={ratio} onChange={(e) => setRatio(Number(e.target.value))} className="w-full" />
                </Field>
              </div>
            )}
          </Card>
          <Card title="Features" subtitle={`${chosenFeatures.length} of ${schemaFeatures.length} schema features selected`} actions={<><Button size="sm" variant="ghost" onClick={() => setFeatures(null)}>All</Button><Button size="sm" variant="ghost" onClick={() => setFeatures([])}>None</Button></>}>
            <div className="flex max-h-44 flex-wrap gap-1.5 overflow-auto">
              {schemaFeatures.map((f) => {
                const on = chosenFeatures.includes(f)
                return <button key={f} onClick={() => setFeatures(on ? chosenFeatures.filter((x) => x !== f) : [...chosenFeatures, f])} className={clsx('rounded-md border px-2 py-1 text-xs', on ? 'border-accent/40 bg-accent/10 text-text' : 'border-line text-dim')}>{f}</button>
              })}
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card title="Preprocessing" subtitle="Fitted on the training partition; persisted with the model.">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Scaling"><Select className="w-full" value={prep.scaling} onChange={(v) => setPrep({ ...prep, scaling: v })} options={[{ value: 'auto', label: 'Auto (per model)' }, { value: 'standard', label: 'Standard' }, { value: 'robust', label: 'Robust' }, { value: 'minmax', label: 'Min-max' }, { value: 'none', label: 'None' }]} /></Field>
              <Field label="Numeric imputation"><Select className="w-full" value={prep.numeric_imputation} onChange={(v) => setPrep({ ...prep, numeric_imputation: v })} options={[{ value: 'median', label: 'Median' }, { value: 'mean', label: 'Mean' }, { value: 'zero', label: 'Zero' }]} /></Field>
              <Field label="Feature selection"><Select className="w-full" value={prep.feature_selection} onChange={(v) => setPrep({ ...prep, feature_selection: v })} options={[{ value: 'none', label: 'None' }, { value: 'variance', label: 'Drop zero variance' }, { value: 'kbest', label: 'SelectKBest (ANOVA F)' }]} /></Field>
              {prep.feature_selection === 'kbest' && <Field label="k"><NumberInput value={prep.k_best} min={1} onChange={(v) => setPrep({ ...prep, k_best: Math.max(1, v || 1) })} /></Field>}
            </div>
            <div className="mt-4 space-y-3">
              <Toggle checked={prep.log_amount} disabled={!cfg.amount_column} onChange={(v) => setPrep({ ...prep, log_amount: v })} label={`log(1 + ${cfg.amount_column ?? 'amount'})`} hint="Compresses the heavy right tail of the amount column." />
              <Toggle checked={prep.pca} onChange={(v) => setPrep({ ...prep, pca: v })} label="Application PCA" hint="Fitted by FTAD on training data only — distinct from precomputed PCA columns." />
              {prep.pca && <Field label={`Retained variance ${(prep.pca_variance * 100).toFixed(0)}%`}><input type="range" min={0.5} max={0.99} step={0.01} value={prep.pca_variance} onChange={(e) => setPrep({ ...prep, pca_variance: Number(e.target.value) })} className="w-full" /></Field>}
            </div>
          </Card>
          <Card title="Decision threshold" subtitle="Chosen on validation, then frozen before the test partition is scored.">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Policy"><Select className="w-full" value={threshold.policy} onChange={(v) => setThreshold({ policy: v, value: v === 'target_recall' || v === 'target_precision' ? 0.8 : v === 'fixed' ? 0.5 : null })}
                options={[{ value: 'max_f1', label: 'Maximise F1' }, { value: 'target_recall', label: 'Target recall' }, { value: 'target_precision', label: 'Target precision' }, { value: 'default', label: 'Model default' }, { value: 'fixed', label: 'Fixed value' }]} /></Field>
              {threshold.value != null && <Field label="Value"><NumberInput value={threshold.value} step={0.05} onChange={(v) => setThreshold({ ...threshold, value: v })} /></Field>}
            </div>
          </Card>
          {mode === 'hpo' && (
            <Card title="Search settings" subtitle="RandomizedSearchCV over each model's declared search space.">
              <div className="grid grid-cols-2 gap-3">
                <Field label="Iterations"><NumberInput value={hpo.n_iter} min={1} max={60} onChange={(v) => setHpo({ ...hpo, n_iter: Math.max(1, Math.min(60, v || 1)) })} /></Field>
                <Field label="CV folds"><NumberInput value={hpo.cv_folds} min={2} max={10} onChange={(v) => setHpo({ ...hpo, cv_folds: Math.max(2, Math.min(10, v || 2)) })} /></Field>
                <Field label="Scoring"><Select className="w-full" value={hpo.scoring} onChange={(v) => setHpo({ ...hpo, scoring: v })} options={[{ value: 'average_precision', label: 'PR-AUC' }, { value: 'roc_auc', label: 'ROC-AUC' }, { value: 'f1', label: 'F1' }, { value: 'balanced_accuracy', label: 'Balanced acc.' }]} /></Field>
                <Field label="Search rows (cap)"><NumberInput value={hpo.max_rows} min={1000} step={10000} onChange={(v) => setHpo({ ...hpo, max_rows: Math.max(1000, v || 1000) })} /></Field>
              </div>
            </Card>
          )}
          <Card title="Options">
            <div className="space-y-3">
              <Toggle checked={opts.permutation_importance} onChange={(v) => setOpts({ ...opts, permutation_importance: v })} label="Permutation importance" hint="On validation (≤ 30k rows); skipped for high-cost models." />
              <Toggle checked={opts.calibrate} onChange={(v) => setOpts({ ...opts, calibrate: v })} label="Platt calibration" hint="Fits a calibrator on validation scores for calibrated probabilities at inference." />
              <Toggle checked={opts.resource_profile === 'extended'} onChange={(v) => setOpts({ ...opts, resource_profile: v ? 'extended' : 'safe' })} label="Extended resource caps" hint="Raises per-model training-row caps 4×." />
              <Toggle checked={opts.anomaly_training === 'normal_only'} onChange={(v) => setOpts({ ...opts, anomaly_training: v ? 'normal_only' : 'unsupervised' })} label="Fit anomaly detectors on negatives only" hint="Semi-supervised novelty detection instead of fully unsupervised." />
            </div>
            <details className="mt-4 group" open={overrides.trim() !== '{}' && overrides.trim() !== ''}>
              <summary className="cursor-pointer select-none text-xs font-medium text-muted hover:text-text">Advanced: hyperparameter overrides</summary>
              <div className="mt-3">
              <Field label="Hyperparameter overrides (JSON per algorithm key)" hint={<>e.g. {'{"rf": {"n_estimators": 400}, "xgb": {"max_depth": 4}}'}{overridesParsed === null && <span className="text-bad"> — invalid JSON</span>}</>}>
                <textarea value={overrides} onChange={(e) => setOverrides(e.target.value)} rows={3} className="num w-full rounded-lg border border-line-2 bg-panel-2 p-2 text-xs outline-none focus:border-accent" />
              </Field>
              </div>
            </details>
          </Card>
          <Card>
            <Field label="Experiment name (optional)"><input value={name} onChange={(e) => setName(e.target.value)} className="h-9 w-full rounded-lg border border-line-2 bg-panel-2 px-2.5 text-sm outline-none focus:border-accent" placeholder="e.g. Baseline comparison" /></Field>
            <div className="mt-4 flex items-center justify-between gap-3">
              <div className="text-xs text-muted"><span className="num text-text">{nRuns}</span> runs · {algos.length} algorithms × {imbalance.length} strategies</div>
              <Button variant="primary" onClick={submit} loading={submitting} disabled={!canRun} data-testid="run-experiment"><Play className="size-4" />Run experiment</Button>
            </div>
            <div className="mt-3"><ErrorBox error={submitErr} /></div>
          </Card>
          {running && <Progress expId={running} onDone={() => history.reload()} />}
        </div>
      </div>
      <div className="mt-8">
        <Card title="Experiment history" subtitle="Experiments of the active dataset" pad={false}>
          <div className="p-4">
            {history.error ? <ErrorBox error={history.error} /> : !history.data ? <Spinner /> : (
              <DataTable<Experiment> rows={history.data} rowKey={(r) => r.id} dense onRowClick={(r) => setRunning(r.id)} selectedKey={running}
                columns={[
                  { key: 'name', header: 'Experiment', value: (r) => r.name, render: (r) => <div className="font-medium">{r.name}</div> },
                  { key: 'mode', header: 'Mode', value: (r) => r.mode, render: (r) => MODE_LABEL[r.mode] ?? r.mode },
                  { key: 'status', header: 'Status', value: (r) => r.status, render: (r) => <Badge tone={r.status === 'completed' ? 'ok' : r.status === 'running' || r.status === 'queued' ? 'accent' : r.status.includes('error') ? 'warn' : 'bad'}>{r.status}</Badge> },
                  { key: 'runs', header: 'Runs', align: 'right', value: (r) => r.n_completed ?? 0, render: (r) => `${r.n_completed}/${r.n_runs}` },
                  { key: 'created', header: 'Created', value: (r) => r.created_at, render: (r) => <span className="text-xs text-muted">{fmtDate(r.created_at)}</span> },
                  { key: 'open', header: '', sortable: false, render: (r) => <Link to={`/benchmark?experiment=${r.id}`} className="text-xs text-accent hover:underline" onClick={(e) => e.stopPropagation()}>leaderboard →</Link> },
                ]} />
            )}
          </div>
        </Card>
      </div>
      <p className="mt-4 text-[11px] text-dim">Score types: {Object.values(SCORE_TYPE_LABEL).join(' · ')}. Only probability outputs are bounded in [0, 1]; none are assumed to be calibrated.</p>
    </div>
  )
}
