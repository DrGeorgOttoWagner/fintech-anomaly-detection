import { useState } from 'react'
import { BarList, BoxPlots, ClassBars, C, Heatmap, Histogram, ProjectionScatter, RateChart, TimeChart } from '../components/charts'
import { Badge, Callout, Card, DataTable, ErrorBox, KeyValue, NeedSchema, PageHeader, Segmented, Select, Spinner, Stat, Tabs } from '../components/ui'
import type { Any, Envelope } from '../lib/api'
import { fmtInt, fmtNum, fmtPct, ROLE_LABEL } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

type Tab = 'descriptive' | 'imbalance' | 'value' | 'time' | 'features' | 'findings'

function useEda<T = Any>(path: string | null) {
  const { dataset, schema } = useSession()
  const r = useApi<Envelope<T>>(dataset && schema && path ? `/api/datasets/${dataset.id}/eda/${path}${path.includes('?') ? '&' : '?'}h=${schema.hash}` : null)
  // Defensive: never render a result computed for another dataset/schema version.
  const fresh = r.data && dataset && r.data.dataset_id === dataset.id && r.data.schema_hash === schema?.hash
  return { ...r, result: fresh ? r.data!.result : undefined }
}

function Loading({ error, label }: { error: unknown; label?: string }) {
  return error ? <ErrorBox error={error} /> : <Spinner label={label ?? 'Computing on the full dataset…'} />
}

function Descriptive() {
  const { result: ov, error } = useEda('overview')
  if (!ov) return <Loading error={error} />
  const cb = ov.class_balance
  const numeric = ov.columns.filter((c: Any) => c.mean !== undefined)
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7" data-testid="overview-stats">
        <Stat label="Records" value={fmtInt(ov.n_records)} />
        <Stat label="Features" value={fmtInt(ov.n_features)} sub={`${ov.n_pca_features} precomputed PCA`} />
        <Stat label="Classes" value={cb ? 2 : '—'} sub={cb ? `${cb.negative_name} / ${cb.positive_name}` : 'no target'} />
        <Stat label="Prevalence" value={cb ? fmtPct(cb.prevalence, 3) : '—'} tone="pos" sub={cb ? `${fmtInt(cb.n_positive)} ${cb.positive_name.toLowerCase()}` : undefined} />
        <Stat label="Imbalance ratio" value={cb ? `1 : ${fmtNum(cb.majority_to_minority, 1)}` : '—'} />
        <Stat label="Missing cells" value={fmtInt(ov.total_missing_cells)} />
        <Stat label="Duplicate rows" value={fmtInt(ov.duplicate_rows)} tone={ov.duplicate_rows ? 'warn' : undefined} sub={fmtPct(ov.duplicate_rows_pct, 3)} />
      </div>
      {ov.warnings.length > 0 && (
        <Card title="Data quality">
          <div className="space-y-2">
            {ov.warnings.map((w: Any) => (
              <div key={w.message} className="flex items-start gap-2 text-sm">
                <Badge tone={w.level === 'warning' ? 'warn' : 'default'}>{w.level}</Badge>
                <span className="text-muted">{w.message}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
      <Card title="Numeric descriptive statistics" subtitle="Computed on all records. Click a header to sort; filter by column name." pad={false}>
        <div className="p-4">
          <DataTable<Any>
            rows={numeric}
            rowKey={(r) => r.name}
            filterable={(r) => `${r.name} ${r.role}`}
            filterPlaceholder="Filter columns…"
            maxHeight={520}
            dense
            columns={[
              { key: 'name', header: 'Column', value: (r) => r.name, render: (r) => <span className="font-medium">{r.name}</span> },
              { key: 'role', header: 'Role', value: (r) => r.role, render: (r) => <span className="text-xs text-muted">{ROLE_LABEL[r.role]}</span> },
              ...(['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'skewness', 'kurtosis', 'missing'] as const).map((k) => ({
                key: k, header: k === 'q1' ? 'Q1' : k === 'q3' ? 'Q3' : k, align: 'right' as const, value: (r: Any) => r[k], render: (r: Any) => (k === 'count' || k === 'missing' ? fmtInt(r[k]) : fmtNum(r[k], 4)),
              })),
            ]}
          />
        </div>
      </Card>
      <Card title="Column types and roles">
        <div className="flex flex-wrap gap-2 text-sm">
          {Object.entries(ov.role_counts).map(([r, n]) => <Badge key={r}>{ROLE_LABEL[r]} · {String(n)}</Badge>)}
          <Badge tone="dim">numeric dtype · {ov.kind_counts.numeric}</Badge>
          <Badge tone="dim">non-numeric dtype · {ov.kind_counts.non_numeric}</Badge>
          <Badge tone="dim">{fmtNum(ov.memory_mb, 1)} MB in memory</Badge>
        </div>
      </Card>
    </div>
  )
}

function Imbalance() {
  const { result: ov, error } = useEda('overview')
  const [log, setLog] = useState(false)
  if (!ov) return <Loading error={error} />
  const cb = ov.class_balance
  if (!cb) return <Callout tone="info">No target column is configured, so class balance cannot be computed.</Callout>
  return (
    <div className="grid gap-6 xl:grid-cols-[1.1fr_1fr]">
      <Card title="Class distribution" subtitle="All records" actions={<Segmented size="sm" value={log ? 'log' : 'lin'} onChange={(v) => setLog(v === 'log')} options={[{ value: 'lin', label: 'Linear' }, { value: 'log', label: 'Log scale' }]} />}>
        <ClassBars log={log} data={[{ name: cb.negative_name, value: cb.n_negative, color: C.neg }, { name: cb.positive_name, value: cb.n_positive, color: C.pos }]} height={280} />
        <div className="mt-4">
          <div className="mb-1.5 flex justify-between text-xs text-muted"><span>{cb.negative_name} {fmtPct(cb.negative_share, 3)}</span><span className="text-pos">{cb.positive_name} {fmtPct(cb.prevalence, 3)}</span></div>
          <div className="flex h-3 overflow-hidden rounded-full bg-panel-3">
            <div style={{ width: `${100 * cb.negative_share}%`, background: C.neg }} />
            <div style={{ width: `${Math.max(0.4, 100 * cb.prevalence)}%`, background: C.pos }} />
          </div>
          <div className="mt-1 text-[11px] text-dim">Positive segment widened to a minimum of 0.4% so it remains visible; numbers are exact.</div>
        </div>
      </Card>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <Stat label={`${cb.negative_name} records`} value={fmtInt(cb.n_negative)} sub={fmtPct(cb.negative_share, 3)} />
          <Stat label={`${cb.positive_name} records`} value={fmtInt(cb.n_positive)} sub={fmtPct(cb.prevalence, 4)} tone="pos" />
          <Stat label="Minority : majority" value={`1 : ${fmtNum(cb.majority_to_minority, 1)}`} sub={`ratio ${fmtNum(cb.minority_to_majority, 5)}`} />
          <Stat label="Majority baseline accuracy" value={fmtPct(cb.majority_baseline_accuracy, 3)} sub={`always '${cb.majority_class}' · recall 0%`} tone="warn" />
        </div>
        <Card title="Why accuracy misleads here">
          <div className="space-y-2.5 text-sm leading-relaxed text-muted">
            <p>Only <b className="text-text">{fmtPct(cb.prevalence, 3)}</b> of records are <b className="text-pos">{cb.positive_name.toLowerCase()}</b> — about one in <b className="num text-text">{fmtInt(cb.one_in_n)}</b>. A model that never raises an alert is correct for every {cb.negative_name.toLowerCase()} record and reaches <b className="text-text">{fmtPct(cb.majority_baseline_accuracy, 3)}</b> accuracy, while catching none of the cases that matter.</p>
            <p>For rare events, evaluation should focus on the positive class: <b className="text-text">recall</b> (share of positives caught), <b className="text-text">precision</b> (share of alerts that are real), their harmonic mean <b className="text-text">F1</b>, and the <b className="text-text">PR-AUC</b>, whose random baseline equals the prevalence ({fmtNum(cb.prevalence, 5)}) rather than 0.5.</p>
            <p>With {fmtInt(cb.n_positive)} positives in total, any held-out partition contains few positives, so stratified splits and careful threshold selection are required.</p>
          </div>
        </Card>
      </div>
    </div>
  )
}

function ValueAnalysis() {
  const { schema } = useSession()
  const cfg = schema!.config
  const numeric = Object.entries(cfg.roles).filter(([, r]) => r === 'numeric' || r === 'pca').map(([c]) => c)
  const [col, setCol] = useState(cfg.amount_column ?? numeric[0])
  const [log, setLog] = useState(false)
  const [mode, setMode] = useState<'density' | 'share' | 'counts'>('density')
  const { result: v, error } = useEda(col ? `value?column=${encodeURIComponent(col)}&log=${log}` : null)
  if (!col) return <Callout>No numeric value column available.</Callout>
  const pos = cfg.positive_name, neg = cfg.negative_name
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Select value={col} onChange={setCol} options={numeric.map((c) => ({ value: c, label: c === cfg.amount_column ? `${c} · amount column` : c }))} ariaLabel="Value column" />
        <Segmented value={log ? 'log' : 'lin'} onChange={(x) => setLog(x === 'log')} options={[{ value: 'lin', label: 'Linear' }, { value: 'log', label: 'log(1 + x)', disabled: v && !v.log_available, title: 'Requires non-negative values' }]} />
        <Segmented value={mode} onChange={setMode} options={[{ value: 'density', label: 'Density' }, { value: 'share', label: 'Share' }, { value: 'counts', label: 'Counts' }]} />
      </div>
      {!v ? <Loading error={error} /> : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5 xl:grid-cols-9">
            <Stat label="Mean" value={fmtNum(v.overall.mean, 2)} />
            <Stat label="Median" value={fmtNum(v.overall.median, 2)} />
            <Stat label="Std. dev." value={fmtNum(v.overall.std, 2)} />
            <Stat label="Min" value={fmtNum(v.overall.min, 2)} />
            <Stat label="Max" value={fmtNum(v.overall.max, 2)} />
            <Stat label="Q1" value={fmtNum(v.overall.q1, 2)} />
            <Stat label="Q3" value={fmtNum(v.overall.q3, 2)} />
            <Stat label="Skewness" value={fmtNum(v.overall.skewness, 2)} />
            <Stat label="Zeros" value={fmtInt(v.overall.n_zero)} />
          </div>
          <div className="grid gap-6 xl:grid-cols-2">
            <Card title={`Distribution of ${col} by class`} subtitle={log ? 'log(1 + x) bins; axis shows original units' : 'Linear bins over the full range (heavy tails compress the view — see clipped panel).'}>
              <Histogram hist={v.histogram} posName={pos} mode={mode} logDisplay={log} xLabel={col} />
            </Card>
            {v.histogram_clipped ? (
              <Card title="Body of the distribution" subtitle="Bins limited to ≤ 99th percentile; values above are counted in the last bin.">
                <Histogram hist={v.histogram_clipped} posName={pos} mode={mode} xLabel={col} />
              </Card>
            ) : (
              <Card title="Box plots by class" subtitle="Box = IQR, whiskers = 1.5 × IQR, white line = median, dot = mean."><BoxPlots boxes={v.box} order={[neg, pos]} posName={pos} log /></Card>
            )}
          </div>
          {v.by_class && (
            <div className="grid gap-6 xl:grid-cols-2">
              {v.histogram_clipped && <Card title="Box plots by class" subtitle="Box = IQR, whiskers = 1.5 × IQR (outliers beyond not drawn), white line = median, dot = mean. Symmetric log axis."><BoxPlots boxes={v.box} order={[neg, pos]} posName={pos} log /></Card>}
              <Card title="Statistics by class" pad={false}>
                <div className="p-4">
                  <DataTable<Any> rows={['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'p95', 'p99', 'max', 'skewness'].map((k) => ({ k, a: v.by_class[neg][k], b: v.by_class[pos][k] }))} rowKey={(r) => r.k} dense
                    columns={[{ key: 'k', header: 'Statistic', render: (r) => r.k }, { key: 'a', header: neg, align: 'right', render: (r) => fmtNum(r.a, 3) }, { key: 'b', header: pos, align: 'right', render: (r) => <span className="text-pos">{fmtNum(r.b, 3)}</span> }]} />
                  {v.tests?.available && (
                    <div className="mt-4 text-xs text-muted">
                      <KeyValue items={[
                        ['Mann–Whitney U', <span className="num">p = {v.tests.mann_whitney_p.toExponential(2)} · rank-biserial r = {fmtNum(v.tests.rank_biserial, 3)}</span>],
                        ['Kolmogorov–Smirnov', <span className="num">D = {fmtNum(v.tests.ks_statistic, 3)} · p = {v.tests.ks_p.toExponential(2)}</span>],
                        ['Univariate ROC-AUC', <span className="num">{fmtNum(v.univariate_auc, 4)}</span>],
                      ]} />
                      <p className="mt-2 text-[11px] text-dim">Statistically different distributions do not mean that a large (or small) value indicates a {pos.toLowerCase()} record. With large samples, small differences become significant; read the effect sizes.</p>
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}
          {v.bands?.length > 0 && (
            <Card title={`${pos} rate across ${col} deciles`} subtitle="Each band holds ≈10% of records (ties may merge bands).">
              <DataTable<Any> rows={v.bands.map((b: Any, i: number) => ({ ...b, i }))} rowKey={(r) => String(r.i)} dense
                columns={[
                  { key: 'i', header: 'Band', render: (r) => r.i + 1 },
                  { key: 'range', header: 'Range', render: (r) => <span className="num">{fmtNum(r.low, 2)} – {fmtNum(r.high, 2)}</span> },
                  { key: 'n', header: 'Records', align: 'right', value: (r) => r.n, render: (r) => fmtInt(r.n) },
                  { key: 'np', header: pos, align: 'right', value: (r) => r.n_positive, render: (r) => fmtInt(r.n_positive) },
                  { key: 'rate', header: `${pos} rate`, align: 'right', value: (r) => r.positive_rate, render: (r) => <span className="text-pos">{fmtPct(r.positive_rate, 3)}</span> },
                ]} />
            </Card>
          )}
        </>
      )}
    </div>
  )
}

function TimeAnalysis() {
  const { schema } = useSession()
  const cfg = schema!.config
  const [bin, setBin] = useState<string>('auto')
  const { result: t, error } = useEda(cfg.time_column ? `time${bin === 'auto' ? '' : `?bin_size=${bin}`}` : null)
  if (!cfg.time_column) return <Callout tone="info">No time column is configured in the schema mapping.</Callout>
  if (!t) return <Loading error={error} />
  const hours = t.kind !== 'ordinal'
  const series = t.series.map((r: Any) => ({ ...r, x: hours ? (r.start - t.start) / 3600 : r.start }))
  const pos = cfg.positive_name, neg = cfg.negative_name
  return (
    <div className="space-y-6">
      {t.interpretation_note && <Callout tone="info" title="Interpretation of the time column">{t.interpretation_note}</Callout>}
      <div className="flex flex-wrap items-center gap-3">
        {hours && <Segmented value={bin} onChange={setBin} options={[{ value: 'auto', label: 'Auto bins' }, { value: '900', label: '15 min' }, { value: '3600', label: '1 hour' }, { value: '21600', label: '6 hours' }]} />}
        <span className="text-xs text-dim">{t.series.length} bins of {hours ? `${fmtNum(t.bin_size / 3600, 2)} h` : fmtNum(t.bin_size)} · span {hours ? `${fmtNum(t.span_seconds / 3600, 1)} hours` : fmtNum(t.span_seconds)}{t.origin ? ` · origin ${t.origin}` : ''}</span>
      </div>
      <Card title="Events over time" subtitle={`All records (bars) and ${pos.toLowerCase()} records (line) per bin of '${t.column}'`}>
        <TimeChart series={series} xKey="x" posName={pos} xLabel={hours ? 'Hours since first record' : t.column} height={300} />
      </Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card title={`${pos} rate per interval`} subtitle={t.rate_range ? `Range ${fmtPct(t.rate_range.min, 3)} – ${fmtPct(t.rate_range.max, 3)} across bins with ≥ ${t.rate_range.min_bin_n} records` : undefined}>
          <RateChart data={series} xKey="x" posName={pos} xLabel={hours ? 'Hours since first record' : t.column} />
        </Card>
        {t.histogram && (
          <Card title="Temporal distribution by class" subtitle="Density within each class (48 bins)">
            <Histogram hist={{ ...t.histogram, display_edges: t.histogram.edges.map((e: number) => (hours ? (e - t.start) / 3600 : e)) }} posName={pos} xLabel={hours ? 'Hours since first record' : t.column} />
          </Card>
        )}
      </div>
      {t.hour_of_cycle && (
        <Card title="Hour-of-cycle profile" subtitle="Hours since the first record modulo 24 — an offset, not a clock hour. Aggregated over all covered days.">
          <TimeChart series={t.hour_of_cycle} xKey="hour" posName={pos} xLabel="Hour offset (mod 24)" height={240} />
          <div className="mt-3"><RateChart data={t.hour_of_cycle} xKey="hour" posName={pos} xLabel="Hour offset (mod 24)" height={180} /></div>
        </Card>
      )}
      {t.by_class && (
        <Card title="Time statistics by class" pad={false}>
          <div className="p-4">
            <DataTable<Any> rows={['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max'].map((k) => ({ k, a: t.by_class[neg][k], b: t.by_class[pos][k] }))} rowKey={(r) => r.k} dense
              columns={[{ key: 'k', header: 'Statistic', render: (r) => r.k }, { key: 'a', header: neg, align: 'right', render: (r) => fmtNum(r.a, 2) }, { key: 'b', header: pos, align: 'right', render: (r) => <span className="text-pos">{fmtNum(r.b, 2)}</span> }]} />
            {t.tests?.available && <p className="mt-3 text-xs text-muted num">KS D = {fmtNum(t.tests.ks_statistic, 3)} (p = {t.tests.ks_p.toExponential(2)}), Mann–Whitney p = {t.tests.mann_whitney_p.toExponential(2)}</p>}
          </div>
        </Card>
      )}
    </div>
  )
}

function Features() {
  const { schema } = useSession()
  const cfg = schema!.config
  const [method, setMethod] = useState<'pearson' | 'spearman'>('pearson')
  const corr = useEda(`correlation?method=${method}`)
  const rel = useEda('relevance')
  const pca = useEda('pca')
  const proj = useEda('projection')
  const numeric = Object.entries(cfg.roles).filter(([, r]) => r === 'numeric' || r === 'pca').map(([c]) => c)
  const [feat, setFeat] = useState<string | null>(null)
  const topFeat = feat ?? rel.result?.features?.[0]?.feature ?? numeric[0]
  const dist = useEda(topFeat ? `distribution?column=${encodeURIComponent(topFeat)}` : null)
  return (
    <div className="space-y-6">
      <div className="grid gap-6 2xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <Card title="Correlation heatmap" subtitle={corr.result ? `${corr.result.computed_on}; includes the binary target` : undefined} actions={<Segmented size="sm" value={method} onChange={setMethod} options={[{ value: 'pearson', label: 'Pearson' }, { value: 'spearman', label: 'Spearman' }]} />}>
          {corr.result ? (
            <>
              <Heatmap labels={corr.result.labels} matrix={corr.result.matrix} />
              {corr.result.note && <p className="mt-2 text-xs text-dim">{corr.result.note}</p>}
            </>
          ) : <Loading error={corr.error} />}
        </Card>
        <Card title="Feature relevance indicators" subtitle="Univariate class separation, computed on all records (MI on a labelled sample)." pad={false}>
          <div className="p-4">
            {rel.result?.available ? (
              <>
                <DataTable<Any> rows={rel.result.features} rowKey={(r) => r.feature} dense maxHeight={420} initialSort={{ key: 'ks', dir: 'desc' }} onRowClick={(r) => setFeat(r.feature)} selectedKey={topFeat}
                  columns={[
                    { key: 'f', header: 'Feature', value: (r) => r.feature, render: (r) => <span className="font-medium">{r.feature}{r.role === 'pca' && <span className="ml-1 text-[10px] text-dim">PCA</span>}</span> },
                    { key: 'ks', header: 'KS', align: 'right', value: (r) => r.ks_statistic, render: (r) => fmtNum(r.ks_statistic, 3) },
                    { key: 'auc', header: 'AUC', align: 'right', value: (r) => r.separability, render: (r) => fmtNum(r.separability, 3) },
                    { key: 'r', header: 'r_pb', align: 'right', value: (r) => r.point_biserial_r, render: (r) => fmtNum(r.point_biserial_r, 3) },
                    { key: 'd', header: "Cohen's d", align: 'right', value: (r) => r.cohens_d, render: (r) => fmtNum(r.cohens_d, 2) },
                    { key: 'mi', header: 'MI', align: 'right', value: (r) => r.mutual_information, render: (r) => fmtNum(r.mutual_information, 4) },
                  ]} />
                <p className="mt-2 text-[11px] text-dim">{rel.result.note} {rel.result.mi_note}</p>
              </>
            ) : rel.result ? <Callout>{rel.result.reason}</Callout> : <Loading error={rel.error} label="Computing relevance indicators…" />}
          </div>
        </Card>
      </div>
      <Card title={`Class-conditional distribution · ${topFeat ?? ''}`} subtitle="Select a feature in the relevance table or below. Display clipped to the 0.5–99.5th percentiles." actions={<Select value={topFeat ?? ''} onChange={setFeat} options={numeric.map((c) => ({ value: c, label: c }))} ariaLabel="Feature" />}>
        {dist.result ? (
          <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
            <Histogram hist={dist.result.histogram} posName={cfg.positive_name} xLabel={topFeat ?? ''} />
            <BoxPlots boxes={dist.result.box} order={Object.keys(dist.result.box)} posName={cfg.positive_name} />
          </div>
        ) : <Loading error={dist.error} />}
      </Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Precomputed PCA components" subtitle="Statistical properties only — no business meaning is attributed.">
          {pca.result?.available ? (
            <>
              <BarList items={pca.result.components.map((c: Any) => ({ label: c.component, value: c.variance }))} format={(v) => fmtNum(v, 3)} />
              <p className="mt-3 text-xs text-muted">Max |r| between components: <span className="num text-text">{fmtNum(pca.result.max_abs_inter_component_corr, 4)}</span> · variance {pca.result.variance_monotone_decreasing ? 'decreases monotonically (consistent with PCA ordering)' : 'is not monotonically ordered'}.</p>
              <p className="mt-1 text-[11px] text-dim">{pca.result.note}</p>
            </>
          ) : pca.result ? <Callout>{pca.result.reason}</Callout> : <Loading error={pca.error} />}
        </Card>
        <Card title="2-D projection (exploratory)" subtitle={proj.result?.available ? `Explained variance ${proj.result.explained_variance_ratio.map((v: number) => fmtPct(v, 1)).join(' + ')}` : undefined}>
          {proj.result?.available ? (
            <>
              <ProjectionScatter points={proj.result.points} posName={cfg.positive_name} negName={cfg.negative_name} />
              <p className="mt-2 text-[11px] text-dim">{proj.result.sample_note}</p>
            </>
          ) : proj.result ? <Callout>{proj.result.reason}</Callout> : <Loading error={proj.error} />}
        </Card>
      </div>
    </div>
  )
}

function Findings() {
  const { result, error } = useEda<string[]>('findings')
  if (!result) return <Loading error={error} label="Collecting computed findings…" />
  return (
    <Card title="What the data shows" subtitle="Statements generated from the computed statistics above. They describe associations in this dataset only.">
      <ul className="space-y-3" data-testid="findings">
        {result.map((f, i) => (
          <li key={i} className="flex gap-3 text-sm leading-relaxed">
            <span className="num mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-accent/15 text-[11px] text-accent">{i + 1}</span>
            <span className="text-muted">{f}</span>
          </li>
        ))}
      </ul>
      <div className="mt-5"><Callout tone="warn">These are not causal conclusions. A rare event is not necessarily an anomaly, an anomaly is not necessarily fraud, and a large amount is not by itself evidence of fraud.</Callout></div>
    </Card>
  )
}

export default function Explore() {
  const { dataset, schema } = useSession()
  const [tab, setTab] = useState<Tab>('descriptive')
  if (!dataset || !schema) return <><PageHeader eyebrow="Step 3 · Explore" title="Exploratory Analysis" /><NeedSchema hasDataset={!!dataset} /></>
  const cfg = schema.config
  return (
    <div>
      <PageHeader eyebrow="Step 3 · Explore" title="Exploratory Analysis"
        description={<>Descriptive statistics and EDA on <b className="text-text">{dataset.original_filename}</b>. Statistics use every record; only scatter projections use a labelled visual sample.{cfg.domain === 'fraud' && <> Positive label <b className="text-pos">{cfg.positive_label}</b> represents <b className="text-pos">{cfg.positive_name.toLowerCase()}</b>.</>}</>} />
      <Tabs value={tab} onChange={setTab} tabs={[
        { value: 'descriptive', label: 'Descriptive statistics' },
        { value: 'imbalance', label: 'Class imbalance' },
        { value: 'value', label: !cfg.amount_column ? 'Value analysis' : cfg.amount_column.toLowerCase() === 'amount' ? 'Amount' : `Amount · ${cfg.amount_column}` },
        { value: 'time', label: !cfg.time_column || cfg.time_column.toLowerCase() === 'time' ? 'Time' : `Time · ${cfg.time_column}` },
        { value: 'features', label: 'Features' },
        { value: 'findings', label: 'Findings' },
      ]} />
      {tab === 'descriptive' && <Descriptive />}
      {tab === 'imbalance' && <Imbalance />}
      {tab === 'value' && <ValueAnalysis />}
      {tab === 'time' && <TimeAnalysis />}
      {tab === 'features' && <Features />}
      {tab === 'findings' && <Findings />}
    </div>
  )
}
