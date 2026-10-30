import clsx from 'clsx'
import { useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import type { Any } from '../lib/api'
import { fmtInt, fmtNum, fmtPct, fmtTick } from '../lib/format'

export const C = {
  pos: '#fb7185',
  neg: '#7d93ab',
  accent: '#2dd4bf',
  accent2: '#22d3ee',
  grid: '#1d2530',
  axis: '#64748b',
  warn: '#f5b84b',
}
export const SERIES = ['#2dd4bf', '#f5b84b', '#a78bfa', '#38bdf8', '#fb7185', '#a3e635', '#f97316', '#e879f9', '#94a3b8', '#facc15']

const axisProps = { stroke: C.axis, tick: { fill: C.axis, fontSize: 11 }, tickLine: false, axisLine: { stroke: C.grid } }
const tooltipProps = { contentStyle: { background: '#141a23', border: '1px solid #2a3441', borderRadius: 8, fontSize: 12 }, labelStyle: { color: '#94a3b4' }, itemStyle: { padding: 0 } }

export function classColor(name: string, posName: string) {
  return name === posName ? C.pos : name === 'All' ? C.accent : C.neg
}

/** Overlaid step-area histogram for one or more groups that share bin edges. */
export function Histogram({ hist, posName, mode = 'density', height = 260, xLabel, logDisplay }: { hist: Any; posName: string; mode?: 'density' | 'share' | 'counts'; height?: number; xLabel?: string; logDisplay?: boolean }) {
  const names = Object.keys(hist?.series ?? {})
  const data = useMemo(() => {
    if (!hist?.edges?.length) return []
    const edges: number[] = hist.display_edges ?? hist.edges
    return edges.slice(0, -1).map((e, i) => {
      const row: Record<string, number | string> = { x: e, x2: edges[i + 1] }
      for (const n of names) row[n] = hist.series[n][mode][i]
      return row
    })
  }, [hist, mode, names])
  if (!data.length) return null
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 12, bottom: xLabel ? 18 : 4, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="x" {...axisProps} tickFormatter={(v) => fmtNum(v, logDisplay ? 1 : 2)} minTickGap={28} label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -10, fill: C.axis, fontSize: 11 } : undefined} />
        <YAxis {...axisProps} width={56} tickFormatter={(v) => (mode === 'share' ? fmtPct(v, 1) : mode === 'counts' ? fmtInt(v) : fmtTick(v))} />
        <Tooltip {...tooltipProps} labelFormatter={(_, p) => (p?.[0] ? `${fmtNum(p[0].payload.x)} – ${fmtNum(p[0].payload.x2)}` : '')} formatter={(v: Any, n: Any) => [mode === 'share' ? fmtPct(v, 3) : fmtNum(v, 5), n]} />
        <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" height={24} />
        {names.map((n) => (
          <Area key={n} type="stepAfter" dataKey={n} stroke={classColor(n, posName)} fill={classColor(n, posName)} fillOpacity={0.22} strokeWidth={1.6} isAnimationActive={false} />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function ClassBars({ data, height = 220, log }: { data: { name: string; value: number; color: string; share?: number }[]; height?: number; log?: boolean }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 22, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="name" {...axisProps} />
        <YAxis {...axisProps} width={64} scale={log ? 'log' : 'auto'} domain={log ? [1, 'auto'] : [0, 'auto']} allowDataOverflow tickFormatter={(v) => fmtInt(v)} />
        <Tooltip {...tooltipProps} cursor={{ fill: 'rgba(148,163,180,0.06)' }} formatter={(v: Any) => [fmtInt(v), 'records']} />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive={false} label={{ position: 'top', fill: '#cbd5e1', fontSize: 11, formatter: (v: Any) => fmtInt(v) }}>
          {data.map((d) => <Cell key={d.name} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/** Horizontal box plots drawn from precomputed quartiles/whiskers (full data). */
export function BoxPlots({ boxes, order, posName, log }: { boxes: Record<string, Any>; order: string[]; posName: string; log?: boolean }) {
  const items = order.filter((k) => boxes[k])
  const lo = Math.min(...items.map((k) => boxes[k].whisker_low))
  const hi = Math.max(...items.map((k) => boxes[k].whisker_high))
  const useLog = log && lo >= 0
  const f = (v: number) => (useLog ? Math.log10(1 + v) : v)
  const [a, b] = [f(lo), f(hi) === f(lo) ? f(lo) + 1 : f(hi)]
  const W = 640, rowH = 46, padL = 92, padR = 16
  const x = (v: number) => padL + ((f(v) - a) / (b - a)) * (W - padL - padR)
  const ticks = Array.from({ length: 5 }, (_, i) => a + ((b - a) * i) / 4).map((t) => (useLog ? 10 ** t - 1 : t))
  return (
    <svg viewBox={`0 0 ${W} ${items.length * rowH + 28}`} className="w-full" role="img" aria-label="Box plots">
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={x(t)} x2={x(t)} y1={4} y2={items.length * rowH + 4} stroke={C.grid} />
          <text x={x(t)} y={items.length * rowH + 20} fill={C.axis} fontSize={11} textAnchor="middle">{fmtNum(t, 1)}</text>
        </g>
      ))}
      {items.map((k, i) => {
        const s = boxes[k]
        const y = i * rowH + 10
        const color = classColor(k, posName)
        return (
          <g key={k}>
            <text x={8} y={y + 16} fill="#cbd5e1" fontSize={12}>{k}</text>
            <text x={8} y={y + 30} fill={C.axis} fontSize={10}>n={fmtInt(s.count)}</text>
            <line x1={x(s.whisker_low)} x2={x(s.whisker_high)} y1={y + 14} y2={y + 14} stroke={color} strokeWidth={1.4} />
            <line x1={x(s.whisker_low)} x2={x(s.whisker_low)} y1={y + 6} y2={y + 22} stroke={color} />
            <line x1={x(s.whisker_high)} x2={x(s.whisker_high)} y1={y + 6} y2={y + 22} stroke={color} />
            <rect x={x(s.q1)} y={y + 2} width={Math.max(1, x(s.q3) - x(s.q1))} height={24} rx={3} fill={color} fillOpacity={0.22} stroke={color} />
            <line x1={x(s.median)} x2={x(s.median)} y1={y + 2} y2={y + 26} stroke="#f8fafc" strokeWidth={2} />
            <circle cx={x(Math.min(Math.max(s.mean, lo), hi))} cy={y + 14} r={3} fill="#0a0d12" stroke="#f8fafc" />
          </g>
        )
      })}
    </svg>
  )
}

export function Heatmap({ labels, matrix }: { labels: string[]; matrix: (number | null)[][] }) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)
  const n = labels.length
  const cell = Math.max(10, Math.min(24, Math.floor(620 / n)))
  const pad = 104
  const color = (v: number | null) => {
    if (v == null) return '#111'
    const t = Math.max(-1, Math.min(1, v))
    const [r, g, b] = t >= 0 ? [251, 113, 133] : [45, 212, 191]
    const a = Math.abs(t)
    return `rgba(${r},${g},${b},${0.08 + a * 0.92})`
  }
  return (
    <div className="relative overflow-auto">
      <svg width={pad + n * cell + 8} height={pad + n * cell + 8} role="img" aria-label="Correlation heatmap">
        {labels.map((l, i) => (
          <g key={l}>
            <text x={pad - 6} y={pad + i * cell + cell * 0.68} fontSize={Math.min(11, cell * 0.6)} fill={hover?.i === i ? '#e6edf3' : C.axis} textAnchor="end">{l}</text>
            <text transform={`translate(${pad + i * cell + cell * 0.68}, ${pad - 6}) rotate(-60)`} fontSize={Math.min(11, cell * 0.6)} fill={hover?.j === i ? '#e6edf3' : C.axis}>{l}</text>
          </g>
        ))}
        {matrix.map((row, i) =>
          row.map((v, j) => (
            <rect key={`${i}-${j}`} x={pad + j * cell} y={pad + i * cell} width={cell - 1} height={cell - 1} fill={color(v)} onMouseEnter={() => setHover({ i, j })} onMouseLeave={() => setHover(null)} />
          )),
        )}
      </svg>
      {hover && (
        <div className="pointer-events-none absolute right-2 top-2 rounded-md border border-line-2 bg-panel-2 px-3 py-2 text-xs">
          <div className="text-muted">{labels[hover.i]} × {labels[hover.j]}</div>
          <div className="num text-base">{fmtNum(matrix[hover.i][hover.j], 4)}</div>
        </div>
      )}
      <div className="mt-2 flex items-center gap-2 text-[11px] text-dim">
        <span>−1</span>
        <div className="h-2 w-40 rounded" style={{ background: 'linear-gradient(90deg, rgba(45,212,191,1), rgba(45,212,191,0.08) 49%, rgba(251,113,133,0.08) 51%, rgba(251,113,133,1))' }} />
        <span>+1</span>
      </div>
    </div>
  )
}

export function ConfusionMatrix({ cm, posName, negName, compact }: { cm: { tp: number; fp: number; tn: number; fn: number }; posName: string; negName: string; compact?: boolean }) {
  const total = cm.tp + cm.fp + cm.tn + cm.fn
  const cells = [
    { k: 'TN', v: cm.tn, label: `True ${negName}`, good: true },
    { k: 'FP', v: cm.fp, label: 'False alarm', good: false },
    { k: 'FN', v: cm.fn, label: `Missed ${posName}`, good: false },
    { k: 'TP', v: cm.tp, label: `Caught ${posName}`, good: true },
  ]
  return (
    <div className={clsx('grid grid-cols-[auto_1fr_1fr] gap-1.5 text-xs', compact ? 'max-w-[300px]' : 'max-w-[420px]')}>
      <div />
      <div className="pb-1 text-center text-dim">Predicted {negName}</div>
      <div className="pb-1 text-center text-dim">Predicted {posName}</div>
      {[0, 1].map((r) => (
        <div key={r} className="contents">
          <div className="flex items-center pr-2 text-right text-dim [writing-mode:horizontal-tb]">Actual {r === 0 ? negName : posName}</div>
          {cells.slice(r * 2, r * 2 + 2).map((c) => (
            <div key={c.k} className={clsx('rounded-lg border px-3 py-2.5', c.good ? 'border-accent/25 bg-accent/[0.07]' : c.v ? 'border-pos/30 bg-pos/[0.08]' : 'border-line bg-panel-2')}>
              <div className="flex items-baseline justify-between">
                <span className="font-semibold text-muted">{c.k}</span>
                <span className="text-[10px] text-dim">{fmtPct(total ? c.v / total : 0, 2)}</span>
              </div>
              <div className={clsx('num font-semibold', compact ? 'text-lg' : 'text-2xl')}>{fmtInt(c.v)}</div>
              {!compact && <div className="text-[11px] text-dim">{c.label}</div>}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

export function SegmentedBar({ parts }: { parts: { label: string; value: number; color: string }[] }) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1
  return (
    <div>
      <div className="flex h-9 overflow-hidden rounded-lg border border-line-2">
        {parts.map((p) => (
          <div key={p.label} className="flex items-center justify-center text-xs font-semibold text-[#06121a] transition-[width] duration-200" style={{ width: `${(100 * p.value) / total}%`, background: p.color }}>
            {p.value / total > 0.07 && `${p.label} ${Math.round((100 * p.value) / total)}%`}
          </div>
        ))}
      </div>
    </div>
  )
}

export interface CurveSeries {
  name: string
  color: string
  points: { x: number; y: number }[]
  dashed?: boolean
}

export function Curves({ series, xLabel, yLabel, height = 300, diagonal, baseline }: { series: CurveSeries[]; xLabel: string; yLabel: string; height?: number; diagonal?: boolean; baseline?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
        <CartesianGrid stroke={C.grid} />
        <XAxis type="number" dataKey="x" domain={[0, 1]} {...axisProps} tickFormatter={(v) => fmtNum(v, 1)} label={{ value: xLabel, position: 'insideBottom', offset: -10, fill: C.axis, fontSize: 11 }} />
        <YAxis type="number" dataKey="y" domain={[0, 1]} {...axisProps} width={44} tickFormatter={(v) => fmtNum(v, 1)} label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 14, fill: C.axis, fontSize: 11 }} />
        <Tooltip {...tooltipProps} formatter={(v: Any) => fmtNum(v, 3)} labelFormatter={(v) => `${xLabel} ${fmtNum(v as number, 3)}`} />
        <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" />
        {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.axis} strokeDasharray="4 4" />}
        {baseline != null && <ReferenceLine y={baseline} stroke={C.axis} strokeDasharray="4 4" label={{ value: `random = ${fmtNum(baseline, 4)}`, fill: C.axis, fontSize: 10, position: 'insideTopRight' }} />}
        {series.map((s) => (
          <Line key={s.name} data={s.points} dataKey="y" name={s.name} stroke={s.color} dot={false} strokeWidth={1.8} strokeDasharray={s.dashed ? '5 4' : undefined} isAnimationActive={false} type="linear" />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

export function BarList({ items, color = C.accent, signed, format = (v: number) => fmtNum(v, 4) }: { items: { label: string; value: number; err?: number }[]; color?: string; signed?: boolean; format?: (v: number) => string }) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-12)
  return (
    <div className="space-y-1.5">
      {items.map((i) => (
        <div key={i.label} className="grid grid-cols-[140px_1fr_76px] items-center gap-3 text-xs">
          <div className="truncate text-muted" title={i.label}>{i.label}</div>
          <div className="relative h-4 rounded bg-panel-3">
            <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${(100 * Math.abs(i.value)) / max}%`, background: signed && i.value < 0 ? C.neg : color, opacity: 0.85 }} />
          </div>
          <div className="num text-right">{format(i.value)}{i.err != null && <span className="text-dim"> ±{fmtNum(i.err, 3)}</span>}</div>
        </div>
      ))}
    </div>
  )
}

export function TimeChart({ series, xKey, height = 260, posName, xLabel }: { series: Any[]; xKey: string; height?: number; posName: string; xLabel: string }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={series} margin={{ top: 8, right: 8, bottom: 18, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey={xKey} {...axisProps} tickFormatter={(v) => fmtNum(v, 1)} minTickGap={24} label={{ value: xLabel, position: 'insideBottom', offset: -10, fill: C.axis, fontSize: 11 }} />
        <YAxis yAxisId="l" {...axisProps} width={56} tickFormatter={(v) => fmtInt(v)} />
        <YAxis yAxisId="r" orientation="right" {...axisProps} width={40} stroke={C.pos} tick={{ fill: C.pos, fontSize: 11 }} />
        <Tooltip {...tooltipProps} formatter={(v: Any, n: Any) => [fmtInt(v), n]} labelFormatter={(v) => `${xLabel}: ${fmtNum(v as number, 2)}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" height={24} />
        <Bar yAxisId="l" dataKey="n" name="All records" fill={C.neg} fillOpacity={0.5} isAnimationActive={false} />
        <Line yAxisId="r" dataKey="n_positive" name={`${posName} records`} stroke={C.pos} dot={false} strokeWidth={1.8} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function RateChart({ data, xKey, height = 220, posName, xLabel }: { data: Any[]; xKey: string; height?: number; posName: string; xLabel: string }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 18, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey={xKey} {...axisProps} tickFormatter={(v) => fmtNum(v, 1)} minTickGap={24} label={{ value: xLabel, position: 'insideBottom', offset: -10, fill: C.axis, fontSize: 11 }} />
        <YAxis {...axisProps} width={56} tickFormatter={(v) => fmtPct(v, 2)} />
        <Tooltip {...tooltipProps} formatter={(v: Any) => [fmtPct(v, 3), `${posName} rate`]} labelFormatter={(v) => `${xLabel}: ${fmtNum(v as number, 2)}`} />
        <Line dataKey="positive_rate" name={`${posName} rate`} stroke={C.pos} dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function ProjectionScatter({ points, posName, negName }: { points: { x: number; y: number; label: number }[]; posName: string; negName: string }) {
  const neg = points.filter((p) => p.label === 0)
  const pos = points.filter((p) => p.label === 1)
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
        <CartesianGrid stroke={C.grid} />
        <XAxis type="number" dataKey="x" name="PC1" {...axisProps} tickFormatter={(v) => fmtNum(v, 1)} label={{ value: 'Component 1', position: 'insideBottom', offset: -10, fill: C.axis, fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name="PC2" {...axisProps} width={44} tickFormatter={(v) => fmtNum(v, 1)} />
        <ZAxis range={[10, 10]} />
        <Tooltip {...tooltipProps} formatter={(v: Any) => fmtNum(v, 3)} />
        <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" height={24} />
        <Scatter name={negName} data={neg} fill={C.neg} fillOpacity={0.35} isAnimationActive={false} />
        <Scatter name={posName} data={pos} fill={C.pos} fillOpacity={0.8} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

export function GroupedBars({ data, keys, height = 260, yFormat = (v: number) => fmtNum(v, 2) }: { data: Any[]; keys: { key: string; name: string; color: string }[]; height?: number; yFormat?: (v: number) => string }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 40, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="name" {...axisProps} interval={0} angle={-25} textAnchor="end" height={60} />
        <YAxis {...axisProps} width={48} tickFormatter={yFormat} />
        <Tooltip {...tooltipProps} cursor={{ fill: 'rgba(148,163,180,0.06)' }} formatter={(v: Any, n: Any) => [v == null ? 'n/a' : fmtNum(v, 4), n]} />
        <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" />
        {keys.map((k) => <Bar key={k.key} dataKey={k.key} name={k.name} fill={k.color} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}
      </BarChart>
    </ResponsiveContainer>
  )
}

export function ThresholdChart({ data, current, frozen, height = 260 }: { data: { t: number; precision: number | null; recall: number | null; f1: number | null }[]; current: number; frozen: number; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
        <CartesianGrid stroke={C.grid} />
        <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} {...axisProps} tickFormatter={(v) => fmtNum(v, 2)} label={{ value: 'Threshold (score ≥ t ⇒ alert)', position: 'insideBottom', offset: -10, fill: C.axis, fontSize: 11 }} />
        <YAxis domain={[0, 1]} {...axisProps} width={40} tickFormatter={(v) => fmtNum(v, 1)} />
        <Tooltip {...tooltipProps} formatter={(v: Any, n: Any) => [v == null ? 'n/a' : fmtNum(v, 3), n]} labelFormatter={(v) => `t = ${fmtNum(v as number, 4)}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" height={24} />
        <ReferenceLine x={frozen} stroke={C.warn} strokeDasharray="4 3" label={{ value: 'frozen', fill: C.warn, fontSize: 10, position: 'insideTopRight' }} />
        <ReferenceLine x={current} stroke="#e6edf3" />
        <Line dataKey="precision" name="Precision" stroke={C.accent} dot={false} isAnimationActive={false} />
        <Line dataKey="recall" name="Recall" stroke={C.pos} dot={false} isAnimationActive={false} />
        <Line dataKey="f1" name="F1" stroke={C.warn} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
