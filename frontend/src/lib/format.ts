export const fmtInt = (v: number | null | undefined) => (v == null ? '—' : Math.round(v).toLocaleString('en-US'))

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return '—'
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e7)) return v.toExponential(2)
  if (a >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 2 })
  return v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export const fmtMetric = (v: number | null | undefined) => (v == null ? 'n/a' : v.toFixed(4))

export function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MB`
  return `${(b / 1024 ** 3).toFixed(2)} GB`
}

export function fmtSeconds(s: number | null | undefined): string {
  if (s == null) return '—'
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`
  if (s < 90) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

export const fmtDate = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString() : '—')

export const ROLE_LABEL: Record<string, string> = {
  target: 'Target',
  numeric: 'Numeric feature',
  categorical: 'Categorical feature',
  pca: 'Existing PCA feature',
  time: 'Time / timestamp',
  transaction_id: 'Transaction ID',
  group_id: 'Group / entity ID',
  ignore: 'Ignore',
}

export const ROLE_SHORT: Record<string, string> = {
  target: 'Target',
  numeric: 'Numeric',
  categorical: 'Categorical',
  pca: 'Existing PCA',
  time: 'Time',
  transaction_id: 'Transaction ID',
  group_id: 'Entity ID',
  ignore: 'Ignore',
}

/** Compact tick label for small densities/shares (2 significant digits). */
export const fmtTick = (v: number) => (v === 0 ? '0' : Math.abs(v) >= 100 ? fmtNum(v, 0) : String(Number(v.toPrecision(2))))

export const IMBALANCE_LABEL: Record<string, string> = {
  none: 'No adjustment',
  class_weight: 'Class weighting',
  undersample: 'Undersampling',
  oversample: 'Oversampling',
  smote: 'SMOTE',
}

export const SCORE_TYPE_LABEL: Record<string, string> = {
  probability: 'Model probability (uncalibrated)',
  decision_score: 'Decision score',
  anomaly_score: 'Anomaly score',
}
