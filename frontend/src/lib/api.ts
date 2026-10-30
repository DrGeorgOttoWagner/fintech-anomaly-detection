// Thin typed client for the local FTAD backend. All requests go to the same origin
// (served by FastAPI, or proxied by Vite in development) — data never leaves the machine.

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Any = any

export interface Dataset {
  id: string
  name: string
  original_filename: string
  sha256: string
  format: string
  size_bytes: number
  n_rows: number
  n_cols: number
  created_at: string
  has_schema?: boolean
  schema_hash?: string | null
  active?: boolean
  n_experiments?: number
}

export type Role = 'target' | 'numeric' | 'categorical' | 'pca' | 'time' | 'transaction_id' | 'group_id' | 'ignore'

export interface SchemaConfig {
  roles: Record<string, Role>
  target: string | null
  positive_label: string | null
  positive_name: string
  negative_name: string
  time_column: string | null
  time_kind: 'elapsed_seconds' | 'timestamp' | 'ordinal' | null
  amount_column: string | null
  id_column: string | null
  group_column: string | null
  domain: 'fraud' | 'generic'
  notes: string[]
}

export interface Session {
  active_dataset: Dataset | null
  schema: { id: string; hash: string; config: SchemaConfig } | null
}

export interface Envelope<T = Any> {
  dataset_id: string
  schema_hash: string
  result: T
}

export interface ModelInfo {
  key: string
  name: string
  family: string
  kind: 'supervised' | 'anomaly'
  library: string
  score_type: 'probability' | 'decision_score' | 'anomaly_score'
  needs_scaling: boolean
  class_weight: boolean
  max_train_rows: number | null
  cost: 'low' | 'medium' | 'high'
  quick: boolean
  hpo: boolean
  default_params: Record<string, Any>
  search_space: Record<string, string>
  available: boolean
  unavailable_reason: string | null
  notes: string
}

export interface Metrics {
  n: number
  n_positive: number
  threshold: number
  confusion: { tp: number; fp: number; tn: number; fn: number }
  accuracy: number | null
  balanced_accuracy: number | null
  precision: number | null
  recall: number | null
  specificity: number | null
  f1: number | null
  mcc: number | null
  roc_auc: number | null
  pr_auc: number | null
  pr_auc_baseline?: number
  alerts: number
  alert_rate: number | null
  undefined: string[]
}

export interface RunSummary {
  id: string
  seq: number
  algorithm: string
  algorithm_name: string
  family: string
  kind: string
  score_type: string
  imbalance: string
  status: string
  error: string | null
  threshold?: number
  default_threshold?: number
  threshold_basis?: string
  train_seconds?: number
  inference_ms_per_1k?: number
  train_rows_used?: number
  train_rows_available?: number
  notes?: string[]
  params?: Record<string, Any>
  cv?: { params: Record<string, Any>; mean_score: number; std_score: number; scoring: string } | null
  calibrated_test_brier?: number | null
  metrics?: { train: Metrics; validation: Metrics; validation_default_threshold: Metrics; test: Metrics }
}

export interface Experiment {
  id: string
  dataset_id: string
  dataset_sha256: string
  schema_hash: string
  name: string
  mode: string
  status: string
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  config: Any
  progress: Any
  split: Any
  runs: RunSummary[]
  n_runs?: number
  n_completed?: number
}

export class ApiError extends Error {
  status: number
  detail: Any
  constructor(status: number, detail: Any) {
    super(typeof detail === 'string' ? detail : detail?.message ?? `Request failed (${status})`)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method,
    signal,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const text = await res.text()
  let data: Any = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data)
  return data as T
}

export const api = {
  get: <T = Any>(path: string, signal?: AbortSignal) => request<T>('GET', path, undefined, signal),
  post: <T = Any>(path: string, body?: unknown) => request<T>('POST', path, body ?? {}),
  put: <T = Any>(path: string, body?: unknown) => request<T>('PUT', path, body ?? {}),
  del: <T = Any>(path: string) => request<T>('DELETE', path),
}

export interface UploadResult {
  dataset: Dataset
  duplicate: boolean
  has_schema: boolean
}

/** Upload with real progress events (XMLHttpRequest exposes upload progress; fetch does not). */
export function uploadDataset(file: File, onProgress: (fraction: number) => void): { promise: Promise<UploadResult>; abort: () => void } {
  const xhr = new XMLHttpRequest()
  const promise = new Promise<UploadResult>((resolve, reject) => {
    xhr.open('POST', '/api/datasets/upload')
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.onload = () => {
      let data: Any = null
      try {
        data = JSON.parse(xhr.responseText)
      } catch {
        data = xhr.responseText
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data)
      else reject(new ApiError(xhr.status, data?.detail ?? data ?? 'Upload failed'))
    }
    xhr.onerror = () => reject(new ApiError(0, 'Network error — is the FTAD backend running?'))
    xhr.onabort = () => reject(new ApiError(0, 'Upload cancelled'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
  return { promise, abort: () => xhr.abort() }
}
