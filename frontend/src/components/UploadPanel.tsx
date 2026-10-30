import clsx from 'clsx'
import { CheckCircle2, FileSpreadsheet, FolderOpen, Loader2, UploadCloud, X } from 'lucide-react'
import { useRef, useState } from 'react'
import { uploadDataset, type UploadResult } from '../lib/api'
import { fmtBytes, fmtInt } from '../lib/format'
import { Button, Callout } from './ui'

const ACCEPT = '.csv,.parquet,.pq,text/csv,application/vnd.apache.parquet'

type Phase = { kind: 'idle' } | { kind: 'uploading'; file: File; progress: number } | { kind: 'processing'; file: File } | { kind: 'done'; result: UploadResult } | { kind: 'error'; message: string; file?: File }

export function UploadPanel({ onUploaded, large, compact }: { onUploaded: (r: UploadResult) => void; large?: boolean; compact?: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  const abortRef = useRef<() => void>(() => {})
  const [drag, setDrag] = useState(false)
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })

  const start = (file: File) => {
    const ext = file.name.toLowerCase().split('.').pop()
    if (!['csv', 'parquet', 'pq'].includes(ext ?? '')) {
      setPhase({ kind: 'error', message: `“${file.name}” is not a CSV or Parquet file.`, file })
      return
    }
    setPhase({ kind: 'uploading', file, progress: 0 })
    const { promise, abort } = uploadDataset(file, (p) => setPhase((cur) => (cur.kind === 'uploading' ? { ...cur, progress: p } : p >= 1 ? cur : cur)))
    abortRef.current = abort
    const tick = setInterval(() => setPhase((cur) => (cur.kind === 'uploading' && cur.progress >= 0.999 ? { kind: 'processing', file } : cur)), 150)
    promise
      .then((result) => {
        setPhase({ kind: 'done', result })
        onUploaded(result)
      })
      .catch((e) => setPhase({ kind: 'error', message: e.message, file }))
      .finally(() => clearInterval(tick))
  }

  const busy = phase.kind === 'uploading' || phase.kind === 'processing'

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault()
          if (!busy) setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDrag(false)
          const f = e.dataTransfer.files?.[0]
          if (f && !busy) start(f)
        }}
        className={clsx(
          'relative flex flex-col items-center justify-center rounded-2xl border border-dashed text-center transition-colors',
          large ? 'px-8 py-16' : compact ? 'px-5 py-6' : 'px-6 py-10',
          drag ? 'border-accent bg-accent/[0.06]' : 'border-line-2 bg-panel/70',
        )}
        data-testid="dropzone"
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          className="hidden"
          data-testid="file-input"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) start(f)
            e.target.value = ''
          }}
        />
        {phase.kind === 'idle' || phase.kind === 'error' || phase.kind === 'done' ? (
          <>
            <div className={clsx('mb-4 flex items-center justify-center rounded-2xl border border-line-2 bg-panel-2', large ? 'size-16' : 'size-12')}>
              <UploadCloud className={clsx('text-accent', large ? 'size-8' : 'size-6')} />
            </div>
            <div className={clsx('font-display font-semibold tracking-tight', large ? 'text-2xl' : 'text-lg')}>Upload Your Dataset</div>
            <p className="mt-1.5 max-w-md text-sm text-muted">
              Choose a CSV or Parquet file from your computer, or drag it here. The file is copied to the local FTAD backend only — nothing is sent anywhere else.
            </p>
            <div className="mt-5 flex items-center gap-3">
              <Button variant="primary" onClick={() => input.current?.click()} data-testid="browse-files">
                <FolderOpen className="size-4" /> Browse Files
              </Button>
            </div>
            <div className="mt-3 text-[11px] text-dim">CSV (comma, semicolon, tab or pipe delimited) · Parquet · up to the configured size limit</div>
          </>
        ) : (
          <div className="w-full max-w-md">
            <div className="mb-3 flex items-center gap-3 text-left">
              <FileSpreadsheet className="size-8 shrink-0 text-accent" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{phase.file.name}</div>
                <div className="text-xs text-dim">{fmtBytes(phase.file.size)}</div>
              </div>
              {phase.kind === 'uploading' && (
                <button className="text-dim hover:text-text" onClick={() => abortRef.current()} aria-label="Cancel upload"><X className="size-4" /></button>
              )}
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-panel-3">
              <div className={clsx('h-full rounded-full bg-accent transition-[width] duration-150', phase.kind === 'processing' && 'animate-pulse')} style={{ width: `${phase.kind === 'processing' ? 100 : Math.round(phase.progress * 100)}%` }} />
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-muted" data-testid="upload-status">
              {phase.kind === 'uploading' ? <span>Uploading… {Math.round(phase.progress * 100)}%</span> : <span className="flex items-center gap-1.5"><Loader2 className="size-3 animate-spin" />Validating, fingerprinting and parsing…</span>}
            </div>
          </div>
        )}
      </div>
      {phase.kind === 'error' && (
        <div className="mt-3" data-testid="upload-error"><Callout tone="error" title="The file could not be loaded">{phase.message}</Callout></div>
      )}
      {phase.kind === 'done' && (
        <div className="mt-3" data-testid="upload-done">
          <Callout tone="ok" title={phase.result.duplicate ? 'This file was uploaded before' : 'Dataset uploaded'}>
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="size-3.5" />
              {phase.result.dataset.original_filename}: {fmtInt(phase.result.dataset.n_rows)} rows × {phase.result.dataset.n_cols} columns · SHA-256 {phase.result.dataset.sha256.slice(0, 12)}…
              {phase.result.duplicate && ' — identical fingerprint, existing dataset reopened.'}
            </span>
          </Callout>
        </div>
      )}
    </div>
  )
}
