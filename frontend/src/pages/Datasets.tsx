import { ArrowRight, ChevronLeft, ChevronRight, Database, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { UploadPanel } from '../components/UploadPanel'
import { Badge, Button, Card, DataTable, EmptyState, ErrorBox, PageHeader, Spinner } from '../components/ui'
import { api, type Any, type Dataset } from '../lib/api'
import { fmtBytes, fmtDate, fmtInt } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

export function Preview({ datasetId }: { datasetId: string }) {
  const [offset, setOffset] = useState(0)
  const limit = 25
  const { data, error, loading } = useApi<Any>(`/api/datasets/${datasetId}/preview?offset=${offset}&limit=${limit}`)
  if (error) return <ErrorBox error={error} />
  if (!data) return loading ? <Spinner /> : null
  return (
    <div>
      <div className="overflow-auto rounded-lg border border-line" style={{ maxHeight: 460 }} data-testid="preview-table">
        <table className="w-full text-[12.5px]">
          <thead className="sticky top-0 bg-panel-2">
            <tr>
              <th className="border-b border-line px-2.5 py-2 text-left text-[11px] font-semibold text-dim">#</th>
              {data.columns.map((c: string) => (
                <th key={c} className="whitespace-nowrap border-b border-line px-2.5 py-2 text-left text-[11px] font-semibold text-muted">
                  {c}
                  <div className="font-normal text-dim">{data.dtypes[c]}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r: Any[], i: number) => (
              <tr key={i} className="border-b border-line/60">
                <td className="num px-2.5 py-1.5 text-dim">{data.row_index[i]}</td>
                {r.map((v, j) => (
                  <td key={j} className="num whitespace-nowrap px-2.5 py-1.5">{v === null ? <span className="text-dim">null</span> : typeof v === 'number' ? (Number.isInteger(v) ? v : v.toPrecision(6)) : String(v)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-muted">
        <span className="num">Rows {fmtInt(offset + 1)}–{fmtInt(Math.min(offset + limit, data.total))} of {fmtInt(data.total)}</span>
        <div className="flex gap-1.5">
          <Button size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}><ChevronLeft className="size-3.5" />Prev</Button>
          <Button size="sm" disabled={offset + limit >= data.total} onClick={() => setOffset(offset + limit)}>Next<ChevronRight className="size-3.5" /></Button>
        </div>
      </div>
    </div>
  )
}

export default function Datasets() {
  const { dataset, refresh, activate } = useSession()
  const nav = useNavigate()
  const list = useApi<Dataset[]>('/api/datasets', { cache: false })
  const [deleting, setDeleting] = useState<string | null>(null)
  const [err, setErr] = useState<unknown>(null)

  return (
    <div>
      <PageHeader eyebrow="Step 1 · Data" title="Datasets" description="Upload a CSV or Parquet file through your system's file picker. Each upload is fingerprinted (SHA-256), stored read-only, and becomes the active dataset session." />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div className="space-y-6">
          <UploadPanel
            large={!dataset}
            onUploaded={async () => {
              await refresh()
              list.reload()
            }}
          />
          {dataset && (
            <Card
              title={<span className="flex items-center gap-2">Preview · {dataset.original_filename}</span>}
              subtitle={`${fmtInt(dataset.n_rows)} rows × ${dataset.n_cols} columns · actual rows from the uploaded file`}
              actions={<Button variant="primary" size="sm" onClick={() => nav('/schema')}>Map schema <ArrowRight className="size-3.5" /></Button>}
            >
              <Preview datasetId={dataset.id} />
            </Card>
          )}
        </div>
        <Card title="Dataset registry" subtitle="Previously uploaded datasets can be reopened explicitly." pad={false}>
          <div className="p-4">
            <ErrorBox error={err || list.error} />
            {!list.data ? <Spinner /> : list.data.length === 0 ? (
              <EmptyState icon={<Database className="size-7" />} title="No datasets yet">Uploaded files will appear here.</EmptyState>
            ) : (
              <DataTable<Dataset>
                rows={list.data}
                rowKey={(r) => r.id}
                dense
                columns={[
                  {
                    key: 'name', header: 'Dataset', value: (r) => r.original_filename,
                    render: (r) => (
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5 font-medium">
                          <span className="truncate">{r.original_filename}</span>
                          {r.active && <Badge tone="accent">active</Badge>}
                        </div>
                        <div className="num text-[11px] text-dim">{fmtInt(r.n_rows)} × {r.n_cols} · {fmtBytes(r.size_bytes)} · {r.sha256.slice(0, 10)}</div>
                        <div className="text-[11px] text-dim">{fmtDate(r.created_at)} · {r.has_schema ? 'schema confirmed' : 'schema pending'} · {r.n_experiments} exp.</div>
                      </div>
                    ),
                  },
                  {
                    key: 'act', header: '', sortable: false,
                    render: (r) => (
                      <div className="flex justify-end gap-1.5">
                        {!r.active && <Button size="sm" onClick={async () => { await activate(r.id); list.reload() }}>Open</Button>}
                        <Button size="sm" variant="ghost" aria-label={`Delete ${r.original_filename}`} loading={deleting === r.id}
                          onClick={async () => {
                            if (!confirm(`Delete ${r.original_filename} and all of its experiments from FTAD? The original file on your disk is not touched.`)) return
                            setDeleting(r.id)
                            try { await api.del(`/api/datasets/${r.id}`); await refresh(); list.reload() } catch (e) { setErr(e) } finally { setDeleting(null) }
                          }}>
                          <Trash2 className="size-3.5" />
                        </Button>
                      </div>
                    ),
                  },
                ]}
              />
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
