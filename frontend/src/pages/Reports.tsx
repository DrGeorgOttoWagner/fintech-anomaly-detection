import { Download, ExternalLink, FileText, FolderOutput } from 'lucide-react'
import { useState } from 'react'
import { Button, Callout, Card, ErrorBox, NeedSchema, PageHeader, Select, Spinner } from '../components/ui'
import { api, type Any, type Experiment } from '../lib/api'
import { fmtBytes, fmtDate } from '../lib/format'
import { useApi } from '../lib/hooks'
import { useSession } from '../lib/session'

export default function Reports() {
  const { dataset, schema } = useSession()
  const exps = useApi<Experiment[]>(dataset ? `/api/experiments?dataset_id=${dataset.id}` : null, { cache: false })
  const reports = useApi<Any[]>(dataset ? `/api/reports?dataset_id=${dataset.id}` : null, { cache: false })
  const config = useApi<Any>('/api/reports/config')
  const [expId, setExpId] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<unknown>(null)
  const [current, setCurrent] = useState<Any>(null)
  const [exported, setExported] = useState<Any>(null)
  const [exporting, setExporting] = useState(false)

  if (!dataset || !schema) return <><PageHeader eyebrow="Step 6 · Report" title="Research Reports" /><NeedSchema hasDataset={!!dataset} /></>
  const eligible = (exps.data ?? []).filter((e) => e.status.startsWith('completed') && e.schema_hash === schema.hash)

  const generate = async () => {
    setBusy(true); setErr(null); setExported(null)
    try {
      const r = await api.post('/api/reports', { dataset_id: dataset.id, experiment_id: expId || null })
      setCurrent(r)
      reports.reload()
    } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  const open = async (id: string) => { setExported(null); setCurrent(await api.get(`/api/reports/${id}`)) }
  const exportIt = async () => {
    setExporting(true); setErr(null)
    try { setExported(await api.post(`/api/reports/${current.id}/export`)) } catch (e) { setErr(e) } finally { setExporting(false) }
  }

  return (
    <div>
      <PageHeader eyebrow="Step 6 · Report" title="Research Reports" description="Generate a reproducible research bundle: Markdown and HTML report, CSV tables, PNG/SVG figures and a manifest. Every value is taken from computations on the uploaded data." />
      <div className="grid gap-6 xl:grid-cols-[400px_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card title="Generate report">
            <div className="space-y-3">
              <Select className="w-full" value={expId} onChange={setExpId} ariaLabel="Experiment to include" options={[{ value: '', label: 'Descriptive & EDA only (no experiment)' }, ...eligible.map((e) => ({ value: e.id, label: `+ ${e.name}` }))]} />
              <p className="text-xs text-dim">Sections: dataset description, data quality, class distribution, descriptive statistics, EDA figures, imbalance discussion{expId ? ', splitting, model selection, evaluation methodology, comparative results' : ''}, findings, limitations{schema.config.domain === 'fraud' ? ' and an analysis-coverage appendix' : ''}.</p>
              <Button variant="primary" className="w-full" onClick={generate} loading={busy} data-testid="generate-report"><FileText className="size-4" />Generate research report</Button>
              <ErrorBox error={err} />
            </div>
          </Card>
          <Card title="Export folder" subtitle="One-click export copies a bundle to a local folder you configure — never anywhere else.">
            {config.data?.export_dir ? (
              <div className="text-xs text-muted">Configured: <code className="break-all text-text">{config.data.export_dir}/ftad_exports</code>{!config.data.export_dir_exists && <div className="mt-1 text-warn">The folder does not exist.</div>}</div>
            ) : <div className="text-xs text-muted">Not configured. {config.data?.how_to_configure} Until then, use “Download bundle”.</div>}
          </Card>
          <Card title="Previous reports" pad={false}>
            <div className="divide-y divide-line">
              {!reports.data ? <div className="p-4"><Spinner /></div> : reports.data.length === 0 ? <p className="p-4 text-sm text-dim">None yet.</p> : reports.data.map((r) => (
                <button key={r.id} onClick={() => open(r.id)} className="block w-full px-4 py-2.5 text-left hover:bg-panel-2">
                  <div className="text-sm">{r.experiment_id ? 'EDA + experiment' : 'EDA only'} <span className="num text-xs text-dim">· {r.id}</span></div>
                  <div className="text-[11px] text-dim">{fmtDate(r.created_at)}</div>
                </button>
              ))}
            </div>
          </Card>
        </div>
        <div className="space-y-6">
          {current ? (
            <>
              <Card title={current.title} subtitle={`${current.id} · ${fmtDate(current.created_at)}`}
                actions={
                  <>
                    <a href={`/api/reports/${current.id}/files/report.html`} target="_blank" rel="noreferrer"><Button size="sm"><ExternalLink className="size-3.5" />Open HTML</Button></a>
                    <a href={`/api/reports/${current.id}/bundle.zip`} download><Button size="sm"><Download className="size-3.5" />Download bundle</Button></a>
                    <Button size="sm" variant="primary" onClick={exportIt} loading={exporting} disabled={!config.data?.export_dir_exists} data-testid="export-report"><FolderOutput className="size-3.5" />Export to folder</Button>
                  </>
                }>
                {exported && <div className="mb-4"><Callout tone="ok" title="Exported">{exported.files} files copied to <code className="break-all">{exported.exported_to}</code></Callout></div>}
                <iframe title="Report preview" src={`/api/reports/${current.id}/files/report.html`} className="h-[720px] w-full rounded-lg border border-line bg-white" />
              </Card>
              <Card title="Bundle contents" pad={false}>
                <div className="grid gap-x-6 p-4 text-xs md:grid-cols-2">
                  {current.files.map((f: Any) => (
                    <a key={f.path} href={`/api/reports/${current.id}/files/${f.path}`} target="_blank" rel="noreferrer" className="flex justify-between border-b border-line/50 py-1.5 text-muted hover:text-accent">
                      <span className="truncate">{f.path}</span><span className="num text-dim">{fmtBytes(f.bytes)}</span>
                    </a>
                  ))}
                </div>
              </Card>
            </>
          ) : (
            <Card><p className="py-10 text-center text-sm text-dim">Generate a report or open a previous one to preview it here.</p></Card>
          )}
        </div>
      </div>
    </div>
  )
}
