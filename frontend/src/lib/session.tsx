import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, type Dataset, type SchemaConfig, type Session } from './api'
import { clearApiCache } from './hooks'

interface SessionCtx {
  ready: boolean
  dataset: Dataset | null
  schema: { id: string; hash: string; config: SchemaConfig } | null
  refresh: () => Promise<void>
  activate: (datasetId: string | null) => Promise<void>
}

const Ctx = createContext<SessionCtx | null>(null)

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session>({ active_dataset: null, schema: null })
  const [ready, setReady] = useState(false)
  const lastId = useRef<string | null>(null)

  const apply = useCallback((s: Session) => {
    const id = s.active_dataset?.id ?? null
    if (id !== lastId.current) clearApiCache() // new dataset session: drop every cached result
    lastId.current = id
    setSession(s)
    setReady(true)
  }, [])

  const refresh = useCallback(async () => {
    apply(await api.get<Session>('/api/session'))
  }, [apply])

  const activate = useCallback(
    async (datasetId: string | null) => {
      apply(await api.post<Session>('/api/session/active', { dataset_id: datasetId }))
    },
    [apply],
  )

  useEffect(() => {
    refresh().catch(() => setReady(true))
  }, [refresh])

  const value = useMemo(
    () => ({ ready, dataset: session.active_dataset, schema: session.schema, refresh, activate }),
    [ready, session, refresh, activate],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useSession() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useSession outside SessionProvider')
  return v
}
