import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Any } from './api'

// Response cache keyed by URL. URLs always embed the dataset id, and the cache is
// cleared whenever the active dataset changes, so results of one dataset can never be
// rendered for another.
const cache = new Map<string, unknown>()

export function clearApiCache(prefix?: string) {
  if (!prefix) return cache.clear()
  for (const k of [...cache.keys()]) if (k.includes(prefix)) cache.delete(k)
}

export function useApi<T = Any>(path: string | null, opts: { cache?: boolean; poll?: number } = {}) {
  const useCache = opts.cache ?? true
  const [data, setData] = useState<T | undefined>(() => (path && useCache ? (cache.get(path) as T | undefined) : undefined))
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState<boolean>(!!path && !(useCache && path && cache.has(path)))
  const current = useRef(path)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    current.current = path
    if (!path) {
      setData(undefined)
      setLoading(false)
      return
    }
    if (useCache && cache.has(path) && tick === 0) {
      setData(cache.get(path) as T)
      setLoading(false)
      setError(null)
      return
    }
    const ctl = new AbortController()
    if (!(useCache && cache.has(path))) setData(undefined)
    setLoading(true)
    setError(null)
    api
      .get<T>(path, ctl.signal)
      .then((d) => {
        if (current.current !== path) return // stale response for a previous dataset/query
        if (useCache) cache.set(path, d)
        setData(d)
      })
      .catch((e) => {
        if (e?.name !== 'AbortError' && current.current === path) setError(e)
      })
      .finally(() => current.current === path && setLoading(false))
    return () => ctl.abort()
  }, [path, tick, useCache])

  useEffect(() => {
    if (!opts.poll || !path) return
    const h = setInterval(() => setTick((t) => t + 1), opts.poll)
    return () => clearInterval(h)
  }, [opts.poll, path])

  const reload = useCallback(() => {
    if (path) cache.delete(path)
    setTick((t) => t + 1)
  }, [path])

  return { data, error, loading, reload }
}

export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const h = setTimeout(() => setV(value), ms)
    return () => clearTimeout(h)
  }, [value, ms])
  return v
}

export function useLocalState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      return raw ? (JSON.parse(raw) as T) : initial
    } catch {
      return initial
    }
  })
  const set = useCallback(
    (nv: T) => {
      setV(nv)
      try {
        localStorage.setItem(key, JSON.stringify(nv))
      } catch {
        /* storage unavailable */
      }
    },
    [key],
  )
  return [v, set]
}
