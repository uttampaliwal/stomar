import { useState, useEffect, useCallback, useRef } from 'react'

interface UseApiOptions {
  immediate?: boolean
  staleTime?: number
}

const _inflight = new Map<string, Promise<unknown>>()
const _cache = new Map<string, { data: unknown; ts: number }>()
const DEFAULT_STALE_MS = 30_000

export function useApi<T>(url: string, options: UseApiOptions = {}) {
  const [data, setData] = useState<T | null>(() => {
    const c = _cache.get(url)
    if (c && Date.now() - c.ts < (options.staleTime ?? DEFAULT_STALE_MS)) {
      return c.data as T
    }
    return null
  })
  const [loading, setLoading] = useState(() => options.immediate !== false && !data)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(true)

  const fetchData = useCallback(() => {
    // Deduplicate in-flight requests
    let promise = _inflight.get(url)
    if (!promise) {
      promise = fetch(url).then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      _inflight.set(url, promise)
    }
    return promise
      .then((json) => {
        _inflight.delete(url)
        if (mountedRef.current) {
          setData(json as T)
          setError(null)
          _cache.set(url, { data: json, ts: Date.now() })
        }
      })
      .catch((err: unknown) => {
        _inflight.delete(url)
        if (mountedRef.current) {
          setError(err instanceof Error ? err.message : 'Unknown error')
        }
      })
      .finally(() => {
        if (mountedRef.current) setLoading(false)
      })
  }, [url])

  useEffect(() => {
    mountedRef.current = true
    if (options.immediate !== false) {
      fetchData()
    }
    return () => { mountedRef.current = false }
  }, [fetchData, options.immediate])

  return { data, loading, error, refetch: fetchData }
}

export function usePostApi<T>(url: string) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const post = useCallback(async (body: unknown) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json: unknown = await res.json()
      setData(json as T)
      return json as T
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      return null
    } finally {
      setLoading(false)
    }
  }, [url])

  return { data, loading, error, post }
}
