/**
 * Central fetch wrapper that attaches the StoMar API key (X-API-Key) to every
 * request. Protected endpoints (paper-trading, ledger, pipeline, automation,
 * risk-guard) reject requests without a valid key.
 *
 * The key is read from VITE_STOMAR_API_KEY at build time, or from a
 * runtime-injected global (window.__STOMAR_API_KEY__) so deployments can keep
 * it out of the static bundle entirely.
 */
const API_KEY =
  import.meta.env.VITE_STOMAR_API_KEY ??
  (typeof window !== 'undefined' ? window.__STOMAR_API_KEY__ : undefined)

export function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  if (API_KEY) {
    init = {
      ...init,
      headers: {
        ...(init.headers ?? {}),
        'X-API-Key': API_KEY,
      },
    }
  }
  return fetch(url, init)
}
