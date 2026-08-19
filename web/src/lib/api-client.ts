/**
 * Central fetch wrapper. The browser authenticates with a session cookie set
 * by POST /api/auth/login (HttpOnly, same-origin via the Vite dev proxy), so
 * no credentials are attached here. Non-browser clients (CLI, scripts, curl)
 * still authenticate with the X-API-Key header — see README.
 */
export function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  return fetch(url, {
    ...init,
    credentials: 'same-origin',
  })
}