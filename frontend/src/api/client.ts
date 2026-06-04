import { API_BASE } from './config'
import type { TraceResponse } from '../types'

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/healthz`)
    return res.ok
  } catch {
    return false
  }
}

export async function fetchTrace(jobId: string): Promise<TraceResponse> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/trace`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.message ?? `Failed to load trace (${res.status})`)
  }
  return res.json()
}
