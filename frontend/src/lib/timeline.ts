import type { SsePayload, TimelineEntry } from '../types'

let seq = 0

function id() {
  seq += 1
  return `tl-${seq}`
}

export function sseToTimelineEntry(eventType: string, data: SsePayload): TimelineEntry | null {
  const ts = data.ts

  if (eventType === 'job_started') {
    const jobId = data.job_id ?? (data as { job_id?: string }).job_id
    return {
      id: id(),
      kind: 'job_started',
      label: 'Job started',
      detail: jobId ? `ID: ${jobId}` : undefined,
      ts,
      status: 'done',
    }
  }

  switch (eventType) {
    case 'agent_start':
      return {
        id: id(),
        kind: 'agent_start',
        label: data.agent ?? 'agent',
        detail: 'Running…',
        ts,
        status: 'running',
      }
    case 'agent_complete':
      return {
        id: id(),
        kind: 'agent_complete',
        label: data.agent ?? 'agent',
        detail: data.content ? truncate(data.content, 120) : 'Complete',
        ts,
        status: 'done',
      }
    case 'tool_start':
      return {
        id: id(),
        kind: 'tool_start',
        label: data.tool ?? 'tool',
        detail: 'Invoking…',
        ts,
        status: 'running',
      }
    case 'tool_complete':
      return {
        id: id(),
        kind: 'tool_complete',
        label: data.tool ?? 'tool',
        detail: data.latency_ms != null ? `${data.status ?? 'done'} · ${Math.round(data.latency_ms)}ms` : data.status,
        ts,
        status: data.status === 'error' ? 'error' : 'done',
      }
    case 'routing':
      return {
        id: id(),
        kind: 'routing',
        label: 'Routing',
        detail: data.next_agent
          ? `Next: ${data.next_agent} — ${truncate(data.justification ?? '', 100)}`
          : truncate(data.justification ?? 'Pipeline complete', 120),
        ts,
        status: 'done',
      }
    case 'budget_status':
      return {
        id: id(),
        kind: 'budget',
        label: `Budget · ${data.agent ?? 'agent'}`,
        detail:
          data.used_tokens != null && data.max_tokens != null
            ? `${data.used_tokens} / ${data.max_tokens} tokens`
            : undefined,
        ts,
        status: 'done',
      }
    case 'final_answer':
      return {
        id: id(),
        kind: 'final_answer',
        label: 'Final answer ready',
        ts,
        status: 'done',
      }
    case 'error':
      return {
        id: id(),
        kind: 'error',
        label: 'Error',
        detail: data.message,
        ts,
        status: 'error',
      }
    default:
      return null
  }
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : `${s.slice(0, max)}…`
}

export function resetTimelineSeq() {
  seq = 0
}
