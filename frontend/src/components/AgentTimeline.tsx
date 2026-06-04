import type { TimelineEntry } from '../types'
import { Card } from './ui/Card'
import { PipelineStepper } from './PipelineStepper'

const AGENT_ICONS: Record<string, string> = {
  decomposition: '◈',
  retrieval: '◎',
  critique: '◐',
  synthesis: '◆',
  compression: '◇',
  routing: '→',
  job_started: '▶',
  final_answer: '✓',
}

function iconFor(entry: TimelineEntry): string {
  if (entry.kind === 'tool_start' || entry.kind === 'tool_complete') return '⚙'
  if (entry.kind === 'budget') return '◷'
  if (entry.kind === 'error') return '!'
  return AGENT_ICONS[entry.label.toLowerCase()] ?? AGENT_ICONS[entry.kind] ?? '●'
}

function cardStyles(status?: TimelineEntry['status']): string {
  if (status === 'running') {
    return 'shimmer-surface border-brand-300 bg-brand-50/90 shadow-sm'
  }
  if (status === 'error') {
    return 'border-red-200 bg-red-50/90'
  }
  if (status === 'done') {
    return 'border-emerald-100 bg-white hover:border-emerald-200'
  }
  return 'border-slate-100 bg-white hover:border-slate-200'
}

interface AgentTimelineProps {
  entries: TimelineEntry[]
  running: boolean
}

export function AgentTimeline({ entries, running }: AgentTimelineProps) {
  return (
    <Card
      title="Live pipeline"
      subtitle="Agents and tools as they execute"
      processing={running}
      accent
      className="flex h-full min-h-[420px] flex-col"
    >
      <PipelineStepper entries={entries} running={running} />

      <div className="flex-1 space-y-2 overflow-y-auto pr-1">
        {entries.length === 0 && !running && (
          <div className="animate-fade-in rounded-xl border border-dashed border-emerald-200 bg-brand-50/50 px-4 py-8 text-center">
            <p className="text-sm font-medium text-brand-800">Ready when you are</p>
            <p className="mt-1 text-xs text-muted">
              Submit a query to watch the orchestrator route work across agents.
            </p>
          </div>
        )}

        {entries.length === 0 && running && (
          <div className="animate-fade-in space-y-2 rounded-xl border border-brand-200 bg-brand-50/60 p-4">
            <p className="text-sm font-medium text-brand-800">Starting pipeline…</p>
            <div className="shimmer-bar h-2 w-full rounded-full" />
            <div className="shimmer-bar h-2 w-4/5 rounded-full" style={{ animationDelay: '0.2s' }} />
          </div>
        )}

        {entries.map((entry, index) => (
          <div
            key={entry.id}
            style={{ animationDelay: `${Math.min(index * 0.04, 0.3)}s` }}
            className={`animate-fade-slide-in flex gap-3 rounded-xl border px-3.5 py-3 text-sm transition-all duration-300 ${cardStyles(entry.status)}`}
          >
            <span
              className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-sm transition-colors ${
                entry.status === 'running'
                  ? 'bg-brand-200 text-brand-800'
                  : entry.status === 'error'
                    ? 'bg-red-100 text-red-700'
                    : 'bg-emerald-100 text-emerald-700'
              }`}
            >
              {iconFor(entry)}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-medium capitalize text-slate-800">{entry.label}</span>
                {entry.ts && (
                  <time className="shrink-0 font-mono text-[10px] text-slate-400">
                    {entry.ts.slice(11, 19)}
                  </time>
                )}
              </div>
              {entry.detail && (
                <p className="mt-0.5 break-words text-xs leading-relaxed text-slate-600">
                  {entry.detail}
                </p>
              )}
            </div>
            {entry.status === 'running' && (
              <span className="relative mt-1 flex h-2.5 w-2.5 shrink-0">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-400 opacity-60" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-brand-500" />
              </span>
            )}
          </div>
        ))}

        {running && entries.length > 0 && (
          <div className="flex items-center justify-center gap-2 py-2 text-xs text-brand-700">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-brand-500" />
            Waiting for next event…
          </div>
        )}
      </div>
    </Card>
  )
}
