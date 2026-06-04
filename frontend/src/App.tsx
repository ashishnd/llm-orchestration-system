import { useCallback, useEffect, useRef, useState } from 'react'
import { checkHealth, fetchTrace } from './api/client'
import { submitJobStream } from './api/sse'
import { AgentTimeline } from './components/AgentTimeline'
import { AnswerPanel } from './components/AnswerPanel'
import { QueryPlayground } from './components/QueryPlayground'
import { StatusStrip } from './components/StatusStrip'
import { TraceSummary } from './components/TraceSummary'
import { resetTimelineSeq, sseToTimelineEntry } from './lib/timeline'
import { upsertTimelineEntry } from './lib/timelineMerge'
import type { SharedContext, TimelineEntry } from './types'

const LAST_JOB_KEY = 'mao_last_job_id'

export default function App() {
  const [healthy, setHealthy] = useState<boolean | null>(null)
  const [query, setQuery] = useState('')
  const [running, setRunning] = useState(false)
  const [jobId, setJobId] = useState<string | null>(() => localStorage.getItem(LAST_JOB_KEY))
  const [timeline, setTimeline] = useState<TimelineEntry[]>([])
  const [answer, setAnswer] = useState<string | null>(null)
  const [trace, setTrace] = useState<SharedContext | null>(null)
  const [traceLoading, setTraceLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    checkHealth().then(setHealthy)
    const interval = setInterval(() => checkHealth().then(setHealthy), 30_000)
    return () => clearInterval(interval)
  }, [])

  const loadTrace = useCallback(async (id: string) => {
    setTraceLoading(true)
    try {
      const res = await fetchTrace(id)
      setTrace(res.context)
      if (res.context.final_answer) setAnswer(res.context.final_answer)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load trace')
    } finally {
      setTraceLoading(false)
    }
  }, [])

  useEffect(() => {
    if (jobId && !trace && !running) {
      loadTrace(jobId)
    }
  }, [jobId, trace, running, loadTrace])

  const handleSubmit = async () => {
    const trimmed = query.trim()
    if (!trimmed || running) return

    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac

    resetTimelineSeq()
    setRunning(true)
    setError(null)
    setTimeline([])
    setAnswer(null)
    setTrace(null)
    setJobId(null)

    let currentJobId: string | null = null

    try {
      await submitJobStream(
        trimmed,
        (eventType, data) => {
          if (eventType === 'job_started' && data.job_id) {
            currentJobId = data.job_id
            setJobId(data.job_id)
            localStorage.setItem(LAST_JOB_KEY, data.job_id)
          }
          if (eventType === 'final_answer' && data.answer != null) {
            setAnswer(data.answer)
          }
          const payloadType = typeof data.type === 'string' ? data.type : eventType
          const entry = sseToTimelineEntry(payloadType, data)
          if (entry) setTimeline((prev) => upsertTimelineEntry(prev, entry))
        },
        ac.signal,
      )

      if (currentJobId) await loadTrace(currentJobId)
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') return
      setError(e instanceof Error ? e.message : 'Pipeline failed')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="min-h-screen">
      <StatusStrip healthy={healthy} jobId={jobId} />
      <main className="mx-auto max-w-6xl px-4 py-8">
        {error && (
          <div className="animate-fade-slide-in mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 shadow-sm">
            {error}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
          <div className="space-y-6">
            <QueryPlayground
              query={query}
              running={running}
              onQueryChange={setQuery}
              onSubmit={handleSubmit}
            />
            <AnswerPanel answer={answer} loading={running && !answer} />
          </div>
          <AgentTimeline entries={timeline} running={running} />
        </div>

        <div className="mt-6">
          <TraceSummary context={trace} loading={traceLoading && running === false && !trace} />
        </div>

        <footer className="mt-10 border-t border-emerald-100 pt-5 text-center text-xs text-muted">
          Eval harness, prompt rewrites, and batch ops run via CLI — see README.
        </footer>
      </main>
    </div>
  )
}
