import type { ReactNode } from 'react'
import { Card, ShimmerLines } from './ui/Card'
import type { SharedContext } from '../types'

interface TraceSummaryProps {
  context: SharedContext | null
  loading: boolean
}

function Section({
  title,
  open,
  children,
}: {
  title: string
  open?: boolean
  children: ReactNode
}) {
  return (
    <details
      open={open}
      className="group overflow-hidden rounded-xl border border-slate-100 bg-white transition-all duration-300 open:border-emerald-200 open:shadow-sm"
    >
      <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-slate-700 transition hover:bg-brand-50/50 hover:text-brand-800">
        {title}
      </summary>
      <div className="border-t border-slate-100 px-4 py-3 text-sm text-slate-600">{children}</div>
    </details>
  )
}

export function TraceSummary({ context, loading }: TraceSummaryProps) {
  if (loading) {
    return (
      <Card title="Execution trace" subtitle="Decomposition, retrieval, and provenance" processing>
        <ShimmerLines rows={5} />
      </Card>
    )
  }

  if (!context) {
    return (
      <Card title="Execution trace" subtitle="Decomposition, retrieval, and provenance">
        <div className="rounded-xl border border-dashed border-emerald-200 bg-brand-50/40 px-4 py-6 text-center">
          <p className="text-sm text-muted">
            After a run completes, decomposition, retrieval hops, and provenance appear here.
          </p>
        </div>
      </Card>
    )
  }

  const subTasks = context.sub_tasks ?? []
  const retrieval = context.agent_outputs?.find(
    (o) => o.structured_output?.type === 'retrieval',
  )
  const hops = retrieval?.structured_output?.hops ?? []
  const synthesis = context.agent_outputs?.find(
    (o) => o.structured_output?.type === 'synthesis',
  )
  const provenance = synthesis?.structured_output?.provenance ?? []

  return (
    <Card
      title="Execution trace"
      subtitle="Full pipeline artifacts from SharedContext"
      accent
      className="animate-fade-in"
    >
      <div className="space-y-2">
        <Section title={`Decomposition (${subTasks.length} sub-tasks)`} open>
          {subTasks.length === 0 ? (
            <p className="text-muted">No sub-tasks recorded.</p>
          ) : (
            <ul className="space-y-2">
              {subTasks.map((t) => (
                <li
                  key={t.id}
                  className="flex flex-wrap items-start gap-2 rounded-lg border border-slate-100 bg-slate-50/50 px-3 py-2 transition hover:border-emerald-200"
                >
                  <span className="shrink-0 rounded-md bg-brand-100 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase text-brand-800">
                    {t.task_type}
                  </span>
                  <span className="flex-1 text-slate-700">{t.description}</span>
                  <span className="text-xs font-medium capitalize text-slate-400">{t.status}</span>
                </li>
              ))}
            </ul>
          )}
        </Section>

        <Section title={`Retrieval (${hops.length} hops)`}>
          {hops.length === 0 ? (
            <p className="text-muted">No retrieval hops recorded.</p>
          ) : (
            <ul className="space-y-3">
              {hops.map((h) => (
                <li
                  key={h.hop_index}
                  className="rounded-lg border border-slate-100 bg-gradient-to-r from-white to-brand-50/30 px-3 py-2.5"
                >
                  <p className="text-xs font-semibold uppercase tracking-wide text-brand-700">
                    Hop {h.hop_index + 1}
                  </p>
                  <p className="mt-1 text-slate-700">{h.query}</p>
                  <p className="mt-1.5 text-xs text-muted">
                    {h.retrieved_chunk_ids.length} chunk(s) retrieved
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Section>

        <Section title={`Provenance (${provenance.length} sentences)`}>
          {provenance.length === 0 ? (
            <p className="text-muted">No provenance map in synthesis output.</p>
          ) : (
            <ul className="space-y-2">
              {provenance.slice(0, 8).map((p) => (
                <li
                  key={p.sentence_index}
                  className="border-l-2 border-brand-300 pl-3 transition hover:border-brand-500"
                >
                  <span className="text-xs font-semibold text-brand-700">{p.source_agent}</span>
                  <p className="text-slate-700">
                    {p.text_span?.text ?? `Sentence ${p.sentence_index + 1}`}
                  </p>
                  {p.citations && p.citations.length > 0 && (
                    <p className="mt-0.5 text-xs text-muted">{p.citations.length} citation(s)</p>
                  )}
                </li>
              ))}
              {provenance.length > 8 && (
                <li className="text-xs text-muted">+ {provenance.length - 8} more</li>
              )}
            </ul>
          )}
        </Section>
      </div>
    </Card>
  )
}
