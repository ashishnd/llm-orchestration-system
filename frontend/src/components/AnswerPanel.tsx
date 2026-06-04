import { Badge, Card, ShimmerLines } from './ui/Card'

interface AnswerPanelProps {
  answer: string | null
  loading: boolean
}

export function AnswerPanel({ answer, loading }: AnswerPanelProps) {
  const showContent = loading || answer

  return (
    <Card
      title="Final answer"
      subtitle="Synthesized response with corpus grounding"
      accent={!!answer}
      processing={loading && !answer}
    >
      {loading && !answer && (
        <div className="animate-fade-in">
          <div className="mb-3 flex items-center gap-2">
            <Badge variant="running">Synthesizing</Badge>
          </div>
          <ShimmerLines rows={4} />
          <p className="mt-4 text-xs text-muted">Critique and synthesis agents are merging results…</p>
        </div>
      )}

      {!loading && !answer && (
        <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50/80 px-4 py-6 text-center">
          <p className="text-sm text-muted">The synthesized answer will appear here after the pipeline completes.</p>
        </div>
      )}

      {answer && (
        <div className="animate-fade-in">
          <div className="mb-3">
            <Badge variant="success">Complete</Badge>
          </div>
          <div className="rounded-xl border border-emerald-100 bg-gradient-to-br from-white to-brand-50/40 p-4 text-sm leading-relaxed text-slate-700">
            {answer.split(/\n\n+/).map((para, i) => (
              <p key={i} className="mb-3 last:mb-0">
                {para}
              </p>
            ))}
          </div>
        </div>
      )}

      {!showContent && null}
    </Card>
  )
}
