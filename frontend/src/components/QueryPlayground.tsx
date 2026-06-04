import { EXAMPLE_PROMPTS } from '../types'
import { Badge, Card } from './ui/Card'

interface QueryPlaygroundProps {
  query: string
  running: boolean
  onQueryChange: (q: string) => void
  onSubmit: () => void
}

export function QueryPlayground({ query, running, onQueryChange, onSubmit }: QueryPlaygroundProps) {
  return (
    <Card
      title="Ask a question"
      subtitle="Queries run against the arXiv cs.CL paper corpus"
      processing={running}
    >
      <textarea
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        disabled={running}
        rows={3}
        maxLength={2000}
        placeholder="What is BERT and how does it differ from GPT?"
        className="w-full resize-y rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800 shadow-inner transition-all placeholder:text-slate-400 focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-200 disabled:bg-slate-50 disabled:opacity-70"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted">Examples:</span>
        {EXAMPLE_PROMPTS.map((ex) => (
          <button
            key={ex.label}
            type="button"
            disabled={running}
            onClick={() => onQueryChange(ex.query)}
            className="rounded-full border border-emerald-200 bg-brand-50 px-3 py-1 text-xs font-medium text-brand-800 transition-all duration-200 hover:border-brand-400 hover:bg-brand-100 hover:shadow-sm disabled:opacity-50"
          >
            {ex.label}
          </button>
        ))}
      </div>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onSubmit}
          disabled={running || !query.trim()}
          className="rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-all duration-200 hover:bg-brand-700 hover:shadow-md active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
        >
          {running ? (
            <span className="flex items-center gap-2">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              Agents working…
            </span>
          ) : (
            'Run pipeline'
          )}
        </button>
        {running && <Badge variant="running">Processing</Badge>}
      </div>
    </Card>
  )
}
