import { Badge } from './ui/Card'

interface StatusStripProps {
  healthy: boolean | null
  jobId: string | null
}

export function StatusStrip({ healthy, jobId }: StatusStripProps) {
  return (
    <header className="sticky top-0 z-10 border-b border-emerald-100/80 bg-white/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3.5">
        <div className="animate-fade-in">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-sm font-bold text-white shadow-sm">
              M
            </span>
            <div>
              <h1 className="text-base font-semibold tracking-tight text-slate-900">
                Multi-Agent LLM Orchestrator
              </h1>
              <p className="text-xs text-muted">
                arXiv cs.CL · decompose → retrieve → critique → synthesize
              </p>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <Badge variant={healthy ? 'success' : healthy === null ? 'muted' : 'error'}>
            <span className="flex items-center gap-1.5">
              <span
                className={`inline-block h-1.5 w-1.5 rounded-full ${
                  healthy === null ? 'bg-slate-400' : healthy ? 'bg-emerald-500' : 'bg-red-500'
                }`}
              />
              {healthy === null ? 'Checking API…' : healthy ? 'API connected' : 'API unreachable'}
            </span>
          </Badge>
          {jobId && (
            <span className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 font-mono text-xs text-slate-600">
              job{' '}
              <button
                type="button"
                className="font-semibold text-brand-700 transition hover:text-brand-600"
                onClick={() => navigator.clipboard.writeText(jobId)}
                title="Copy job ID"
              >
                {jobId}
              </button>
            </span>
          )}
        </div>
      </div>
    </header>
  )
}
