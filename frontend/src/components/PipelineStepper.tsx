import type { TimelineEntry } from '../types'

const PIPELINE_STAGES = [
  { key: 'decomposition', label: 'Decompose', icon: '◈' },
  { key: 'retrieval', label: 'Retrieve', icon: '◎' },
  { key: 'critique', label: 'Critique', icon: '◐' },
  { key: 'synthesis', label: 'Synthesize', icon: '◆' },
] as const

type StageState = 'idle' | 'active' | 'done'

function deriveStageStates(entries: TimelineEntry[], running: boolean): StageState[] {
  const states: StageState[] = PIPELINE_STAGES.map(() => 'idle')
  let activeIndex = -1

  for (const entry of entries) {
    const agent = entry.label.toLowerCase()
    const idx = PIPELINE_STAGES.findIndex((s) => agent.includes(s.key))
    if (idx < 0) continue

    if (entry.kind === 'agent_complete' || entry.status === 'done') {
      states[idx] = 'done'
      if (idx > activeIndex) activeIndex = idx
    } else if (entry.kind === 'agent_start' || entry.status === 'running') {
      states[idx] = 'active'
      activeIndex = idx
    }
  }

  if (running) {
    const nextIdle = states.findIndex((s) => s === 'idle')
    if (nextIdle >= 0 && activeIndex >= 0 && states[activeIndex] === 'done') {
      // keep done stages done; next may become active via events
    } else if (nextIdle === 0 && entries.length === 0) {
      states[0] = 'active'
    }
  }

  return states
}

interface PipelineStepperProps {
  entries: TimelineEntry[]
  running: boolean
}

export function PipelineStepper({ entries, running }: PipelineStepperProps) {
  const states = deriveStageStates(entries, running)

  return (
    <div className="mb-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
      {PIPELINE_STAGES.map((stage, i) => {
        const state = states[i]
        const isActive = state === 'active' && running
        const isDone = state === 'done'

        return (
          <div
            key={stage.key}
            className={`relative overflow-hidden rounded-xl border px-3 py-3 text-center transition-all duration-500 ${
              isActive
                ? 'shimmer-surface scale-[1.02] border-brand-400 bg-brand-50 shadow-sm'
                : isDone
                  ? 'border-emerald-200 bg-emerald-50/80'
                  : 'border-slate-100 bg-slate-50/80'
            }`}
          >
            <div
              className={`text-lg transition-transform duration-300 ${
                isActive ? 'scale-110 text-brand-600' : isDone ? 'text-emerald-600' : 'text-slate-400'
              }`}
            >
              {stage.icon}
            </div>
            <p
              className={`mt-1 text-xs font-medium transition-colors ${
                isActive ? 'text-brand-800' : isDone ? 'text-emerald-800' : 'text-slate-500'
              }`}
            >
              {stage.label}
            </p>
            {isActive && (
              <span className="mt-1.5 inline-block h-1 w-8 overflow-hidden rounded-full bg-brand-200">
                <span className="progress-indeterminate block h-full w-full bg-brand-500" />
              </span>
            )}
            {isDone && !isActive && (
              <span className="mt-1 block text-[10px] font-medium text-emerald-600">Done</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
