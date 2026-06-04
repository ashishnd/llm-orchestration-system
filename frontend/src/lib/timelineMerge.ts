import type { TimelineEntry } from '../types'

/** Merge start/complete pairs so running cards transition to done in-place. */
export function upsertTimelineEntry(
  prev: TimelineEntry[],
  entry: TimelineEntry,
): TimelineEntry[] {
  const mergeKinds: TimelineEntry['kind'][] = [
    'agent_complete',
    'tool_complete',
  ]

  if (mergeKinds.includes(entry.kind)) {
    const startKind = entry.kind.replace('_complete', '_start') as TimelineEntry['kind']
    const idx = findLastIndex(
      prev,
      (e) => e.label === entry.label && (e.kind === startKind || e.status === 'running'),
    )
    if (idx >= 0) {
      const updated = [...prev]
      updated[idx] = {
        ...updated[idx],
        kind: entry.kind,
        status: entry.status ?? 'done',
        detail: entry.detail ?? updated[idx].detail,
        ts: entry.ts ?? updated[idx].ts,
      }
      return updated
    }
  }

  return [...prev, entry]
}

function findLastIndex<T>(arr: T[], pred: (item: T) => boolean): number {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (pred(arr[i])) return i
  }
  return -1
}
