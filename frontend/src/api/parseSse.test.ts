import { describe, expect, it } from 'vitest'
import { parseSseStream } from './parseSse'

describe('parseSseStream', () => {
  it('parses CRLF-delimited SSE blocks', () => {
    const raw =
      'event: job_started\r\n' +
      'data: {"job_id":"abc123"}\r\n\r\n' +
      'event: agent_start\r\n' +
      'data: {"type":"agent_start","ts":"2026-01-01T00:00:00Z","agent":"decomposition"}\r\n\r\n' +
      'event: final_answer\r\n' +
      'data: {"type":"final_answer","answer":"Hello world"}\r\n\r\n'

    const events: { type: string; data: unknown }[] = []
    parseSseStream(raw, (type, data) => events.push({ type, data }))

    expect(events).toHaveLength(3)
    expect(events[0].type).toBe('job_started')
    expect(events[0].data).toEqual({ job_id: 'abc123' })
    expect(events[1].type).toBe('agent_start')
    expect(events[2].type).toBe('final_answer')
    expect((events[2].data as { answer: string }).answer).toBe('Hello world')
  })

  it('ignores ping comments', () => {
    const raw = ': ping\r\n\r\nevent: done\r\ndata: {"type":"done"}\r\n\r\n'
    const events: string[] = []
    parseSseStream(raw, (type) => events.push(type))
    expect(events).toEqual(['done'])
  })
})
