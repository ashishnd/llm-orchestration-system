import { API_BASE } from './config'
import type { SsePayload } from '../types'

export type SseHandler = (eventType: string, data: SsePayload) => void

/** Normalize CRLF/CR line endings so event blocks split reliably. */
export function normalizeNewlines(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
}

function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block.split('\n')
  let event = 'message'
  const dataLines: string[] = []

  for (const line of lines) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }

  if (dataLines.length === 0) return null
  return { event, data: dataLines.join('\n') }
}

function dispatchBlocks(blocks: string[], onEvent: SseHandler): void {
  for (const block of blocks) {
    const parsed = parseSseBlock(block)
    if (!parsed) continue
    try {
      const payload = JSON.parse(parsed.data) as SsePayload
      const type =
        parsed.event !== 'message' ? parsed.event : (payload.type ?? parsed.event)
      onEvent(type, payload)
    } catch {
      onEvent(parsed.event, { message: parsed.data })
    }
  }
}

function takeCompleteBlocks(buffer: string): { blocks: string[]; rest: string } {
  const normalized = normalizeNewlines(buffer)
  const parts = normalized.split('\n\n')
  const rest = parts.pop() ?? ''
  return { blocks: parts.filter(Boolean), rest }
}

/** Incrementally parse SSE text (for streaming or tests). */
export function parseSseStream(text: string, onEvent: SseHandler): void {
  const normalized = normalizeNewlines(text).trim()
  if (!normalized) return
  const { blocks } = takeCompleteBlocks(`${normalized}\n\n`)
  dispatchBlocks(blocks, onEvent)
}

/** POST /jobs and stream SSE events until the connection closes. */
export async function submitJobStream(
  query: string,
  onEvent: SseHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ query }),
    signal,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }))
    throw new Error(err.message ?? 'Job submission failed')
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (value) {
      buffer += decoder.decode(value, { stream: true })
      const { blocks, rest } = takeCompleteBlocks(buffer)
      buffer = rest
      dispatchBlocks(blocks, onEvent)
    }
    if (done) break
  }

  buffer += decoder.decode()
  if (buffer.trim()) {
    dispatchBlocks([normalizeNewlines(buffer).trim()], onEvent)
  }
}
