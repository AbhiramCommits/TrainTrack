import { useEffect, useRef, useState } from 'react'
import { streamUrl } from '../api/api'

export interface LogLine {
  ts: number
  level: string
  line: string
}

export interface StreamSource {
  onmessage: ((event: MessageEvent<string>) => void) | null
  onerror: ((event: Event) => void) | null
  close: () => void
}

export type StreamFactory = (url: string) => StreamSource

const defaultFactory: StreamFactory = (url) => new EventSource(url)

export function LiveLogViewer({
  runId,
  sourceFactory = defaultFactory,
  maxLines = 1000,
}: {
  runId: string
  sourceFactory?: StreamFactory
  maxLines?: number
}) {
  const [lines, setLines] = useState<LogLine[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const [connection, setConnection] = useState<'connecting' | 'open' | 'ended'>('connecting')
  const viewport = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    setLines([])
    setConnection('connecting')
    const source = sourceFactory(streamUrl(runId))
    source.onmessage = (event: MessageEvent<string>) => {
      try {
        const payload = JSON.parse(event.data) as Partial<LogLine>
        const line: LogLine = {
          ts: typeof payload.ts === 'number' ? payload.ts : Date.now() / 1000,
          level: typeof payload.level === 'string' ? payload.level : 'info',
          line: typeof payload.line === 'string' ? payload.line : event.data,
        }
        setConnection('open')
        setLines((current) => [...current.slice(Math.max(0, current.length - maxLines + 1)), line])
      } catch {
        setLines((current) => [
          ...current.slice(Math.max(0, current.length - maxLines + 1)),
          { ts: Date.now() / 1000, level: 'info', line: event.data },
        ])
      }
    }
    source.onerror = () => setConnection('ended')
    return () => source.close()
  }, [runId, sourceFactory, maxLines])

  useEffect(() => {
    if (autoScroll && viewport.current) {
      viewport.current.scrollTop = viewport.current.scrollHeight
    }
  }, [lines, autoScroll])

  return (
    <section className="logs-panel" aria-label="Run logs">
      <div className="panel-heading">
        <div>
          <h2>Live logs</h2>
          <span className={`live-dot live-${connection}`} aria-hidden="true" />
          <span className="muted small">
            {connection === 'open' ? 'streaming' : connection === 'ended' ? 'stream closed' : 'connecting'}
          </span>
        </div>
        <div className="toolbar">
          <button
            type="button"
            className="button button-secondary"
            aria-pressed={!autoScroll}
            onClick={() => setAutoScroll((value) => !value)}
          >
            {autoScroll ? 'Pause autoscroll' : 'Resume autoscroll'}
          </button>
          <button type="button" className="button button-secondary" onClick={() => setLines([])}>
            Clear
          </button>
        </div>
      </div>
      <div className="log-viewport" ref={viewport} tabIndex={0} aria-live="polite">
        {lines.length === 0 ? (
          <span className="muted">Waiting for log output…</span>
        ) : (
          lines.map((line, index) => (
            <div key={`${line.ts}-${index}`} className={`log-line log-${line.level}`}>
              <span className="log-time">
                {new Date(line.ts * 1000).toLocaleTimeString(undefined, {
                  hour12: false,
                })}
              </span>
              <span>{line.line}</span>
            </div>
          ))
        )}
      </div>
    </section>
  )
}
