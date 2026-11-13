import { useGetQueueHealthQuery } from '../api/api'
import { ErrorState, LoadingState } from '../components/States'
import type { QueueHealth } from '../types'

export function QueueHealthSummary({ data }: { data: QueueHealth }) {
  return (
    <section className="stat-grid" aria-label="Queue summary">
      <div className="stat-card">
        <span className="stat-label">Queued runs</span>
        <strong>{data.depth}</strong>
      </div>
      <div className="stat-card">
        <span className="stat-label">Delayed retries</span>
        <strong>{data.retry_depth}</strong>
      </div>
      <div className={`stat-card ${data.dlq_messages > 0 ? 'stat-danger' : ''}`}>
        <span className="stat-label">Dead letters</span>
        <strong>{data.dlq_messages}</strong>
        <span className="muted small">runs.dead</span>
      </div>
      <div className="stat-card">
        <span className="stat-label">Status</span>
        <strong className="capitalize">{data.status}</strong>
      </div>
    </section>
  )
}

export function QueueHealthPage() {
  const { data, isLoading, error, refetch, isFetching } = useGetQueueHealthQuery(undefined, {
    pollingInterval: 5000,
  })

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Queue health</h1>
          <p className="muted">
            RabbitMQ work, delayed-retry, and dead-letter depths. Polling every 5 seconds.
          </p>
        </div>
        {isFetching ? <span className="muted small">refreshing…</span> : null}
      </header>

      {isLoading ? <LoadingState label="Reading queue depths…" /> : null}
      {error ? (
        <ErrorState message="RabbitMQ management is unreachable." onRetry={() => void refetch()} />
      ) : null}

      {data ? (
        <>
          <QueueHealthSummary data={data} />
          <section className="panel">
            <h2>Queues</h2>
            <table className="diff-table">
              <thead>
                <tr>
                  <th scope="col">Queue</th>
                  <th scope="col">Messages</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.queues).map(([queue, count]) => (
                  <tr key={queue}>
                    <td className="mono">{queue}</td>
                    <td className="mono">{count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              Dead letters are runs that exhausted their retry budget. Investigate the run's
              failure reason and logs before re-submitting.
            </p>
          </section>
        </>
      ) : null}
    </div>
  )
}
