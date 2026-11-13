import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  useCancelRunMutation,
  useGetArtifactsQuery,
  useGetLineageQuery,
  useGetMetricsQuery,
  useGetResourcesQuery,
  useGetRunQuery,
} from '../api/api'
import { GpuUtilPanel } from '../components/GpuUtilPanel'
import { LineagePanel } from '../components/LineagePanel'
import { LiveLogViewer } from '../components/LiveLogViewer'
import { MetricCurves } from '../components/MetricCurves'
import { ErrorState, LoadingState } from '../components/States'
import { StatusChip } from '../components/StatusChip'
import type { RunStatus } from '../types'
import { formatBytes, formatDate, formatDuration, formatPercent, shortId } from '../utils/format'
import { bestMetric } from '../utils/format'

const ACTIVE: RunStatus[] = ['QUEUED', 'SCHEDULED', 'RUNNING']

export function RunDetailPage() {
  const { runId = '' } = useParams()
  const [lineageSha, setLineageSha] = useState<string | null>(null)
  const runQuery = useGetRunQuery(runId, { pollingInterval: 3000 })
  const metricsQuery = useGetMetricsQuery({ runId, limit: 10000 }, { pollingInterval: 5000 })
  const resourcesQuery = useGetResourcesQuery({ runId }, { pollingInterval: 5000 })
  const artifactsQuery = useGetArtifactsQuery(runId)
  const lineageQuery = useGetLineageQuery(lineageSha ?? '', { skip: !lineageSha })
  const [cancelRun, cancelState] = useCancelRunMutation()

  if (runQuery.isLoading) return <LoadingState label="Loading run…" />
  if (runQuery.error || !runQuery.data) {
    return <ErrorState message="Run not found or the API is unavailable." />
  }
  const run = runQuery.data
  const points = metricsQuery.data?.points ?? []
  const valAccuracy = bestMetric(points, 'val_accuracy')
  const testAccuracy = bestMetric(points, 'test_accuracy')

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="small muted">
            <Link to="/">← Back to runs</Link>
          </p>
          <h1 className="mono">{shortId(run.id, 16)}</h1>
          <div className="header-meta">
            <StatusChip status={run.status} />
            <span className="muted small">
              {run.job_kind} · {run.executor ?? 'queued'} · attempt {run.attempt_count}/
              {run.max_attempts}
            </span>
            {run.failure_reason ? <span className="failure-text">{run.failure_reason}</span> : null}
          </div>
        </div>
        {ACTIVE.includes(run.status) ? (
          <button
            type="button"
            className="button button-danger"
            disabled={cancelState.isLoading}
            onClick={() => void cancelRun(run.id)}
          >
            {cancelState.isLoading ? 'Canceling…' : 'Cancel run'}
          </button>
        ) : null}
      </header>

      <section className="stat-grid" aria-label="Run summary">
        <div className="stat-card">
          <span className="stat-label">Validation accuracy</span>
          <strong>{formatPercent(valAccuracy)}</strong>
        </div>
        <div className="stat-card">
          <span className="stat-label">Test accuracy</span>
          <strong>{formatPercent(testAccuracy)}</strong>
        </div>
        <div className="stat-card">
          <span className="stat-label">Wall clock</span>
          <strong>{formatDuration(run.duration_seconds)}</strong>
        </div>
        <div className="stat-card">
          <span className="stat-label">Requested GPUs</span>
          <strong>{run.requested_gpus}</strong>
          <span className="muted small">{run.pod_name ? `pod ${run.pod_name}` : 'no pod'}</span>
        </div>
      </section>

      <div className="content-grid">
        <section className="panel panel-wide">
          <h2>Metric curves</h2>
          {metricsQuery.isLoading ? <LoadingState /> : null}
          <MetricCurves points={points} />
        </section>

        <section className="panel">
          <h2>Status timeline</h2>
          <ol className="timeline">
            {(run.transitions ?? []).map((transition) => (
              <li key={transition.id}>
                <StatusChip status={transition.to_status} />
                <span className="muted small">{formatDate(transition.created_at)}</span>
                <span className="small">{transition.reason ?? '—'}</span>
                <span className="muted small">by {transition.actor ?? 'system'}</span>
              </li>
            ))}
          </ol>
        </section>
      </div>

      <div className="content-grid">
        <section className="panel">
          <h2>Hyperparameters</h2>
          <dl className="kv-grid">
            {Object.entries(run.hyperparams).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd className="mono">{String(value ?? '—')}</dd>
              </div>
            ))}
            <div>
              <dt>git_sha</dt>
              <dd className="mono">{run.git_sha ?? '—'}</dd>
            </div>
            <div>
              <dt>snapshot</dt>
              <dd className="mono">{shortId(run.dataset_snapshot_id, 12)}</dd>
            </div>
          </dl>
        </section>

        <section className="panel">
          <h2>Device utilization</h2>
          <GpuUtilPanel
            samples={resourcesQuery.data?.samples ?? []}
            device={resourcesQuery.data?.device ?? null}
          />
        </section>
      </div>

      <section className="panel">
        <h2>Artifacts</h2>
        {artifactsQuery.isLoading ? <LoadingState /> : null}
        {artifactsQuery.data && artifactsQuery.data.artifacts.length === 0 ? (
          <p className="muted">No artifacts yet. Checkpoints and reports appear after training.</p>
        ) : null}
        <div className="artifact-list">
          {artifactsQuery.data?.artifacts.map((artifact) => (
            <div key={artifact.sha256} className="artifact-row">
              <span className="badge">{artifact.kind}</span>
              <span className="mono small">{shortId(artifact.sha256, 16)}</span>
              <span className="muted small">{formatBytes(artifact.size)}</span>
              <button
                type="button"
                className="button button-secondary"
                onClick={() => setLineageSha(artifact.sha256)}
              >
                Lineage
              </button>
            </div>
          ))}
        </div>
        <LineagePanel lineage={lineageQuery.data} loading={lineageQuery.isFetching} />
      </section>

      <LiveLogViewer runId={run.id} />
    </div>
  )
}
