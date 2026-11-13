import type { Lineage } from '../types'
import { formatBytes, shortId } from '../utils/format'
import { StatusChip } from './StatusChip'

export function LineagePanel({ lineage, loading }: { lineage?: Lineage; loading: boolean }) {
  if (loading) return <p className="muted">Resolving artifact lineage…</p>
  if (!lineage) return null
  return (
    <div className="lineage-panel">
      <h3>Lineage chain</h3>
      <ol className="lineage-chain">
        {lineage.chain.map((node) => (
          <li key={node.artifact.sha256} className="lineage-node">
            <div className="lineage-kind">{node.artifact.kind}</div>
            <div className="mono small">{node.artifact.sha256}</div>
            <div className="muted small">
              {formatBytes(node.artifact.size)}
              {node.artifact.parent_sha ? ` · parent ${shortId(node.artifact.parent_sha, 12)}` : ' · root artifact'}
            </div>
            {node.run ? (
              <div className="lineage-meta">
                <span>run {shortId(node.run.id, 12)}</span>
                <StatusChip status={node.run.status} />
                <span className="mono small">
                  lr {String(node.run.hyperparams.lr ?? '—')} · batch{' '}
                  {String(node.run.hyperparams.batch_size ?? '—')} · git {node.run.git_sha ?? '—'}
                </span>
              </div>
            ) : null}
            {node.dataset_snapshot ? (
              <div className="muted small">
                snapshot {node.dataset_snapshot.version} ·{' '}
                {node.dataset_snapshot.content_hash.slice(0, 16)} ·{' '}
                {node.dataset_snapshot.row_count.toLocaleString()} rows
              </div>
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  )
}
