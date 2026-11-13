import { StatusChip } from './StatusChip'
import type { Experiment, Run } from '../types'
import { formatDate, formatDuration, shortId } from '../utils/format'

interface RunsTableProps {
  runs: Run[]
  experiments?: Experiment[]
  onOpen: (runId: string) => void
  emptyDetail?: string
}

export function RunsTable({
  runs,
  experiments = [],
  onOpen,
  emptyDetail = 'Submit a run to populate the registry.',
}: RunsTableProps) {
  const names = new Map(experiments.map((experiment) => [experiment.id, experiment.name]))

  if (runs.length === 0) {
    return (
      <div className="table-empty">
        <strong>No runs match these filters</strong>
        <span>{emptyDetail}</span>
      </div>
    )
  }

  return (
    <div className="table-scroll">
      <table className="runs-table">
        <caption className="sr-only">Training and evaluation runs</caption>
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Status</th>
            <th scope="col">Kind</th>
            <th scope="col">LR / batch</th>
            <th scope="col">Queue</th>
            <th scope="col">Duration</th>
            <th scope="col">Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.id}
              tabIndex={0}
              aria-label={`Open run ${run.id}`}
              onClick={() => onOpen(run.id)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onOpen(run.id)
                }
              }}
            >
              <td>
                <span className="mono run-id">{shortId(run.id, 10)}</span>
                <span className="muted small">
                  {names.get(run.experiment_id) ?? shortId(run.experiment_id)}
                </span>
              </td>
              <td>
                <StatusChip status={run.status} />
              </td>
              <td>{run.job_kind}</td>
              <td className="mono">
                {run.job_kind === 'train'
                  ? `${run.hyperparams.lr ?? '—'} / ${run.hyperparams.batch_size ?? '—'}`
                  : shortId(run.hyperparams.checkpoint_sha ?? '—', 10)}
              </td>
              <td>{run.queue_position ?? '—'}</td>
              <td>{formatDuration(run.duration_seconds)}</td>
              <td>{formatDate(run.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
