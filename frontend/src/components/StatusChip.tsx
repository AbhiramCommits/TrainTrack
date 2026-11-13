import type { RunStatus } from '../types'

const LABELS: Record<RunStatus, string> = {
  QUEUED: 'Queued',
  SCHEDULED: 'Scheduled',
  RUNNING: 'Running',
  SUCCEEDED: 'Succeeded',
  FAILED: 'Failed',
  CANCELED: 'Canceled',
}

export function StatusChip({ status }: { status: RunStatus }) {
  return (
    <span className={`status-chip status-${status.toLowerCase()}`} aria-label={`Status ${LABELS[status]}`}>
      <span className="status-dot" aria-hidden="true" />
      {LABELS[status]}
    </span>
  )
}
