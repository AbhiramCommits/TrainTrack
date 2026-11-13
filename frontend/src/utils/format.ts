import type { Hyperparams, MetricPoint, Run, RunStatus, Snapshot } from '../types'
import type { StatusFilter } from '../store/filtersSlice'

export const STATUS_ORDER: RunStatus[] = [
  'QUEUED',
  'SCHEDULED',
  'RUNNING',
  'SUCCEEDED',
  'FAILED',
  'CANCELED',
]

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—'
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.round(seconds % 60)
  if (minutes < 60) return `${minutes}m ${remainder}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  return new Date(value).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function shortId(value: string | null | undefined, length = 8): string {
  if (!value) return '—'
  return value.slice(0, length)
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return `${(value * 100).toFixed(2)}%`
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`
}

export function filterRuns(
  runs: Run[],
  filters: { status: StatusFilter; experimentId: string; jobKind: string; search: string },
): Run[] {
  const search = filters.search.trim().toLowerCase()
  return runs.filter((run) => {
    if (filters.status !== 'ALL' && run.status !== filters.status) return false
    if (filters.experimentId !== 'ALL' && run.experiment_id !== filters.experimentId) return false
    if (filters.jobKind !== 'ALL' && run.job_kind !== filters.jobKind) return false
    if (!search) return true
    const haystack = [
      run.id,
      run.status,
      run.job_kind,
      String(run.hyperparams.lr ?? ''),
      String(run.hyperparams.batch_size ?? ''),
      run.git_sha ?? '',
    ]
      .join(' ')
      .toLowerCase()
    return haystack.includes(search)
  })
}

export interface SeriesPoint {
  step: number
  [key: string]: number
}

export function mergeMetricSeries(
  series: Array<{ label: string; points: MetricPoint[] }>,
): SeriesPoint[] {
  const byStep = new Map<number, SeriesPoint>()
  for (const item of series) {
    for (const point of item.points) {
      const existing = byStep.get(point.step) ?? { step: point.step }
      const bucket = Math.round(point.value * 1_000_000) / 1_000_000
      existing[item.label] = bucket
      byStep.set(point.step, existing)
    }
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step)
}

export function bestMetric(points: MetricPoint[], name: string): number | null {
  let best: number | null = null
  for (const point of points) {
    if (point.name !== name) continue
    if (best === null || point.value > best) best = point.value
  }
  return best
}

export function hyperparamDiff(runs: Run[]): Array<{ key: string; values: string[] }> {
  if (runs.length === 0) return []
  const keys = new Set<string>()
  runs.forEach((run) => Object.keys(run.hyperparams).forEach((key) => keys.add(key)))
  return [...keys]
    .sort()
    .map((key) => ({
      key,
      values: runs.map((run) => {
        const value = run.hyperparams[key]
        return value === null || value === undefined ? '—' : String(value)
      }),
    }))
}

export function snapshotLabel(snapshot: Snapshot): string {
  return `${snapshot.version} · ${snapshot.row_count.toLocaleString()} rows · ${snapshot.content_hash.slice(0, 10)}`
}

export function isValidHyperparams(values: {
  lr: string
  batchSize: string
  epochs: string
  dropout: string
  seed: string
}): Record<string, string> {
  const errors: Record<string, string> = {}
  const lr = Number(values.lr)
  const batchSize = Number(values.batchSize)
  const epochs = Number(values.epochs)
  const dropout = Number(values.dropout)
  const seed = Number(values.seed)
  if (!Number.isFinite(lr) || lr <= 0 || lr > 1) {
    errors.lr = 'Learning rate must be greater than 0 and at most 1.'
  }
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 4096) {
    errors.batchSize = 'Batch size must be an integer between 1 and 4096.'
  }
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
    errors.epochs = 'Epochs must be an integer between 1 and 50.'
  }
  if (!Number.isFinite(dropout) || dropout < 0 || dropout > 0.9) {
    errors.dropout = 'Dropout must be between 0 and 0.9.'
  }
  if (!Number.isInteger(seed) || seed < 0) {
    errors.seed = 'Seed must be a non-negative integer.'
  }
  return errors
}

export function buildSubmitHyperparams(values: {
  lr: string
  batchSize: string
  epochs: string
  dropout: string
  seed: string
  optimizer: 'sgd' | 'adam' | 'adamw'
  weightDecay: string
  momentum: string
}): Hyperparams {
  return {
    lr: Number(values.lr),
    batch_size: Number(values.batchSize),
    epochs: Number(values.epochs),
    optimizer: values.optimizer,
    dropout: Number(values.dropout),
    seed: Number(values.seed),
    weight_decay: Number(values.weightDecay),
    momentum: Number(values.momentum),
    label_smoothing: 0,
  }
}
