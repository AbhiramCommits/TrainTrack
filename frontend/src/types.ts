export type RunStatus =
  | 'QUEUED'
  | 'SCHEDULED'
  | 'RUNNING'
  | 'SUCCEEDED'
  | 'FAILED'
  | 'CANCELED'

export type JobKind = 'train' | 'eval' | 'noop'

export interface Hyperparams {
  lr: number
  batch_size: number
  epochs: number
  optimizer: 'sgd' | 'adam' | 'adamw'
  dropout: number
  seed: number
  weight_decay: number
  momentum: number
  label_smoothing: number
  max_batches?: number | null
  checkpoint_sha?: string
  [key: string]: unknown
}

export interface StatusTransition {
  id: number
  from_status: string | null
  to_status: RunStatus
  reason: string | null
  actor: string | null
  created_at: string
}

export interface Run {
  id: string
  experiment_id: string
  status: RunStatus
  job_kind: JobKind
  hyperparams: Hyperparams
  git_sha: string | null
  dataset_snapshot_id: string | null
  queue_position: number | null
  requested_gpus: number
  image: string | null
  command: string[] | null
  timeout_seconds: number
  priority: number
  attempt_count: number
  max_attempts: number
  idempotency_key: string | null
  failure_reason: string | null
  executor: string | null
  worker_id: string | null
  pod_name: string | null
  scheduling_latency_seconds: number | null
  created_at: string
  scheduled_at: string | null
  started_at: string | null
  finished_at: string | null
  duration_seconds: number | null
  transitions?: StatusTransition[]
}

export interface Experiment {
  id: string
  name: string
  description: string | null
  created_at: string
  run_count?: number
}

export interface Snapshot {
  id: string
  version: string
  content_hash: string
  row_count: number
  manifest: Record<string, unknown>
  validated: boolean
  path: string
  created_at: string
}

export interface MetricPoint {
  name: string
  step: number
  value: number
  wall_time: number
}

export interface Artifact {
  id: number
  sha256: string
  kind: string
  size: number
  uri: string
  run_id: string | null
  dataset_snapshot_id: string | null
  parent_sha: string | null
  ref_count: number
  bytes_saved: number
  created_at: string
}

export interface LineageNode {
  artifact: Artifact
  run: Run | null
  dataset_snapshot: Snapshot | null
}

export interface Lineage {
  sha256: string
  chain: LineageNode[]
  artifact: Artifact
  run: Run | null
  dataset_snapshot: Snapshot | null
}

export interface LeaderboardEntry {
  run_id: string
  experiment_id: string
  status: RunStatus
  metric_name: string
  best_value: number
  hyperparams: Hyperparams
  dataset_snapshot_id: string | null
  created_at: string
  finished_at: string | null
  rank: number
}

export interface QueueHealth {
  status: string
  depth: number
  retry_depth: number
  dlq_messages: number
  queues: Record<string, number>
}

export interface ResourceSample {
  id?: string
  step: number
  device: string
  utilization_pct: number
  memory_used_mb: number
  cpu_pct: number
  rss_mb: number
  wall_time: number
}

export interface RunListResponse {
  runs: Run[]
  next_cursor: string | null
  count: number
}

export interface SubmitRunPayload {
  experiment_id: string
  job_kind: JobKind
  hyperparams: Hyperparams
  dataset_snapshot_id?: string | null
  requested_gpus: number
  priority: number
  timeout_seconds: number
}

export interface MetricResponse {
  run_id: string
  points: MetricPoint[]
  count: number
}

export interface ResourceResponse {
  run_id: string
  samples: ResourceSample[]
  count: number
  device: string | null
}
