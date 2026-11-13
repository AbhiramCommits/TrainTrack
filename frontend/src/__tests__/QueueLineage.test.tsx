import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueueHealthSummary } from '../pages/QueueHealthPage'
import { LineagePanel } from '../components/LineagePanel'
import type { Lineage, Run, Snapshot } from '../types'

const run: Run = {
  id: 'run-1234567890',
  experiment_id: 'exp-1',
  status: 'SUCCEEDED',
  job_kind: 'train',
  hyperparams: {
    lr: 0.001,
    batch_size: 64,
    epochs: 2,
    optimizer: 'adam',
    dropout: 0,
    seed: 42,
    weight_decay: 0,
    momentum: 0,
    label_smoothing: 0,
  },
  git_sha: 'abc123',
  dataset_snapshot_id: 'snap-1',
  queue_position: null,
  requested_gpus: 0,
  image: null,
  command: null,
  timeout_seconds: 3600,
  priority: 5,
  attempt_count: 1,
  max_attempts: 3,
  idempotency_key: null,
  failure_reason: null,
  executor: 'local',
  worker_id: null,
  pod_name: null,
  scheduling_latency_seconds: null,
  created_at: '2026-11-12T00:00:00Z',
  scheduled_at: null,
  started_at: '2026-11-12T00:00:01Z',
  finished_at: '2026-11-12T00:01:00Z',
  duration_seconds: 59,
}

const snapshot: Snapshot = {
  id: 'snap-1',
  version: 'mnist-v1',
  content_hash: 'a'.repeat(64),
  row_count: 70000,
  manifest: {},
  validated: true,
  path: '/data/snapshots/mnist-v1',
  created_at: '2026-11-12T00:00:00Z',
}

const lineage: Lineage = {
  sha256: 'b'.repeat(64),
  chain: [
    {
      artifact: {
        id: 1,
        sha256: 'b'.repeat(64),
        kind: 'checkpoint',
        size: 831000,
        uri: '/artifacts/bb/bbbb',
        run_id: run.id,
        dataset_snapshot_id: snapshot.id,
        parent_sha: null,
        ref_count: 1,
        bytes_saved: 0,
        created_at: '2026-11-12T00:01:00Z',
      },
      run,
      dataset_snapshot: snapshot,
    },
  ],
  artifact: {
    id: 1,
    sha256: 'b'.repeat(64),
    kind: 'checkpoint',
    size: 831000,
    uri: '/artifacts/bb/bbbb',
    run_id: run.id,
    dataset_snapshot_id: snapshot.id,
    parent_sha: null,
    ref_count: 1,
    bytes_saved: 0,
    created_at: '2026-11-12T00:01:00Z',
  },
  run,
  dataset_snapshot: snapshot,
}

describe('QueueHealthSummary', () => {
  it('shows depth, retries, and dead-letter count', () => {
    render(
      <QueueHealthSummary
        data={{
          status: 'ok',
          depth: 4,
          retry_depth: 2,
          dlq_messages: 1,
          queues: { 'runs.submitted': 4, 'runs.retry': 2, 'runs.dead': 1 },
        }}
      />,
    )
    expect(screen.getByText('Queued runs')).toBeInTheDocument()
    expect(screen.getByText('4')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('1')).toBeInTheDocument()
    expect(screen.getByText('runs.dead')).toBeInTheDocument()
  })
})

describe('LineagePanel', () => {
  it('renders the artifact to run to snapshot chain', () => {
    render(<LineagePanel lineage={lineage} loading={false} />)
    expect(screen.getByText('Lineage chain')).toBeInTheDocument()
    expect(screen.getByText('checkpoint')).toBeInTheDocument()
    expect(screen.getByText(/run run-12345678/)).toBeInTheDocument()
    expect(screen.getByText(/snapshot mnist-v1/)).toBeInTheDocument()
  })
})
