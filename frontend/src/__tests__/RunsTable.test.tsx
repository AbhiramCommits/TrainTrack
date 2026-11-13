import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { StatusChip } from '../components/StatusChip'
import { RunsTable } from '../components/RunsTable'
import type { Run } from '../types'

function run(overrides: Partial<Run> = {}): Run {
  return {
    id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    experiment_id: 'exp-1',
    status: 'SUCCEEDED',
    job_kind: 'train',
    hyperparams: {
      lr: 0.001,
      batch_size: 128,
      epochs: 2,
      optimizer: 'adam',
      dropout: 0,
      seed: 42,
      weight_decay: 0,
      momentum: 0,
      label_smoothing: 0,
    },
    git_sha: 'abc',
    dataset_snapshot_id: 'snap',
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
    created_at: '2026-11-12T10:00:00Z',
    scheduled_at: null,
    started_at: null,
    finished_at: null,
    duration_seconds: 42,
    ...overrides,
  }
}

describe('StatusChip', () => {
  it('maps every status to an accessible label and tone class', () => {
    render(<StatusChip status="FAILED" />)
    const chip = screen.getByLabelText('Status Failed')
    expect(chip).toHaveClass('status-failed')
  })
})

describe('RunsTable', () => {
  it('renders runs with queue position and hyperparameters', () => {
    render(
      <RunsTable
        runs={[
          run({ status: 'QUEUED', queue_position: 1 }),
          run({ id: 'second-run-id-0000', status: 'RUNNING' }),
        ]}
        onOpen={() => {}}
      />,
    )
    expect(screen.getAllByRole('row')).toHaveLength(3)
    expect(screen.getByText('Queued')).toBeInTheDocument()
    expect(screen.getByText('Running')).toBeInTheDocument()
    expect(screen.getAllByText('0.001 / 128')).toHaveLength(2)
    expect(screen.getAllByRole('row')[1]?.textContent).toContain('Queued')
  })

  it('opens a run with the keyboard and shows the empty state', () => {
    const opened: string[] = []
    const { rerender } = render(
      <RunsTable runs={[run()]} onOpen={(id) => opened.push(id)} />,
    )
    const row = screen.getAllByRole('row')[1]
    expect(row).toBeDefined()
    row?.focus()
    row?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
    expect(opened).toEqual(['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'])

    rerender(<RunsTable runs={[]} onOpen={() => {}} />)
    expect(screen.getByText('No runs match these filters')).toBeInTheDocument()
  })
})
