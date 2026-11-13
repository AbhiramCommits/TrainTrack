import { describe, expect, it } from 'vitest'
import type { Run, RunStatus } from '../types'
import {
  buildSubmitHyperparams,
  filterRuns,
  formatBytes,
  formatDuration,
  formatPercent,
  hyperparamDiff,
  isValidHyperparams,
  mergeMetricSeries,
  shortId,
} from '../utils/format'

function makeRun(overrides: Partial<Run> = {}): Run {
  return {
    id: '11111111-1111-4111-8111-111111111111',
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
    idempotency_key: 'key',
    failure_reason: null,
    executor: 'local',
    worker_id: 'w1',
    pod_name: null,
    scheduling_latency_seconds: null,
    created_at: '2026-11-12T00:00:00Z',
    scheduled_at: null,
    started_at: '2026-11-12T00:00:05Z',
    finished_at: '2026-11-12T00:00:45Z',
    duration_seconds: 40,
    ...overrides,
  }
}

describe('format helpers', () => {
  it('formats durations across scales', () => {
    expect(formatDuration(null)).toBe('—')
    expect(formatDuration(0.25)).toBe('250 ms')
    expect(formatDuration(40)).toBe('40.0 s')
    expect(formatDuration(125)).toBe('2m 5s')
    expect(formatDuration(3725)).toBe('1h 2m')
  })

  it('formats percentages, bytes, and ids', () => {
    expect(formatPercent(0.9851)).toBe('98.51%')
    expect(formatPercent(null)).toBe('—')
    expect(formatBytes(2048)).toBe('2.0 KiB')
    expect(shortId('abcdefghijkl', 6)).toBe('abcdef')
  })

  it('filters runs by status, experiment, kind and search', () => {
    const runs = [
      makeRun({ id: 'a', status: 'SUCCEEDED' }),
      makeRun({ id: 'b', status: 'RUNNING', job_kind: 'eval' }),
      makeRun({ id: 'c', status: 'FAILED', experiment_id: 'exp-2', hyperparams: { ...makeRun().hyperparams, lr: 0.5 } }),
    ]
    expect(filterRuns(runs, { status: 'ALL', experimentId: 'ALL', jobKind: 'ALL', search: '' })).toHaveLength(3)
    expect(
      filterRuns(runs, { status: 'RUNNING', experimentId: 'ALL', jobKind: 'ALL', search: '' }),
    ).toEqual([runs[1]])
    expect(
      filterRuns(runs, { status: 'ALL', experimentId: 'exp-2', jobKind: 'ALL', search: '' }),
    ).toEqual([runs[2]])
    expect(
      filterRuns(runs, { status: 'ALL', experimentId: 'ALL', jobKind: 'ALL', search: '0.001' }),
    ).toHaveLength(2)
  })

  it('merges metric series by step and rounds to six decimals', () => {
    const merged = mergeMetricSeries([
      {
        label: 'val_accuracy',
        points: [
          { name: 'val_accuracy', step: 1, value: 0.5, wall_time: 0 },
          { name: 'val_accuracy', step: 2, value: 0.75, wall_time: 0 },
        ],
      },
      {
        label: 'train_loss',
        points: [{ name: 'train_loss', step: 1, value: 0.1234567, wall_time: 0 }],
      },
    ])
    expect(merged).toEqual([
      { step: 1, val_accuracy: 0.5, train_loss: 0.123457 },
      { step: 2, val_accuracy: 0.75 },
    ])
  })

  it('validates hyperparameters with useful errors', () => {
    const valid = { lr: '0.001', batchSize: '128', epochs: '2', dropout: '0.1', seed: '42' }
    expect(isValidHyperparams(valid)).toEqual({})
    const invalid = isValidHyperparams({
      lr: '-1',
      batchSize: '0',
      epochs: '0',
      dropout: '2',
      seed: '-4',
    })
    expect(Object.keys(invalid).sort()).toEqual(['batchSize', 'dropout', 'epochs', 'lr', 'seed'])
  })

  it('builds the API hyperparameter payload', () => {
    expect(
      buildSubmitHyperparams({
        lr: '0.002',
        batchSize: '64',
        epochs: '3',
        dropout: '0',
        seed: '7',
        optimizer: 'sgd',
        weightDecay: '0.01',
        momentum: '0.9',
      }),
    ).toMatchObject({ lr: 0.002, batch_size: 64, epochs: 3, optimizer: 'sgd', seed: 7 })
  })

  it('diffs hyperparameters across runs', () => {
    const rows = hyperparamDiff([
      makeRun({ hyperparams: { ...makeRun().hyperparams, lr: 0.1 } }),
      makeRun({ id: 'b', hyperparams: { ...makeRun().hyperparams, lr: 0.2 } }),
    ])
    const lrRow = rows.find((row) => row.key === 'lr')
    expect(lrRow?.values).toEqual(['0.1', '0.2'])
  })

  it('accepts every status value in the filter', () => {
    const statuses: RunStatus[] = ['QUEUED', 'SCHEDULED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELED']
    expect(statuses).toHaveLength(6)
  })
})
