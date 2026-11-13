import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SubmitRunForm } from '../components/SubmitRunForm'
import type { Experiment, Snapshot, SubmitRunPayload } from '../types'

const experiments: Experiment[] = [
  { id: 'exp-1', name: 'mnist-sweep', description: null, created_at: '2026-11-12T00:00:00Z' },
]
const snapshots: Snapshot[] = [
  {
    id: 'snap-1',
    version: 'mnist-v1',
    content_hash: 'a'.repeat(64),
    row_count: 70000,
    manifest: {},
    validated: true,
    path: '/data/snapshots/mnist-v1',
    created_at: '2026-11-12T00:00:00Z',
  },
]

describe('SubmitRunForm', () => {
  it('rejects invalid hyperparameters before submitting', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(
      <SubmitRunForm experiments={experiments} snapshots={snapshots} onSubmit={onSubmit} />,
    )
    const lr = screen.getByLabelText('Learning rate')
    await user.clear(lr)
    await user.type(lr, '-1')
    await user.click(screen.getByRole('button', { name: 'Submit run' }))
    expect(await screen.findByText(/Learning rate must be greater/)).toBeInTheDocument()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('submits a valid payload with the idempotency key and GPU count', async () => {
    const user = userEvent.setup()
    const capture: { payload?: SubmitRunPayload; key?: string } = {}
    render(
      <SubmitRunForm
        experiments={experiments}
        snapshots={snapshots}
        onSubmit={(payload, idempotencyKey) => {
          capture.payload = payload
          capture.key = idempotencyKey
        }}
      />,
    )
    const gpus = screen.getByLabelText(/Requested GPUs/)
    await user.clear(gpus)
    await user.type(gpus, '2')
    await user.click(screen.getByRole('button', { name: 'Submit run' }))
    await waitFor(() => expect(capture.payload).toBeDefined())
    expect(capture.payload?.dataset_snapshot_id).toBe('snap-1')
    expect(capture.payload?.requested_gpus).toBe(2)
    expect(capture.payload?.hyperparams.optimizer).toBe('adam')
    expect(capture.key?.length).toBeGreaterThan(0)
  })

  it('requires an experiment and snapshot when none exist', async () => {
    const user = userEvent.setup()
    render(<SubmitRunForm experiments={[]} snapshots={[]} onSubmit={() => {}} />)
    await user.click(screen.getByRole('button', { name: 'Submit run' }))
    expect(await screen.findByText('Choose an experiment.')).toBeInTheDocument()
    expect(screen.getByText('Choose a validated dataset snapshot.')).toBeInTheDocument()
  })
})
