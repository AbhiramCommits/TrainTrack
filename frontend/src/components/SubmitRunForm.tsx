import { useState, type FormEvent } from 'react'
import type { Experiment, Snapshot, SubmitRunPayload } from '../types'
import { buildSubmitHyperparams, isValidHyperparams, snapshotLabel } from '../utils/format'
import { newIdempotencyKey } from '../utils/idempotency'

interface SubmitRunFormProps {
  experiments: Experiment[]
  snapshots: Snapshot[]
  onSubmit: (payload: SubmitRunPayload, idempotencyKey: string) => Promise<void> | void
  submitting?: boolean
  submitError?: string | null
}

const initialValues = {
  lr: '0.001',
  batchSize: '128',
  epochs: '2',
  dropout: '0.0',
  seed: '42',
  optimizer: 'adam' as const,
  weightDecay: '0.0',
  momentum: '0.9',
}

export function SubmitRunForm({
  experiments,
  snapshots,
  onSubmit,
  submitting = false,
  submitError = null,
}: SubmitRunFormProps) {
  const [values, setValues] = useState(initialValues)
  const [experimentId, setExperimentId] = useState(experiments[0]?.id ?? '')
  const [snapshotId, setSnapshotId] = useState(snapshots[0]?.id ?? '')
  const [gpus, setGpus] = useState('0')
  const [idempotencyKey, setIdempotencyKey] = useState(newIdempotencyKey)
  const [errors, setErrors] = useState<Record<string, string>>({})

  const update = (field: keyof typeof initialValues, value: string) => {
    setValues((current) => ({ ...current, [field]: value }))
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const nextErrors = isValidHyperparams(values)
    if (!experimentId) nextErrors.experiment = 'Choose an experiment.'
    if (!snapshotId) nextErrors.snapshot = 'Choose a validated dataset snapshot.'
    const gpuCount = Number(gpus)
    if (!Number.isInteger(gpuCount) || gpuCount < 0 || gpuCount > 8) {
      nextErrors.gpus = 'GPU count must be an integer between 0 and 8.'
    }
    if (!idempotencyKey.trim()) nextErrors.idempotency = 'Idempotency key is required.'
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return

    await onSubmit(
      {
        experiment_id: experimentId,
        job_kind: 'train',
        hyperparams: buildSubmitHyperparams(values),
        dataset_snapshot_id: snapshotId,
        requested_gpus: gpuCount,
        priority: 5,
        timeout_seconds: 3600,
      },
      idempotencyKey.trim(),
    )
  }

  return (
    <form className="panel form-panel" onSubmit={(event) => void handleSubmit(event)} noValidate>
      <h2>Train a model</h2>
      <p className="muted">
        Hyperparameters are validated both here and by the API. The snapshot gate refuses
        unvalidated or hash-mismatched data.
      </p>

      <div className="form-grid">
        <label>
          Experiment
          <select
            value={experimentId}
            onChange={(event) => setExperimentId(event.target.value)}
            aria-invalid={Boolean(errors.experiment)}
          >
            {experiments.length === 0 ? <option value="">No experiments</option> : null}
            {experiments.map((experiment) => (
              <option key={experiment.id} value={experiment.id}>
                {experiment.name}
              </option>
            ))}
          </select>
          {errors.experiment ? <span className="field-error">{errors.experiment}</span> : null}
        </label>

        <label>
          Dataset snapshot
          <select
            value={snapshotId}
            onChange={(event) => setSnapshotId(event.target.value)}
            aria-invalid={Boolean(errors.snapshot)}
          >
            {snapshots.length === 0 ? <option value="">No validated snapshots</option> : null}
            {snapshots.map((snapshot) => (
              <option key={snapshot.id} value={snapshot.id}>
                {snapshotLabel(snapshot)}
              </option>
            ))}
          </select>
          {errors.snapshot ? <span className="field-error">{errors.snapshot}</span> : null}
        </label>

        <label>
          Learning rate
          <input
            inputMode="decimal"
            value={values.lr}
            onChange={(event) => update('lr', event.target.value)}
            aria-invalid={Boolean(errors.lr)}
          />
          {errors.lr ? <span className="field-error">{errors.lr}</span> : null}
        </label>

        <label>
          Batch size
          <input
            inputMode="numeric"
            value={values.batchSize}
            onChange={(event) => update('batchSize', event.target.value)}
            aria-invalid={Boolean(errors.batchSize)}
          />
          {errors.batchSize ? <span className="field-error">{errors.batchSize}</span> : null}
        </label>

        <label>
          Epochs
          <input
            inputMode="numeric"
            value={values.epochs}
            onChange={(event) => update('epochs', event.target.value)}
            aria-invalid={Boolean(errors.epochs)}
          />
          {errors.epochs ? <span className="field-error">{errors.epochs}</span> : null}
        </label>

        <label>
          Optimizer
          <select
            value={values.optimizer}
            onChange={(event) =>
              update('optimizer', event.target.value as 'sgd' | 'adam' | 'adamw')
            }
          >
            <option value="adam">adam</option>
            <option value="sgd">sgd</option>
            <option value="adamw">adamw</option>
          </select>
        </label>

        <label>
          Dropout
          <input
            inputMode="decimal"
            value={values.dropout}
            onChange={(event) => update('dropout', event.target.value)}
            aria-invalid={Boolean(errors.dropout)}
          />
          {errors.dropout ? <span className="field-error">{errors.dropout}</span> : null}
        </label>

        <label>
          Weight decay
          <input
            inputMode="decimal"
            value={values.weightDecay}
            onChange={(event) => update('weightDecay', event.target.value)}
          />
        </label>

        <label>
          Momentum
          <input
            inputMode="decimal"
            value={values.momentum}
            onChange={(event) => update('momentum', event.target.value)}
          />
        </label>

        <label>
          Seed
          <input
            inputMode="numeric"
            value={values.seed}
            onChange={(event) => update('seed', event.target.value)}
            aria-invalid={Boolean(errors.seed)}
          />
          {errors.seed ? <span className="field-error">{errors.seed}</span> : null}
        </label>

        <label>
          Requested GPUs
          <input
            inputMode="numeric"
            value={gpus}
            onChange={(event) => setGpus(event.target.value)}
            aria-invalid={Boolean(errors.gpus)}
          />
          {errors.gpus ? <span className="field-error">{errors.gpus}</span> : null}
          <span className="hint">
            Sets nvidia.com/gpu on the Kubernetes Job. Local runs fall back to CPU/MPS.
          </span>
        </label>

        <label className="span-2">
          Idempotency key
          <div className="inline-field">
            <input
              value={idempotencyKey}
              onChange={(event) => setIdempotencyKey(event.target.value)}
              aria-invalid={Boolean(errors.idempotency)}
            />
            <button
              type="button"
              className="button button-secondary"
              onClick={() => setIdempotencyKey(newIdempotencyKey())}
            >
              Regenerate
            </button>
          </div>
          {errors.idempotency ? <span className="field-error">{errors.idempotency}</span> : null}
          <span className="hint">Retrying with the same key replays the original run.</span>
        </label>
      </div>

      {submitError ? (
        <div className="form-error" role="alert">
          {submitError}
        </div>
      ) : null}

      <div className="form-actions">
        <button className="button button-primary" type="submit" disabled={submitting}>
          {submitting ? 'Submitting…' : 'Submit run'}
        </button>
        <span className="muted small">POST /api/runs · validated by pydantic</span>
      </div>
    </form>
  )
}
