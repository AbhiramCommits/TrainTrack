import { useNavigate } from 'react-router-dom'
import { useGetExperimentsQuery, useGetSnapshotsQuery, useSubmitRunMutation } from '../api/api'
import { SubmitRunForm } from '../components/SubmitRunForm'
import { ErrorState, LoadingState } from '../components/States'
import type { SubmitRunPayload } from '../types'

export function SubmitRunPage() {
  const navigate = useNavigate()
  const experimentsQuery = useGetExperimentsQuery()
  const snapshotsQuery = useGetSnapshotsQuery()
  const [submitRun, { isLoading, error }] = useSubmitRunMutation()

  const experiments = experimentsQuery.data?.experiments ?? []
  const snapshots = snapshotsQuery.data?.snapshots ?? []

  const handleSubmit = async (payload: SubmitRunPayload, idempotencyKey: string) => {
    const run = await submitRun({ payload, idempotencyKey }).unwrap()
    navigate(`/runs/${run.id}`)
  }

  const errorMessage =
    error && 'data' in error && typeof error.data === 'object' && error.data !== null
      ? JSON.stringify(error.data)
      : error
        ? 'Submission failed.'
        : null

  if (experimentsQuery.isLoading || snapshotsQuery.isLoading) {
    return <LoadingState label="Loading experiments and snapshots…" />
  }
  if (experimentsQuery.error || snapshotsQuery.error) {
    return <ErrorState message="Could not load experiments or snapshots from the API." />
  }

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Submit run</h1>
          <p className="muted">
            Runs are durably queued in Postgres, published to RabbitMQ, and executed by a worker.
          </p>
        </div>
      </header>
      <SubmitRunForm
        experiments={experiments}
        snapshots={snapshots}
        onSubmit={handleSubmit}
        submitting={isLoading}
        submitError={errorMessage}
      />
    </div>
  )
}
