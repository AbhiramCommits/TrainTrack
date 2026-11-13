import { useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useGetExperimentsQuery, useGetLeaderboardQuery, useGetQueueHealthQuery, useGetRunsQuery } from '../api/api'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { RunsTable } from '../components/RunsTable'
import { useAppDispatch, useAppSelector } from '../store/store'
import {
  resetFilters,
  setExperiment,
  setJobKind,
  setSearch,
  setStatus,
  type StatusFilter,
} from '../store/filtersSlice'
import { STATUS_ORDER, filterRuns, formatPercent, shortId } from '../utils/format'

export function RunsPage() {
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const filters = useAppSelector((state) => state.filters)
  const runsQuery = useGetRunsQuery({ limit: 100 }, { pollingInterval: 3000 })
  const experimentsQuery = useGetExperimentsQuery()
  const queueQuery = useGetQueueHealthQuery(undefined, { pollingInterval: 5000 })
  const experiments = experimentsQuery.data?.experiments ?? []

  const effectiveExperimentId =
    filters.experimentId !== 'ALL'
      ? filters.experimentId
      : experiments[0]?.id ?? ''
  const leaderboardQuery = useGetLeaderboardQuery(effectiveExperimentId, {
    skip: !effectiveExperimentId,
    pollingInterval: 15000,
  })

  const filtered = useMemo(
    () => filterRuns(runsQuery.data?.runs ?? [], filters),
    [runsQuery.data, filters],
  )
  const counts = useMemo(() => {
    const runs = runsQuery.data?.runs ?? []
    return {
      total: runs.length,
      active: runs.filter((run) => run.status === 'RUNNING' || run.status === 'QUEUED').length,
      succeeded: runs.filter((run) => run.status === 'SUCCEEDED').length,
    }
  }, [runsQuery.data])

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Runs</h1>
          <p className="muted">
            Live registry of training and evaluation jobs. Polling every 3 seconds.
          </p>
        </div>
        <Link className="button button-primary" to="/submit">
          Submit run
        </Link>
      </header>

      <section className="stat-grid" aria-label="Registry summary">
        <div className="stat-card">
          <span className="stat-label">Total runs</span>
          <strong>{counts.total}</strong>
        </div>
        <div className="stat-card">
          <span className="stat-label">Active</span>
          <strong>{counts.active}</strong>
        </div>
        <div className="stat-card">
          <span className="stat-label">Succeeded</span>
          <strong>{counts.succeeded}</strong>
        </div>
        <div className="stat-card">
          <span className="stat-label">Queue depth</span>
          <strong>{queueQuery.data?.depth ?? '—'}</strong>
          <span className="muted small">DLQ {queueQuery.data?.dlq_messages ?? '—'}</span>
        </div>
      </section>

      <section className="panel" aria-label="Filters">
        <div className="filter-row">
          <label>
            Status
            <select
              value={filters.status}
              onChange={(event) => dispatch(setStatus(event.target.value as StatusFilter))}
            >
              <option value="ALL">All statuses</option>
              {STATUS_ORDER.map((status) => (
                <option key={status} value={status}>
                  {status}
                </option>
              ))}
            </select>
          </label>
          <label>
            Experiment
            <select
              value={filters.experimentId}
              onChange={(event) => dispatch(setExperiment(event.target.value))}
            >
              <option value="ALL">All experiments</option>
              {experiments.map((experiment) => (
                <option key={experiment.id} value={experiment.id}>
                  {experiment.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Kind
            <select
              value={filters.jobKind}
              onChange={(event) =>
                dispatch(setJobKind(event.target.value as 'ALL' | 'train' | 'eval' | 'noop'))
              }
            >
              <option value="ALL">All kinds</option>
              <option value="train">train</option>
              <option value="eval">eval</option>
              <option value="noop">noop</option>
            </select>
          </label>
          <label className="grow">
            Search
            <input
              type="search"
              placeholder="run id, lr, batch size…"
              value={filters.search}
              onChange={(event) => dispatch(setSearch(event.target.value))}
            />
          </label>
          <button type="button" className="button button-secondary" onClick={() => dispatch(resetFilters())}>
            Reset
          </button>
        </div>
      </section>

      <div className="content-grid">
        <section className="panel panel-wide" aria-label="Runs table">
          {runsQuery.isLoading ? <LoadingState label="Loading runs…" /> : null}
          {runsQuery.error ? (
            <ErrorState
              message="The run registry could not be reached."
              onRetry={() => void runsQuery.refetch()}
            />
          ) : null}
          {runsQuery.data ? (
            <>
              <div className="panel-heading">
                <h2>{filtered.length} runs</h2>
                {runsQuery.isFetching ? <span className="muted small">refreshing…</span> : null}
              </div>
              <RunsTable
                runs={filtered}
                experiments={experiments}
                onOpen={(runId) => navigate(`/runs/${runId}`)}
              />
            </>
          ) : null}
        </section>

        <aside className="panel" aria-label="Leaderboard">
          <h2>Leaderboard</h2>
          <p className="muted small">Best validation accuracy per run, cached for 30s.</p>
          {leaderboardQuery.isLoading ? <LoadingState label="Loading leaderboard…" /> : null}
          {leaderboardQuery.error ? <ErrorState message="Leaderboard unavailable." /> : null}
          {leaderboardQuery.data && leaderboardQuery.data.entries.length === 0 ? (
            <EmptyState title="No scored runs yet" detail="Runs appear after val_accuracy is ingested." />
          ) : null}
          <ol className="leaderboard">
            {leaderboardQuery.data?.entries.slice(0, 8).map((entry) => (
              <li key={entry.run_id}>
                <Link to={`/runs/${entry.run_id}`} className="leaderboard-id mono">
                  #{entry.rank} {shortId(entry.run_id, 10)}
                </Link>
                <span className="leaderboard-score">{formatPercent(entry.best_value)}</span>
                <span className="muted small">
                  lr {String(entry.hyperparams.lr ?? '—')} · batch{' '}
                  {String(entry.hyperparams.batch_size ?? '—')}
                </span>
              </li>
            ))}
          </ol>
        </aside>
      </div>
    </div>
  )
}
