import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, useGetRunsQuery } from '../api/api'
import { EmptyState, LoadingState } from '../components/States'
import { StatusChip } from '../components/StatusChip'
import { useAppDispatch, useAppSelector } from '../store/store'
import { clearSelection, toggleRun } from '../store/selectionSlice'
import type { MetricPoint, Run } from '../types'
import {
  bestMetric,
  formatPercent,
  hyperparamDiff,
  mergeMetricSeries,
  shortId,
} from '../utils/format'

const COLORS = ['#38bdf8', '#f472b6', '#facc15', '#4ade80', '#c084fc', '#fb923c']
const METRICS = ['val_accuracy', 'train_loss', 'test_accuracy'] as const

function useRunMetrics(runIds: string[]) {
  const dispatch = useAppDispatch()
  const [series, setSeries] = useState<Record<string, MetricPoint[]>>({})
  const [error, setError] = useState<string | null>(null)
  const key = runIds.join(',')

  useEffect(() => {
    let active = true
    if (runIds.length === 0) {
      setSeries({})
      return () => {
        active = false
      }
    }
    const requests = runIds.map((runId) =>
      dispatch(
        api.endpoints.getMetrics.initiate({
          runId,
          names: METRICS.join(','),
          limit: 5000,
        }),
      ),
    )
    void Promise.all(requests.map((request) => request.unwrap()))
      .then((results) => {
        if (!active) return
        const mapped: Record<string, MetricPoint[]> = {}
        results.forEach((result, index) => {
          const runId = runIds[index]
          if (runId) mapped[runId] = result.points
        })
        setSeries(mapped)
        setError(null)
      })
      .catch(() => {
        if (active) setError('Could not load metrics for the selected runs.')
      })
    return () => {
      active = false
      requests.forEach((request) => request.unsubscribe())
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, dispatch])

  return { series, error }
}

export function ExperimentComparePage() {
  const dispatch = useAppDispatch()
  const selected = useAppSelector((state) => state.selection.runIds)
  const runsQuery = useGetRunsQuery({ limit: 100 })
  const runs = useMemo(
    () => (runsQuery.data?.runs ?? []).filter((run) => run.job_kind === 'train'),
    [runsQuery.data],
  )
  const selectedRuns = useMemo(
    () => runs.filter((run) => selected.includes(run.id)),
    [runs, selected],
  )
  const { series, error } = useRunMetrics(selected)
  const [metric, setMetric] = useState<'val_accuracy' | 'train_loss'>('val_accuracy')
  const [sort, setSort] = useState<{ column: string; direction: 'asc' | 'desc' }>({
    column: 'key',
    direction: 'asc',
  })

  const chartData = useMemo(() => {
    const sources = selectedRuns.map((run) => ({
      label: `${shortId(run.id, 6)} · lr ${run.hyperparams.lr}/${run.hyperparams.batch_size}`,
      points: (series[run.id] ?? []).filter((point) => point.name === metric),
    }))
    return mergeMetricSeries(sources)
  }, [selectedRuns, series, metric])

  const rows = useMemo(() => {
    const hyperparamRows = hyperparamDiff(selectedRuns)
    const metricRows = [
      {
        key: 'val_accuracy',
        values: selectedRuns.map((run) => formatPercent(bestMetric(series[run.id] ?? [], 'val_accuracy'))),
      },
      {
        key: 'test_accuracy',
        values: selectedRuns.map((run) => formatPercent(bestMetric(series[run.id] ?? [], 'test_accuracy'))),
      },
      { key: 'status', values: selectedRuns.map((run) => run.status) },
    ]
    const combined = [...hyperparamRows, ...metricRows]
    const direction = sort.direction === 'asc' ? 1 : -1
    return [...combined].sort((a, b) => {
      if (sort.column === 'key') return a.key.localeCompare(b.key) * direction
      const index = selectedRuns.findIndex((run) => run.id === sort.column)
      if (index < 0) return 0
      const left = a.values[index] ?? ''
      const right = b.values[index] ?? ''
      const leftNumber = Number(left)
      const rightNumber = Number(right)
      if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
        return (leftNumber - rightNumber) * direction
      }
      return left.localeCompare(right) * direction
    })
  }, [selectedRuns, series, sort])

  const toggleSort = (column: string) => {
    setSort((current) =>
      current.column === column
        ? { column, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { column, direction: 'asc' },
    )
  }

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Compare experiments</h1>
          <p className="muted">
            Select training runs to overlay validation curves and diff their hyperparameters.
          </p>
        </div>
        {selected.length > 0 ? (
          <button type="button" className="button button-secondary" onClick={() => dispatch(clearSelection())}>
            Clear selection ({selected.length})
          </button>
        ) : null}
      </header>

      <section className="panel">
        <h2>Select runs</h2>
        {runsQuery.isLoading ? <LoadingState label="Loading runs…" /> : null}
        {runsQuery.data && runs.length === 0 ? (
          <EmptyState title="No training runs" detail="Submit a run to compare results." />
        ) : null}
        <div className="run-picker">
          {runs.map((run: Run) => (
            <label key={run.id} className={`run-option ${selected.includes(run.id) ? 'run-selected' : ''}`}>
              <input
                type="checkbox"
                checked={selected.includes(run.id)}
                onChange={() => dispatch(toggleRun(run.id))}
              />
              <span className="mono">{shortId(run.id, 10)}</span>
              <StatusChip status={run.status} />
              <span className="muted small">
                lr {String(run.hyperparams.lr)} · batch {String(run.hyperparams.batch_size)} ·{' '}
                {String(run.hyperparams.optimizer)}
              </span>
            </label>
          ))}
        </div>
      </section>

      {selectedRuns.length === 0 ? (
        <EmptyState title="Nothing selected" detail="Pick at least two runs to compare curves and hyperparameters." />
      ) : (
        <>
          <section className="panel">
            <div className="panel-heading">
              <h2>Overlaid curves</h2>
              <div className="toolbar" role="group" aria-label="Compare metric">
                <button
                  type="button"
                  className={`tab ${metric === 'val_accuracy' ? 'tab-active' : ''}`}
                  aria-pressed={metric === 'val_accuracy'}
                  onClick={() => setMetric('val_accuracy')}
                >
                  Validation accuracy
                </button>
                <button
                  type="button"
                  className={`tab ${metric === 'train_loss' ? 'tab-active' : ''}`}
                  aria-pressed={metric === 'train_loss'}
                  onClick={() => setMetric('train_loss')}
                >
                  Training loss
                </button>
              </div>
            </div>
            {error ? <p className="form-error">{error}</p> : null}
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                  <XAxis dataKey="step" stroke="var(--text-muted)" />
                  <YAxis
                    stroke="var(--text-muted)"
                    width={64}
                    domain={metric === 'val_accuracy' ? [0.8, 1] : ['auto', 'auto']}
                  />
                  <Tooltip />
                  <Legend />
                  {selectedRuns.map((run, index) => (
                    <Line
                      key={run.id}
                      type="monotone"
                      dataKey={`${shortId(run.id, 6)} · lr ${run.hyperparams.lr}/${run.hyperparams.batch_size}`}
                      stroke={COLORS[index % COLORS.length]}
                      dot={false}
                      strokeWidth={2}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel">
            <h2>Hyperparameter and metric diff</h2>
            <div className="table-scroll">
              <table className="diff-table">
                <thead>
                  <tr>
                    <th scope="col">
                      <button type="button" className="sort-button" onClick={() => toggleSort('key')}>
                        Parameter {sort.column === 'key' ? (sort.direction === 'asc' ? '↑' : '↓') : ''}
                      </button>
                    </th>
                    {selectedRuns.map((run) => (
                      <th key={run.id} scope="col">
                        <button
                          type="button"
                          className="sort-button"
                          onClick={() => toggleSort(run.id)}
                        >
                          {shortId(run.id, 8)}{' '}
                          {sort.column === run.id ? (sort.direction === 'asc' ? '↑' : '↓') : ''}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.key}>
                      <th scope="row">{row.key}</th>
                      {row.values.map((value, index) => (
                        <td key={`${row.key}-${selectedRuns[index]?.id ?? index}`} className="mono">
                          {value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
