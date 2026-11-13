import { useMemo, useState } from 'react'
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
import type { MetricPoint } from '../types'
import { mergeMetricSeries } from '../utils/format'

const COLORS = ['#38bdf8', '#f472b6', '#facc15', '#4ade80', '#c084fc', '#fb923c']

const GROUPS: Record<string, string[]> = {
  Loss: ['train_loss', 'train_loss_epoch', 'val_loss', 'test_loss'],
  Accuracy: ['accuracy', 'train_accuracy', 'val_accuracy', 'test_accuracy', 'test_macro_f1'],
  Latency: ['inference_p50_ms', 'inference_p95_ms'],
}

export function MetricCurves({ points }: { points: MetricPoint[] }) {
  const availableGroups = useMemo(
    () =>
      Object.entries(GROUPS).filter(([, names]) =>
        points.some((point) => names.includes(point.name)),
      ),
    [points],
  )
  const [group, setGroup] = useState(availableGroups[0]?.[0] ?? 'Loss')
  const names = useMemo(() => GROUPS[group] ?? [], [group])
  const series = useMemo(() => {
    const selected = points.filter((point) => names.includes(point.name))
    return [...new Set(selected.map((point) => point.name))].map((name) => ({
      label: name,
      points: selected.filter((point) => point.name === name),
    }))
  }, [points, names])
  const data = useMemo(() => mergeMetricSeries(series), [series])
  const accuracyView = names.some((name) => name.includes('accuracy'))

  if (points.length === 0) {
    return <p className="muted">No metrics recorded yet. Curves appear as the job trains.</p>
  }

  return (
    <div className="chart-panel">
      <div className="chart-toolbar" role="group" aria-label="Metric group">
        {availableGroups.map(([name]) => (
          <button
            key={name}
            type="button"
            className={`tab ${group === name ? 'tab-active' : ''}`}
            aria-pressed={group === name}
            onClick={() => setGroup(name)}
          >
            {name}
          </button>
        ))}
      </div>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke="var(--text-muted)" tickLine={false} />
            <YAxis
              stroke="var(--text-muted)"
              tickLine={false}
              width={64}
              domain={accuracyView ? [0.8, 1] : ['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                background: 'var(--surface-strong)',
                border: '1px solid var(--border)',
                borderRadius: 8,
              }}
            />
            <Legend />
            {series.map((item, index) => (
              <Line
                key={item.label}
                type="monotone"
                dataKey={item.label}
                stroke={COLORS[index % COLORS.length]}
                dot={false}
                strokeWidth={2}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
