import { useMemo } from 'react'
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
import type { ResourceSample } from '../types'
import { mergeMetricSeries } from '../utils/format'

export function GpuUtilPanel({
  samples,
  device,
}: {
  samples: ResourceSample[]
  device: string | null
}) {
  const utilization = useMemo(
    () =>
      mergeMetricSeries([
        {
          label: 'utilization_pct',
          points: samples.map((sample) => ({
            name: 'utilization_pct',
            step: sample.step,
            value: sample.utilization_pct,
            wall_time: sample.wall_time,
          })),
        },
      ]),
    [samples],
  )
  const memory = useMemo(
    () =>
      mergeMetricSeries([
        {
          label: 'memory_used_mb',
          points: samples.map((sample) => ({
            name: 'memory_used_mb',
            step: sample.step,
            value: sample.memory_used_mb,
            wall_time: sample.wall_time,
          })),
        },
      ]),
    [samples],
  )

  if (samples.length === 0) {
    return (
      <p className="muted">
        No device samples yet. The training loop posts utilization and memory every 40 steps.
      </p>
    )
  }

  return (
    <div className="chart-panel">
      <p className="muted small">
        Device: <strong>{device ?? samples[samples.length - 1]?.device}</strong>. For MPS the
        utilization value is the training process CPU utilization (no vendor counter exists).
      </p>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={utilization}>
            <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke="var(--text-muted)" />
            <YAxis domain={[0, 100]} stroke="var(--text-muted)" width={48} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="utilization_pct"
              stroke="#38bdf8"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={memory}>
            <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke="var(--text-muted)" />
            <YAxis stroke="var(--text-muted)" width={64} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="memory_used_mb"
              stroke="#4ade80"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
