import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type {
  Artifact,
  Experiment,
  LeaderboardEntry,
  Lineage,
  MetricResponse,
  QueueHealth,
  ResourceResponse,
  Run,
  RunListResponse,
  RunStatus,
  Snapshot,
  SubmitRunPayload,
} from '../types'

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api'
export const API_KEY =
  import.meta.env.VITE_API_KEY ?? 'tt_dev_local_key_do_not_use_in_prod'

export function streamUrl(runId: string): string {
  return `${API_BASE}/runs/${runId}/logs/stream?api_key=${encodeURIComponent(API_KEY)}`
}

interface RunQuery {
  status?: RunStatus
  experiment_id?: string
  job_kind?: string
  limit?: number
  cursor?: string
}

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: API_BASE,
    prepareHeaders: (headers) => {
      headers.set('X-API-Key', API_KEY)
      return headers
    },
  }),
  tagTypes: ['Run', 'Experiment', 'Snapshot', 'Metric', 'Artifact', 'Queue', 'Resource'],
  endpoints: (builder) => ({
    getRuns: builder.query<RunListResponse, RunQuery | void>({
      query: (args) => ({ url: '/runs', params: args ? { ...args } : {} }),
      providesTags: ['Run'],
    }),
    getRun: builder.query<Run, string>({
      query: (id) => `/runs/${id}`,
      providesTags: (_result, _error, id) => [{ type: 'Run', id }],
    }),
    submitRun: builder.mutation<Run, { payload: SubmitRunPayload; idempotencyKey: string }>({
      query: ({ payload, idempotencyKey }) => ({
        url: '/runs',
        method: 'POST',
        body: payload,
        headers: { 'Idempotency-Key': idempotencyKey },
      }),
      invalidatesTags: ['Run'],
    }),
    cancelRun: builder.mutation<Run, string>({
      query: (id) => ({ url: `/runs/${id}/cancel`, method: 'POST' }),
      invalidatesTags: (_result, _error, id) => [{ type: 'Run', id }, 'Run'],
    }),
    getExperiments: builder.query<{ experiments: Experiment[] }, void>({
      query: () => '/experiments',
      providesTags: ['Experiment'],
    }),
    getSnapshots: builder.query<{ snapshots: Snapshot[] }, void>({
      query: () => '/snapshots',
      providesTags: ['Snapshot'],
    }),
    getMetrics: builder.query<MetricResponse, { runId: string; names?: string; limit?: number }>({
      query: ({ runId, names, limit }) => ({
        url: `/runs/${runId}/metrics`,
        params: { ...(names ? { names } : {}), limit: limit ?? 5000 },
      }),
      providesTags: (_result, _error, arg) => [{ type: 'Metric', id: arg.runId }],
    }),
    getResources: builder.query<ResourceResponse, { runId: string; limit?: number }>({
      query: ({ runId, limit }) => ({
        url: `/runs/${runId}/resources`,
        params: { limit: limit ?? 1000 },
      }),
      providesTags: (_result, _error, arg) => [{ type: 'Resource', id: arg.runId }],
    }),
    getArtifacts: builder.query<{ artifacts: Artifact[] }, string>({
      query: (runId) => ({ url: '/artifacts', params: { run_id: runId } }),
      providesTags: ['Artifact'],
    }),
    getLineage: builder.query<Lineage, string>({
      query: (sha) => `/artifacts/${sha}/lineage`,
      providesTags: ['Artifact'],
    }),
    getLeaderboard: builder.query<
      { experiment_id: string; metric: string; cache_hit: boolean; entries: LeaderboardEntry[] },
      string
    >({
      query: (experimentId) => `/experiments/${experimentId}/leaderboard`,
      providesTags: ['Experiment'],
    }),
    getQueueHealth: builder.query<QueueHealth, void>({
      query: () => '/queue/health',
      providesTags: ['Queue'],
    }),
  }),
})

export const {
  useGetRunsQuery,
  useGetRunQuery,
  useSubmitRunMutation,
  useCancelRunMutation,
  useGetExperimentsQuery,
  useGetSnapshotsQuery,
  useGetMetricsQuery,
  useGetResourcesQuery,
  useGetArtifactsQuery,
  useGetLineageQuery,
  useGetLeaderboardQuery,
  useGetQueueHealthQuery,
} = api
