import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { JobKind, RunStatus } from '../types'

export type StatusFilter = RunStatus | 'ALL'

export interface FiltersState {
  status: StatusFilter
  experimentId: string
  jobKind: JobKind | 'ALL'
  search: string
}

const initialState: FiltersState = {
  status: 'ALL',
  experimentId: 'ALL',
  jobKind: 'ALL',
  search: '',
}

const filtersSlice = createSlice({
  name: 'filters',
  initialState,
  reducers: {
    setStatus(state, action: PayloadAction<StatusFilter>) {
      state.status = action.payload
    },
    setExperiment(state, action: PayloadAction<string>) {
      state.experimentId = action.payload
    },
    setJobKind(state, action: PayloadAction<JobKind | 'ALL'>) {
      state.jobKind = action.payload
    },
    setSearch(state, action: PayloadAction<string>) {
      state.search = action.payload
    },
    resetFilters() {
      return initialState
    },
  },
})

export const { setStatus, setExperiment, setJobKind, setSearch, resetFilters } =
  filtersSlice.actions
export default filtersSlice.reducer
