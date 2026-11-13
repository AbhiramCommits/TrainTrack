import { createSlice, type PayloadAction } from '@reduxjs/toolkit'

export interface SelectionState {
  runIds: string[]
}

const initialState: SelectionState = { runIds: [] }

const selectionSlice = createSlice({
  name: 'selection',
  initialState,
  reducers: {
    toggleRun(state, action: PayloadAction<string>) {
      const runId = action.payload
      state.runIds = state.runIds.includes(runId)
        ? state.runIds.filter((id) => id !== runId)
        : [...state.runIds, runId]
    },
    setSelection(state, action: PayloadAction<string[]>) {
      state.runIds = [...new Set(action.payload)]
    },
    clearSelection(state) {
      state.runIds = []
    },
  },
})

export const { toggleRun, setSelection, clearSelection } = selectionSlice.actions

export function toggleSelection(selected: string[], runId: string): string[] {
  return selected.includes(runId)
    ? selected.filter((id) => id !== runId)
    : [...selected, runId]
}

export default selectionSlice.reducer
