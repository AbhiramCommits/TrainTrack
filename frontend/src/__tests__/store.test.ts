import { describe, expect, it } from 'vitest'
import selectionReducer, {
  clearSelection,
  setSelection,
  toggleRun,
  toggleSelection,
} from '../store/selectionSlice'
import themeReducer, { setTheme, toggleTheme } from '../store/themeSlice'

describe('selection slice', () => {
  it('adds and removes run ids without duplicates', () => {
    let state = selectionReducer({ runIds: [] }, toggleRun('run-a'))
    state = selectionReducer(state, toggleRun('run-b'))
    state = selectionReducer(state, toggleRun('run-a'))
    expect(state.runIds).toEqual(['run-b'])
    expect(toggleSelection(['a', 'b'], 'c')).toEqual(['a', 'b', 'c'])
    expect(toggleSelection(['a', 'b'], 'a')).toEqual(['b'])
  })

  it('deduplicates a set selection and clears', () => {
    let state = selectionReducer({ runIds: [] }, setSelection(['a', 'a', 'b']))
    expect(state.runIds).toEqual(['a', 'b'])
    state = selectionReducer(state, clearSelection())
    expect(state.runIds).toEqual([])
  })
})

describe('theme slice', () => {
  it('toggles between dark and light and sets explicitly', () => {
    let state = themeReducer({ mode: 'dark' }, toggleTheme())
    expect(state.mode).toBe('light')
    state = themeReducer(state, setTheme('dark'))
    expect(state.mode).toBe('dark')
  })
})
