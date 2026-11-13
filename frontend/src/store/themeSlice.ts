import { createSlice, type PayloadAction } from '@reduxjs/toolkit'

export type ThemeMode = 'dark' | 'light'

const STORAGE_KEY = 'traintrack-theme'

function initialMode(): ThemeMode {
  try {
    const stored = window.localStorage?.getItem(STORAGE_KEY)
    return stored === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

const themeSlice = createSlice({
  name: 'theme',
  initialState: { mode: initialMode() as ThemeMode },
  reducers: {
    setTheme(state, action: PayloadAction<ThemeMode>) {
      state.mode = action.payload
    },
    toggleTheme(state) {
      state.mode = state.mode === 'dark' ? 'light' : 'dark'
    },
  },
})

export const { setTheme, toggleTheme } = themeSlice.actions
export { STORAGE_KEY }
export default themeSlice.reducer
