import { useEffect } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../store/store'
import { STORAGE_KEY, toggleTheme } from '../store/themeSlice'

const LINKS = [
  { to: '/', label: 'Runs', end: true },
  { to: '/submit', label: 'Submit run' },
  { to: '/compare', label: 'Compare' },
  { to: '/queue', label: 'Queue health' },
]

export function Layout() {
  const dispatch = useAppDispatch()
  const mode = useAppSelector((state) => state.theme.mode)

  useEffect(() => {
    document.documentElement.dataset.theme = mode
    try {
      window.localStorage?.setItem(STORAGE_KEY, mode)
    } catch {
      // storage may be unavailable in private/test contexts
    }
  }, [mode])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            TT
          </span>
          <div>
            <strong>TrainTrack</strong>
            <span className="muted small">Experiment control plane</span>
          </div>
        </div>
        <nav aria-label="Primary">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => (isActive ? 'nav-link nav-active' : 'nav-link')}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button
            type="button"
            className="button button-secondary"
            onClick={() => dispatch(toggleTheme())}
            aria-label={`Switch to ${mode === 'dark' ? 'light' : 'dark'} theme`}
          >
            {mode === 'dark' ? 'Light theme' : 'Dark theme'}
          </button>
          <span className="muted small">Local control plane · v0.1.0</span>
        </div>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}
