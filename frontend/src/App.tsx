import { Link, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { ExperimentComparePage } from './pages/ExperimentComparePage'
import { QueueHealthPage } from './pages/QueueHealthPage'
import { RunDetailPage } from './pages/RunDetailPage'
import { RunsPage } from './pages/RunsPage'
import { SubmitRunPage } from './pages/SubmitRunPage'

function NotFound() {
  return (
    <div className="page">
      <h1>Not found</h1>
      <p className="muted">
        That page does not exist. <Link to="/">Back to runs</Link>
      </p>
    </div>
  )
}

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<RunsPage />} />
        <Route path="submit" element={<SubmitRunPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="compare" element={<ExperimentComparePage />} />
        <Route path="queue" element={<QueueHealthPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
