import type { ReactNode } from 'react'

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="state-box" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string
  detail: string
  action?: ReactNode
}) {
  return (
    <div className="state-box state-empty">
      <strong>{title}</strong>
      <span>{detail}</span>
      {action}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-box state-error" role="alert">
      <strong>Something went wrong</strong>
      <span>{message}</span>
      {onRetry ? (
        <button type="button" className="button button-secondary" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  )
}
