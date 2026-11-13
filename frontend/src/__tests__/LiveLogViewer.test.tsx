import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { LiveLogViewer, type StreamFactory, type StreamSource } from '../components/LiveLogViewer'

class FakeSource implements StreamSource {
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  close = vi.fn()

  emit(payload: { ts: number; level: string; line: string }): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent<string>)
  }
}

describe('LiveLogViewer', () => {
  it('appends streamed lines and closes the stream on unmount', () => {
    const source = new FakeSource()
    const factory: StreamFactory = () => source
    const view = render(<LiveLogViewer runId="run-1" sourceFactory={factory} />)
    expect(screen.getByText('Waiting for log output…')).toBeInTheDocument()
    act(() => {
      source.emit({ ts: 1, level: 'info', line: 'epoch 1/2' })
      source.emit({ ts: 2, level: 'error', line: 'loss spike' })
    })
    expect(screen.getByText('epoch 1/2')).toBeInTheDocument()
    expect(screen.getByText('loss spike').parentElement).toHaveClass('log-line')
    view.unmount()
    expect(source.close).toHaveBeenCalled()
  })

  it('toggles autoscroll and clears the viewport', async () => {
    const source = new FakeSource()
    const factory: StreamFactory = () => source
    render(<LiveLogViewer runId="run-2" sourceFactory={factory} />)
    act(() => {
      source.emit({ ts: 3, level: 'info', line: 'ready' })
    })
    const pause = screen.getByRole('button', { name: 'Pause autoscroll' })
    act(() => pause.click())
    expect(screen.getByRole('button', { name: 'Resume autoscroll' })).toBeInTheDocument()
    act(() => screen.getByRole('button', { name: 'Clear' }).click())
    expect(screen.getByText('Waiting for log output…')).toBeInTheDocument()
  })
})
