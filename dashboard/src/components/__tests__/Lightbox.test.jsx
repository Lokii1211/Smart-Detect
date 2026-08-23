import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import Lightbox from '../Lightbox'

describe('Lightbox', () => {
  it('renders the enlarged image with the API URL applied', () => {
    const { container } = render(
      <Lightbox src="snapshots/SDT-0001/registered.jpg" onClose={() => {}} />,
    )
    // The image is decorative (alt="") so it is not exposed as role=img
    const img = container.querySelector('img')
    expect(img).toBeInTheDocument()
    expect(img).toHaveAttribute(
      'src',
      'http://localhost:8000/snapshots/SDT-0001/registered.jpg',
    )
  })

  it('shows the optional label', () => {
    render(<Lightbox src="snapshots/SDT-0001/registered.jpg" label="SDT-0001" onClose={() => {}} />)
    expect(screen.getByText('SDT-0001')).toBeInTheDocument()
  })

  it('closes when the Close button is clicked', () => {
    const onClose = vi.fn()
    render(<Lightbox src="snapshots/SDT-0001/registered.jpg" onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('closes when the Escape key is pressed', () => {
    const onClose = vi.fn()
    render(<Lightbox src="snapshots/SDT-0001/registered.jpg" onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('closes when the dark backdrop is clicked', () => {
    const onClose = vi.fn()
    render(<Lightbox src="snapshots/SDT-0001/registered.jpg" onClose={onClose} />)
    // The backdrop is the fixed full-screen overlay
    const backdrop = document.querySelector('div[style*="position: fixed"]')
    fireEvent.click(backdrop)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('does NOT close when the image itself is clicked', () => {
    const onClose = vi.fn()
    const { container } = render(
      <Lightbox src="snapshots/SDT-0001/registered.jpg" onClose={onClose} />,
    )
    fireEvent.click(container.querySelector('img'))
    expect(onClose).not.toHaveBeenCalled()
  })

  it('renders nothing when there is no src', () => {
    const { container } = render(<Lightbox src={null} onClose={() => {}} />)
    expect(container.firstChild).toBeNull()
  })

  it('removes the Escape listener on unmount', () => {
    const onClose = vi.fn()
    const { unmount } = render(<Lightbox src="snapshots/x.jpg" onClose={onClose} />)
    unmount()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
  })
})
