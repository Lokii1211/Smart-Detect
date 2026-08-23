import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import axios from 'axios'
import Dashboard from '../../pages/Dashboard'

vi.mock('axios')

const statusPayload = {
  connected: true,
  active_cameras: 2,
  total_cameras: 3,
  active_tracks: 5,
  persons_detected_today: 12,
  cameras: [
    { camera_id: 'CAM-004', label: 'ChokePoint C1', is_active: true },
    { camera_id: 'CAM-005', label: 'ChokePoint C2', is_active: true },
    { camera_id: 'CAM-006', label: 'ChokePoint C3', is_active: false },
  ],
}

beforeEach(() => {
  vi.clearAllMocks()
  sessionStorage.clear()
  axios.get.mockImplementation((url) => {
    if (url.includes('/camera/status')) {
      return Promise.resolve({ data: statusPayload })
    }
    if (url.includes('/persons')) {
      return Promise.resolve({ data: [{ unique_code: 'SDT-0001' }] })
    }
    if (url.includes('/camera/detections/recent')) {
      return Promise.resolve({ data: [] })
    }
    return Promise.resolve({ data: {} })
  })
})

describe('Dashboard live-feed tiles', () => {
  it('renders one clickable tile per active camera', async () => {
    render(<Dashboard />)
    const tiles = await screen.findAllByRole('button', { name: /^Enlarge / })
    expect(tiles).toHaveLength(2)
    expect(tiles[0]).toHaveAttribute('aria-label', 'Enlarge CAM-004')
    expect(tiles[1]).toHaveAttribute('aria-label', 'Enlarge CAM-005')
  })

  it('opens the Lightbox with the live stream URL when a tile is clicked', async () => {
    render(<Dashboard />)
    const tile = await screen.findByRole('button', { name: 'Enlarge CAM-004' })
    fireEvent.click(tile)

    const lightbox = document.querySelector('div[style*="position: fixed"]')
    const img = lightbox.querySelector('img')
    expect(img).toBeInTheDocument()
    expect(img).toHaveAttribute(
      'src',
      'http://localhost:8000/camera/stream/CAM-004',
    )
    // The enlarged view labels the camera
    expect(within(lightbox).getByText(/CAM-004 · ChokePoint C1/)).toBeInTheDocument()
  })

  it('closes the Lightbox via the Close button and Escape', async () => {
    render(<Dashboard />)
    const tile = await screen.findByRole('button', { name: 'Enlarge CAM-005' })
    fireEvent.click(tile)
    expect(document.querySelector('div[style*="position: fixed"] img')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(document.querySelector('div[style*="position: fixed"]')).toBeNull()

    fireEvent.click(tile)
    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => {
      expect(document.querySelector('div[style*="position: fixed"]')).toBeNull()
    })
  })
})
