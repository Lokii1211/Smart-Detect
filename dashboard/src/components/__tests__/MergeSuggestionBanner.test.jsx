import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import axios from 'axios'
import MergeSuggestionBanner from '../MergeSuggestionBanner'

// A high-confidence (>=85%) duplicate-suggestion pair, as returned by
// GET /persons/duplicate-suggestions: code_b is the duplicate that gets
// merged INTO code_a (the kept code).
const pair = {
  code_a: 'SDT-0001', name_a: 'Alice', photo_a: 'snapshots/SDT-0001/registered.jpg',
  code_b: 'SDT-0002', name_b: 'Bob',   photo_b: 'snapshots/SDT-0002/registered.jpg',
  similarity: 0.92, confidence_band: 'high',
  guidance: 'Very likely the same person.', auto_merge: false,
}

const API = 'http://localhost:8000'

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('MergeSuggestionBanner', () => {
  it('shows the similarity as a percentage and names the duplicate pair', () => {
    render(<MergeSuggestionBanner pair={pair} code="SDT-0001" />)
    expect(screen.getByText(/92% same face/)).toBeInTheDocument()
    // The dup code appears in the description and in the merge button
    expect(screen.getAllByText(/SDT-0002/).length).toBeGreaterThan(0)
    expect(screen.getByText(/looks like a duplicate of this person/)).toBeInTheDocument()
  })

  it('offers to merge the duplicate into this person when viewing the kept code', () => {
    render(<MergeSuggestionBanner pair={pair} code="SDT-0001" />)
    expect(
      screen.getByRole('button', { name: /Merge SDT-0002 into this/i }),
    ).toBeInTheDocument()
  })

  it('offers "Merge into {keep}" when viewing the duplicate itself', () => {
    render(<MergeSuggestionBanner pair={pair} code="SDT-0002" />)
    expect(
      screen.getByRole('button', { name: /Merge into SDT-0001/i }),
    ).toBeInTheDocument()
  })

  it('merges the duplicate into the kept code on click and reports success', async () => {
    const post = vi.spyOn(axios, 'post').mockResolvedValue({ data: {} })
    const onMerged = vi.fn()
    render(<MergeSuggestionBanner pair={pair} code="SDT-0001" onMerged={onMerged} />)

    fireEvent.click(screen.getByRole('button', { name: /Merge SDT-0002 into this/i }))

    await waitFor(() =>
      expect(post).toHaveBeenCalledWith(
        `${API}/persons/SDT-0002/merge-into/SDT-0001`,
        {},
      ),
    )
    await waitFor(() => expect(onMerged).toHaveBeenCalledWith('SDT-0002', 'SDT-0001'))
    expect(screen.getByRole('button', { name: /Merged into SDT-0001/i })).toBeInTheDocument()
  })

  it('uses the duplicate as the merge source regardless of which code is being viewed', async () => {
    const post = vi.spyOn(axios, 'post').mockResolvedValue({ data: {} })
    const onMerged = vi.fn()
    // Viewing the duplicate itself (code_b): still merges code_b into code_a
    render(<MergeSuggestionBanner pair={pair} code="SDT-0002" onMerged={onMerged} />)

    fireEvent.click(screen.getByRole('button', { name: /Merge into SDT-0001/i }))

    await waitFor(() =>
      expect(post).toHaveBeenCalledWith(
        `${API}/persons/SDT-0002/merge-into/SDT-0001`,
        {},
      ),
    )
    await waitFor(() => expect(onMerged).toHaveBeenCalledWith('SDT-0002', 'SDT-0001'))
  })

  it('shows the API error detail when the merge fails', async () => {
    vi.spyOn(axios, 'post').mockRejectedValue({
      response: { data: { detail: 'Merge failed' } },
    })
    render(<MergeSuggestionBanner pair={pair} code="SDT-0001" />)

    fireEvent.click(screen.getByRole('button', { name: /Merge SDT-0002 into this/i }))

    await waitFor(() => expect(screen.getByText('Merge failed')).toBeInTheDocument())
  })

  it('never merges twice: the button is disabled after a successful merge', async () => {
    const post = vi.spyOn(axios, 'post').mockResolvedValue({ data: {} })
    render(<MergeSuggestionBanner pair={pair} code="SDT-0001" onMerged={() => {}} />)

    fireEvent.click(screen.getByRole('button', { name: /Merge SDT-0002 into this/i }))
    await waitFor(() => expect(post).toHaveBeenCalledTimes(1))

    const doneButton = screen.getByRole('button', { name: /Merged into SDT-0001/i })
    expect(doneButton).toBeDisabled()
    fireEvent.click(doneButton)
    expect(post).toHaveBeenCalledTimes(1)
  })

  it('renders nothing when there is no pair', () => {
    const { container } = render(<MergeSuggestionBanner pair={null} code="SDT-0001" />)
    expect(container.firstChild).toBeNull()
  })
})
