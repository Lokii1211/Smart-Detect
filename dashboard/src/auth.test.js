import { describe, it, expect, beforeEach } from 'vitest'
import { streamMediaUrl } from './auth'

describe('streamMediaUrl', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it('appends the stream token when present', () => {
    sessionStorage.setItem('sd_stream_token', 'abc123')
    expect(streamMediaUrl('CAM-007')).toBe(
      'http://localhost:8000/camera/stream/CAM-007?token=abc123',
    )
  })

  it('appends the cache-buster with & (not a second ?) after the token', () => {
    sessionStorage.setItem('sd_stream_token', 'abc123')
    const url = streamMediaUrl('CAM-007', true)
    expect(url).toMatch(
      /^http:\/\/localhost:8000\/camera\/stream\/CAM-007\?token=abc123&t=\d+$/,
    )
  })

  it('uses ? for the cache-buster when no token is present', () => {
    const url = streamMediaUrl('CAM-007', true)
    expect(url).toMatch(/^http:\/\/localhost:8000\/camera\/stream\/CAM-007\?t=\d+$/)
  })
})
