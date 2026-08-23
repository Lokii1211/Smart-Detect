import React, { useEffect } from 'react'
import { mediaUrl } from '../auth'

/**
 * Lightbox — click-to-enlarge overlay for person photos.
 *
 * Renders a full-screen dark overlay with the image enlarged. Closes on
 * backdrop click, the Close button, or the Escape key. `src` is a server
 * path (e.g. "snapshots/SDT-0001/registered.jpg") — mediaUrl() appends the
 * short-lived stream token so the snapshot is served past auth.
 */
export default function Lightbox({ src, alt = '', label, onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!src) return null

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(8, 8, 12, 0.88)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 28, cursor: 'zoom-out',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
          maxWidth: '94vw', maxHeight: '92vh',
        }}
      >
        <img
          src={mediaUrl(src)}
          alt={alt}
          style={{
            maxWidth: '94vw', maxHeight: '82vh', objectFit: 'contain',
            borderRadius: 10, background: '#000',
            boxShadow: '0 24px 70px rgba(0,0,0,0.55)',
          }}
        />
        {label && <div style={{ fontSize: 12, color: '#d1d5db' }}>{label}</div>}
        <button
          onClick={onClose}
          style={{
            padding: '7px 18px', borderRadius: 8, cursor: 'pointer',
            border: '1px solid rgba(255,255,255,0.35)',
            background: 'rgba(255,255,255,0.08)', color: '#fff',
            fontSize: 12, fontFamily: 'inherit',
          }}
        >
          Close ✕
        </button>
      </div>
    </div>
  )
}
