import React, { useState } from 'react'
import axios from 'axios'
import { mediaUrl } from '../auth'
import Lightbox from './Lightbox'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * MergeSuggestionBanner — inline, high-visibility duplicate suggestion for a
 * person that is currently on screen (Photo Search result or People row).
 *
 * UI PROMINENCE ONLY. Nothing merges automatically: `auto_merge` stays false
 * in the backend and the single button here is the ONLY action — the click is
 * the operator's confirmation, and it calls the existing
 * POST /persons/{source}/merge-into/{keep} endpoint.
 *
 * Props:
 *   pair     {object}  one entry from GET /persons/duplicate-suggestions
 *                      ({code_a, code_b, photo_a, photo_b, similarity, ...})
 *   code     {string}  the SDT code of the person currently displayed
 *   onMerged {fn(source, keep)} called after a successful merge so the parent
 *                      can refresh its data.
 */
export default function MergeSuggestionBanner({ pair, code, onMerged }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)
  const [zoom, setZoom] = useState(null)   // photo to enlarge in the lightbox

  if (!pair) return null

  // The duplicate is always code_b; it is merged INTO code_a (the kept code),
  // matching the merge-into endpoint's source → target direction.
  const keep   = pair.code_a
  const source = pair.code_b
  const isViewedDup = code === pair.code_b

  const merge = async () => {
    if (busy || done) return
    setBusy(true); setError(null)
    try {
      await axios.post(`${API}/persons/${source}/merge-into/${keep}`, {})
      setDone(true)
      onMerged?.(source, keep)
    } catch (e) {
      setError(e.response?.data?.detail || 'Merge failed')
    } finally {
      setBusy(false)
    }
  }

  const photo = (path, key) => (
    <div
      key={key}
      onClick={() => path && setZoom(path)}
      title="Click to enlarge"
      style={{
        width: 34, height: 34, borderRadius: 7, overflow: 'hidden', flexShrink: 0,
        background: '#f0f0f0', cursor: path ? 'zoom-in' : 'default',
        border: '0.5px solid #fcd34d',
      }}
    >
      {path && (
        <img src={mediaUrl(path)} alt=""
             style={{ width: '100%', height: '100%', objectFit: 'cover' }}
             onError={e => { e.target.style.display = 'none' }} />
      )}
    </div>
  )

  return (
    <div style={{
      marginTop: 8, padding: '8px 10px', borderRadius: 10,
      background: '#fff7ed', border: '1px solid #fdba74',
      display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
    }}>
      {photo(pair.photo_b, 'b')}
      {photo(pair.photo_a, 'a')}

      <div style={{ flex: 1, minWidth: 140, fontSize: 11, lineHeight: 1.45 }}>
        <span style={{ fontWeight: 600, color: '#7c2d12' }}>
          ⚠ {Math.round(pair.similarity * 100)}% same face
        </span>
        <span style={{ color: '#92400e' }}>
          {' '}— {isViewedDup
            ? <>this person looks like a duplicate of <b style={{ fontFamily: 'monospace' }}>{keep}</b></>
            : <><b style={{ fontFamily: 'monospace' }}>{source}</b> looks like a duplicate of this person</>}
        </span>
      </div>

      <button
        onClick={merge}
        disabled={busy || done}
        title="Click to merge — this is the confirmation"
        style={{
          fontSize: 11, fontWeight: 600, padding: '6px 12px', borderRadius: 7,
          border: 'none', cursor: busy || done ? 'default' : 'pointer',
          fontFamily: 'inherit', whiteSpace: 'nowrap',
          background: done ? '#16a34a' : '#ea580c', color: '#fff',
          opacity: busy ? 0.6 : 1,
        }}
      >
        {busy ? 'Merging…'
          : done ? `Merged into ${keep} ✓`
          : isViewedDup ? `Merge into ${keep}` : `Merge ${source} into this`}
      </button>

      {error && (
        <div style={{ flexBasis: '100%', fontSize: 10, color: '#dc2626' }}>{error}</div>
      )}

      {zoom && <Lightbox src={zoom} onClose={() => setZoom(null)} />}
    </div>
  )
}
