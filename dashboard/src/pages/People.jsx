import React, { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/* ── Auth (same pattern as Alerts.jsx) ─────────────── */
function getToken() {
  return localStorage.getItem('smartdetect_token')
}

async function ensureToken() {
  let token = getToken()
  if (token) return token
  try {
    const res = await axios.post(`${API}/auth/login`, { username: 'operator', password: 'smartOp2024' })
    token = res.data.access_token
    localStorage.setItem('smartdetect_token', token)
    return token
  } catch {
    return null
  }
}

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {}
}

const TYPE_COLORS = {
  staff:   { bg: '#3b82f618', fg: '#3b82f6' },
  visitor: { bg: '#22c55e18', fg: '#22c55e' },
  unknown: { bg: '#f0f0f0',   fg: '#888' },
}

// Appearance gallery: the person's captured frames from cameras/videos
function PersonAppearances({ code }) {
  const [detail, setDetail] = useState(null)
  const [error,  setError]  = useState(false)

  useEffect(() => {
    axios.get(`${API}/persons/${code}`)
      .then(r => setDetail(r.data))
      .catch(() => setError(true))
  }, [code])

  if (error) return (
    <div style={{ padding: '10px 14px 14px 70px', fontSize: 11, color: '#dc2626' }}>
      Could not load appearances.
    </div>
  )
  if (!detail) return (
    <div style={{ padding: '10px 14px 14px 70px', fontSize: 11, color: '#bbb' }}>
      Loading appearances…
    </div>
  )

  const shots = (detail.appearances || []).filter(a => a.snapshot)
  return (
    <div style={{ padding: '2px 14px 14px 70px', background: '#fbfbfc' }}>
      <div style={{ fontSize: 11, color: '#888', margin: '8px 0 8px' }}>
        {detail.total_sightings} appearance{detail.total_sightings === 1 ? '' : 's'}
        {detail.first_seen_at && (
          <> · first seen {new Date(detail.first_seen_at + 'Z').toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</>
        )}
      </div>
      {shots.length === 0 ? (
        <div style={{ fontSize: 11, color: '#bbb' }}>
          No captured frames yet — frames are saved each time this person is sighted.
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {shots.slice(-12).reverse().map((a, i) => (
            <div key={i} style={{ width: 92 }}>
              <div style={{
                width: 92, height: 92, borderRadius: 8, overflow: 'hidden',
                background: '#f0f0f0', border: '0.5px solid #e8e8e8',
              }}>
                <img src={`${API}/${a.snapshot}`} alt=""
                     style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                     onError={e => { e.target.parentElement.style.display = 'none' }} />
              </div>
              <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 3, lineHeight: 1.4 }}>
                {a.seen_at ? new Date(a.seen_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'}
                <br />{a.camera_id || a.location_name || ''}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Duplicate-code suggestions: same face registered under two SDT codes
function DuplicateSuggestions({ onMerged }) {
  const [pairs,   setPairs]   = useState([])
  const [busy,    setBusy]    = useState(null)   // code being merged
  const [hidden,  setHidden]  = useState(new Set())

  const fetchPairs = async () => {
    try {
      const token = await ensureToken()
      const r = await axios.get(`${API}/persons/duplicate-suggestions`,
        { headers: authHeaders(token) })
      setPairs(r.data || [])
    } catch { /* endpoint unavailable — hide silently */ }
  }

  useEffect(() => { fetchPairs() }, [])

  const merge = async (from, into) => {
    if (!window.confirm(`Merge ${from} into ${into}? ${from} will be deleted and all its sightings and photos move to ${into}. This cannot be undone.`)) return
    setBusy(from)
    try {
      const token = await ensureToken()
      await axios.post(`${API}/persons/${from}/merge-into/${into}`, {},
        { headers: authHeaders(token) })
      await fetchPairs()
      onMerged()
    } catch (e) {
      alert(e.response?.data?.detail || 'Merge failed')
    } finally { setBusy(null) }
  }

  const visible = pairs.filter(p => !hidden.has(p.code_a + p.code_b))
  if (visible.length === 0) return null

  return (
    <div style={{
      margin: '10px 14px 0', padding: '10px 12px', borderRadius: 10,
      background: '#fffbeb', border: '1px solid #fde68a',
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: '#92400e', marginBottom: 8 }}>
        ⚠ Possible duplicate identities — same face under two codes
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {visible.map(p => (
          <div key={p.code_a + p.code_b} style={{
            display: 'flex', alignItems: 'center', gap: 8, fontSize: 11,
          }}>
            {[p.photo_a, p.photo_b].map((ph, i) => (
              <div key={i} style={{
                width: 30, height: 30, borderRadius: 6, overflow: 'hidden',
                background: '#f0f0f0', flexShrink: 0,
              }}>
                {ph && <img src={`${API}/${ph}`} alt=""
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  onError={e => { e.target.style.display = 'none' }} />}
              </div>
            ))}
            <span style={{ fontFamily: 'monospace', color: '#334155' }}>
              {p.code_b} ≈ {p.code_a}
            </span>
            <span style={{ color: '#b45309' }}>{Math.round(p.similarity * 100)}% same face</span>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
              <button
                onClick={() => merge(p.code_b, p.code_a)}
                disabled={busy !== null}
                style={{
                  fontSize: 10, fontWeight: 600, padding: '4px 10px', borderRadius: 6,
                  border: 'none', cursor: 'pointer', fontFamily: 'inherit',
                  background: '#f59e0b', color: '#fff', opacity: busy ? 0.6 : 1,
                }}>
                {busy === p.code_b ? 'Merging…' : `Merge into ${p.code_a}`}
              </button>
              <button
                onClick={() => setHidden(h => new Set([...h, p.code_a + p.code_b]))}
                style={{
                  fontSize: 10, padding: '4px 8px', borderRadius: 6,
                  border: '1px solid #fde68a', cursor: 'pointer', fontFamily: 'inherit',
                  background: 'transparent', color: '#92400e',
                }}>
                Not same
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PersonRow({ p, onSaved, expanded, onToggle }) {
  const [editing, setEditing] = useState(false)
  const [name, setName]       = useState(p.display_name || '')
  const [saving, setSaving]   = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      const token = await ensureToken()
      await axios.put(`${API}/persons/${p.unique_code}`,
        { display_name: name },
        { headers: authHeaders(token) })
      setEditing(false)
      onSaved()
    } catch { /* keep editing open on failure */ }
    setSaving(false)
  }

  const setType = async (person_type) => {
    try {
      const token = await ensureToken()
      await axios.put(`${API}/persons/${p.unique_code}`,
        { person_type },
        { headers: authHeaders(token) })
      onSaved()
    } catch { /* ignore */ }
  }

  const tc = TYPE_COLORS[p.person_type] || TYPE_COLORS.unknown
  const photoUrl = p.photo_path ? `${API}/${p.photo_path.replace(/^snapshots\//, 'snapshots/')}` : null

  return (
    <div style={{ borderBottom: '0.5px solid #f7f8fa', background: expanded ? '#fbfbfc' : 'transparent' }}>
    <div onClick={onToggle} style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '10px 14px', cursor: 'pointer',
    }}>
      {/* Photo */}
      <div style={{
        width: 44, height: 44, borderRadius: 10, flexShrink: 0, overflow: 'hidden',
        background: '#f0f0f0', border: '0.5px solid #e8e8e8',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {photoUrl ? (
          <img src={photoUrl} alt={p.unique_code}
               style={{ width: '100%', height: '100%', objectFit: 'cover' }}
               onError={e => { e.target.style.display = 'none' }} />
        ) : (
          <svg style={{ width: 18, height: 18, color: '#bbb' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.6}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        )}
      </div>

      {/* Name + code */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {editing ? (
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }} onClick={e => e.stopPropagation()}>
            <input
              autoFocus value={name}
              onChange={e => setName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') setEditing(false) }}
              placeholder="Person name"
              style={{
                fontSize: 12, padding: '4px 8px', borderRadius: 6,
                border: '1px solid #ddd', outline: 'none', width: 150,
                fontFamily: 'inherit',
              }}
            />
            <button onClick={save} disabled={saving} style={{
              fontSize: 11, padding: '4px 10px', borderRadius: 6, border: 'none',
              background: '#111', color: '#fff', cursor: 'pointer', fontFamily: 'inherit',
            }}>{saving ? '…' : 'Save'}</button>
            <button onClick={() => setEditing(false)} style={{
              fontSize: 11, padding: '4px 8px', borderRadius: 6, border: 'none',
              background: 'transparent', color: '#888', cursor: 'pointer', fontFamily: 'inherit',
            }}>Cancel</button>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 13, fontWeight: 500, color: '#111' }}>
              {p.display_name || <span style={{ color: '#bbb' }}>Unnamed</span>}
            </span>
            <button onClick={e => { e.stopPropagation(); setName(p.display_name || ''); setEditing(true) }} title="Rename" style={{
              border: 'none', background: 'transparent', cursor: 'pointer',
              color: '#bbb', fontSize: 11, padding: 2, fontFamily: 'inherit',
            }}>✎</button>
          </div>
        )}
        <div style={{ fontSize: 11, color: '#aaa', fontFamily: 'monospace', marginTop: 1 }}>
          {p.unique_code}
        </div>
      </div>

      {/* Type selector */}
      <select
        value={p.person_type || 'unknown'}
        onClick={e => e.stopPropagation()}
        onChange={e => setType(e.target.value)}
        style={{
          fontSize: 11, fontWeight: 500, padding: '3px 6px', borderRadius: 6,
          border: 'none', background: tc.bg, color: tc.fg,
          cursor: 'pointer', fontFamily: 'inherit',
        }}
      >
        <option value="unknown">Unknown</option>
        <option value="visitor">Visitor</option>
        <option value="staff">Staff</option>
      </select>

      {/* Stats */}
      <div style={{ textAlign: 'right', flexShrink: 0, width: 110 }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: '#111' }}>
          {p.total_sightings} sighting{p.total_sightings === 1 ? '' : 's'}
        </div>
        <div style={{ fontSize: 10, color: '#aaa', marginTop: 1 }}>
          {p.last_seen_at
            ? `Seen ${new Date(p.last_seen_at + 'Z').toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`
            : 'Never seen'}
        </div>
      </div>

      {/* Expand chevron */}
      <span style={{
        fontSize: 10, color: '#bbb', flexShrink: 0,
        transform: expanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s',
      }}>▶</span>
    </div>

    {expanded && <PersonAppearances code={p.unique_code} />}
    </div>
  )
}

export default function People() {
  const [persons, setPersons] = useState(null)
  const [query,   setQuery]   = useState('')
  const [searchParams, setSearchParams] = useSearchParams()
  // ?code=SDT-0001 (from Photo Search) opens that person's appearances
  const expandedCode = searchParams.get('code')

  const fetchPersons = () => {
    axios.get(`${API}/persons`).then(r => setPersons(r.data)).catch(() => setPersons([]))
  }

  useEffect(() => {
    fetchPersons()
    const iv = setInterval(fetchPersons, 10000)
    return () => clearInterval(iv)
  }, [])

  const toggle = (code) => {
    setSearchParams(code === expandedCode ? {} : { code }, { replace: true })
  }

  const filtered = (persons || []).filter(p => {
    if (!query) return true
    const q = query.toLowerCase()
    return (p.display_name || '').toLowerCase().includes(q)
        || p.unique_code.toLowerCase().includes(q)
  })

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div className="card" style={{ padding: 0, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <div style={{
          padding: '12px 14px', borderBottom: '0.5px solid #f0f0f0',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexShrink: 0,
        }}>
          <span style={{ fontSize: 13, fontWeight: 500 }}>
            Registered People {persons !== null && <span style={{ color: '#aaa', fontWeight: 400 }}>({filtered.length})</span>}
          </span>
          <input
            value={query} onChange={e => setQuery(e.target.value)}
            placeholder="Search name or code…"
            style={{
              fontSize: 12, padding: '5px 10px', borderRadius: 8,
              border: '0.5px solid #e0e0e0', outline: 'none', width: 190,
              fontFamily: 'inherit', background: '#fafafa',
            }}
          />
        </div>

        <DuplicateSuggestions onMerged={fetchPersons} />

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {persons === null ? (
            <div style={{ padding: '24px 16px', textAlign: 'center', color: '#bbb', fontSize: 12 }}>Loading…</div>
          ) : filtered.length === 0 ? (
            <div style={{ padding: '24px 16px', textAlign: 'center', color: '#bbb', fontSize: 12 }}>
              {query ? 'No matches' : 'No people registered yet — start a camera and step in front of it'}
            </div>
          ) : filtered.map(p => (
            <PersonRow key={p.unique_code} p={p} onSaved={fetchPersons}
              expanded={p.unique_code === expandedCode}
              onToggle={() => toggle(p.unique_code)} />
          ))}
        </div>
      </div>
    </div>
  )
}
