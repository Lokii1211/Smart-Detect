import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { mediaUrl } from '../auth'
import Lightbox from '../components/Lightbox'
import MergeSuggestionBanner from '../components/MergeSuggestionBanner'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/* ── Helpers ───────────────────────────────────────── */
function toBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader()
    r.onload = e => res(e.target.result.split(',')[1])
    r.onerror = rej
    r.readAsDataURL(file)
  })
}

function fmtTime(iso) {
  if (!iso) return '—'
  return new Date(iso + (iso.endsWith('Z') ? '' : 'Z'))
    .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const METHOD_COLORS = {
  face:             { bg: '#eff6ff', color: '#1d4ed8', label: 'Face' },
  dress_color:      { bg: '#f0fdf4', color: '#166534', label: 'Color' },
  body_structure:   { bg: '#faf5ff', color: '#7c3aed', label: 'Body' },
  multi_feature:    { bg: '#fffbeb', color: '#b45309', label: 'Multi' },
  new_registration: { bg: '#f9fafb', color: '#6b7280', label: 'New' },
}

/* ── Search by SDT code ────────────────────────────── */
function SearchById({ navigate, onPerson }) {
  const [query,   setQuery]   = useState('')
  const [person,  setPerson]  = useState(null)
  const [error,   setError]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [zoom,    setZoom]    = useState(null)   // photo enlarged in lightbox

  const lookup = async () => {
    const raw = query.trim()
    if (!raw) return
    // Normalize "8" / "008" / "sdt-8" → SDT-0008 (codes are always 4 digits)
    const digits = raw.replace(/\D/g, '')
    const code = digits ? `SDT-${digits.padStart(4, '0')}` : raw.toUpperCase()
    setLoading(true); setError(null); setPerson(null); onPerson?.(null)
    try {
      const res = await axios.get(`${API}/persons/${encodeURIComponent(code)}`)
      setPerson(res.data); onPerson?.(res.data)
    } catch (err) {
      if (err.response?.status === 404) {
        // Fall back to a name search across all registered people
        try {
          const all = await axios.get(`${API}/persons`)
          const q = raw.toLowerCase()
          const hit = (all.data || []).find(p =>
            (p.display_name || '').toLowerCase().includes(q))
          if (hit) {
            const res2 = await axios.get(`${API}/persons/${encodeURIComponent(hit.unique_code)}`)
            setPerson(res2.data); onPerson?.(res2.data)
            return
          }
        } catch { /* fall through to error */ }
        setError(`No person found for "${raw}" (looked up ${code})`)
      } else {
        setError('Lookup failed — is the backend running?')
      }
    } finally {
      setLoading(false)
    }
  }

  const shots = (person?.appearances || []).filter(a => a.snapshot).slice(-6).reverse()

  return (
    <div className="card" style={{ padding: 12 }}>
      <div style={{ fontSize: 11, color: '#aaa', marginBottom: 8 }}>Or find by ID</div>
      <div style={{ display: 'flex', gap: 6 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') lookup() }}
          placeholder="SDT-0001"
          style={{
            flex: 1, fontSize: 12, padding: '6px 10px', borderRadius: 7,
            border: '0.5px solid #e0e0e0', outline: 'none',
            fontFamily: 'monospace', background: '#fafafa', minWidth: 0,
          }}
        />
        <button onClick={lookup} disabled={loading || !query.trim()} style={{
          padding: '6px 14px', borderRadius: 7, border: 'none', cursor: 'pointer',
          background: '#111', color: '#fff', fontSize: 11, fontWeight: 500,
          fontFamily: 'inherit', opacity: loading || !query.trim() ? 0.5 : 1,
        }}>{loading ? '…' : 'Find'}</button>
      </div>

      {error && (
        <div style={{ fontSize: 11, color: '#dc2626', marginTop: 8 }}>{error}</div>
      )}

      {person && (
        <div className="fade-in" style={{ marginTop: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              onClick={() => person.photo_path && setZoom(person.photo_path)}
              title="Click to enlarge"
              style={{
                width: 42, height: 42, borderRadius: 9, overflow: 'hidden', flexShrink: 0,
                background: '#f0f0f0', border: '0.5px solid #e8e8e8',
                cursor: person.photo_path ? 'zoom-in' : 'default',
              }}>
              {person.photo_path && (
                <img src={mediaUrl(person.photo_path)} alt=""
                     style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                     onError={e => { e.target.style.display = 'none' }} />
              )}
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#111' }}>
                {person.display_name || 'Unnamed'}
                <span style={{ fontFamily: 'monospace', fontWeight: 400, color: '#888', marginLeft: 6 }}>
                  {person.unique_code}
                </span>
              </div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>
                {person.person_type} · {person.total_sightings} sighting{person.total_sightings === 1 ? '' : 's'}
                {person.last_seen_at && <> · last seen {fmtTime(person.last_seen_at)}</>}
              </div>
            </div>
          </div>

          {/* Every camera/area this person has been seen in */}
          {(() => {
            const areas = {}
            for (const a of person.appearances || []) {
              const key = a.camera_id || a.location_name || '—'
              if (!areas[key]) areas[key] = { count: 0, last: null, zone: a.zone_id }
              areas[key].count += 1
              if (!areas[key].last || a.seen_at > areas[key].last) areas[key].last = a.seen_at
            }
            const list = Object.entries(areas)
            return list.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
                  Seen in {list.length} camera{list.length === 1 ? '' : 's'}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {list.map(([cam, info]) => (
                    <div key={cam} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      fontSize: 10, padding: '4px 8px', borderRadius: 6, background: '#f8fafc',
                    }}>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#334155' }}>
                        {cam}{info.zone ? ` · ${info.zone}` : ''}
                      </span>
                      <span style={{ color: '#94a3b8' }}>
                        {info.count}× · last {info.last ? fmtTime(info.last) : '—'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )
          })()}

          {shots.length > 0 && (
            <div style={{ display: 'flex', gap: 5, marginTop: 8, flexWrap: 'wrap' }}>
              {shots.map((a, i) => (
                <div
                  key={i}
                  onClick={() => setZoom(a.snapshot)}
                  title="Click to enlarge"
                  style={{
                    width: 46, height: 46, borderRadius: 6, overflow: 'hidden',
                    background: '#f0f0f0', border: '0.5px solid #e8e8e8',
                    cursor: 'zoom-in',
                  }}>
                  <img src={mediaUrl(a.snapshot)} alt=""
                       style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                       onError={e => { e.target.parentElement.style.display = 'none' }} />
                </div>
              ))}
            </div>
          )}

          <button
            onClick={() => navigate(`/people?code=${encodeURIComponent(person.unique_code)}`)}
            style={{
              marginTop: 8, width: '100%', padding: '6px 0', borderRadius: 7,
              border: '0.5px solid #e0e0e0', background: '#f8f8f8', cursor: 'pointer',
              fontSize: 11, color: '#6366f1', fontWeight: 500, fontFamily: 'inherit',
            }}>
            View all appearances →
          </button>

          {zoom && <Lightbox src={zoom} onClose={() => setZoom(null)} />}
        </div>
      )}
    </div>
  )
}

/* ── Face overlay corner brackets ──────────────────── */
function FaceBracket() {
  const s = { position: 'absolute', width: 16, height: 16, border: '2px solid #22c55e' }
  return (
    <>
      <div style={{ ...s, top: 8, left: 8, borderRight: 'none', borderBottom: 'none' }} />
      <div style={{ ...s, top: 8, right: 8, borderLeft: 'none', borderBottom: 'none' }} />
      <div style={{ ...s, bottom: 8, left: 8, borderRight: 'none', borderTop: 'none' }} />
      <div style={{ ...s, bottom: 8, right: 8, borderLeft: 'none', borderTop: 'none' }} />
    </>
  )
}

/* ── Camera card in result grid ─────────────────────── */
function CameraCard({ cam }) {
  const isLive    = cam.match_type === 'live'
  const isHistory = cam.match_type === 'history'
  const isMatch   = isLive || isHistory
  const borderColor = isLive ? '#22c55e' : isHistory ? '#3b82f6' : '#e8e8e8'
  const borderWidth = isMatch ? '1.5px' : '0.5px'

  return (
    <div style={{
      background: '#fff', borderRadius: 10,
      border: `${borderWidth} solid ${borderColor}`,
      overflow: 'hidden',
      opacity: isMatch ? 1 : 0.55,
    }}>
      {/* Camera screen — the REAL annotated stream when the camera is live */}
      <div style={{ position: 'relative', background: '#0a0a0a', aspectRatio: '16/9', overflow: 'hidden' }}>
        {cam.is_active ? (
          <img src={mediaUrl(`camera/stream/${cam.camera_id}`)} alt={cam.camera_name}
               style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
               onError={e => { e.target.style.display = 'none' }} />
        ) : (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            color: '#475569', fontSize: 11,
          }}>
            camera offline
          </div>
        )}

        {/* Match confidence badge */}
        {isMatch && (
          <div style={{ position: 'absolute', top: 6, left: 6 }}>
            <span style={{
              background: borderColor, color: '#fff', borderRadius: 4,
              fontSize: 9, fontWeight: 600, padding: '2px 6px',
            }}>
              {Math.round((cam.confidence || 0) * 100)}% match
            </span>
          </div>
        )}

        {/* Live badge */}
        {isLive && (
          <div style={{ position: 'absolute', top: 6, right: 6 }}>
            <span className="badge badge-green" style={{ fontSize: 9 }}>
              <span className="dot dot-green pulse" style={{ width: 4, height: 4 }} />LIVE
            </span>
          </div>
        )}
        {isHistory && (
          <div style={{ position: 'absolute', top: 6, right: 6 }}>
            <span className="badge badge-blue" style={{ fontSize: 9 }}>{fmtTime(cam.detected_at)}</span>
          </div>
        )}
        {!isMatch && (
          <div style={{ position: 'absolute', top: 6, right: 6 }}>
            <span className="badge badge-gray" style={{ fontSize: 9 }}>—</span>
          </div>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: '8px 10px' }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: '#111', marginBottom: 2 }}>
          {cam.camera_name || cam.camera_id}
        </div>
        <div style={{ fontSize: 10, color: '#aaa' }}>
          {cam.camera_id} · {cam.zone_id || 'main'}
          {cam.count > 1 && <span style={{ color: '#3b82f6', fontWeight: 600 }}> · {cam.count} sightings</span>}
        </div>
      </div>
    </div>
  )
}

/* ── Timeline ──────────────────────────────────────── */
function Timeline({ stops }) {
  if (!stops || stops.length === 0) return null
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Movement timeline today</div>
      <div style={{ display: 'flex', overflowX: 'auto', gap: 0, paddingBottom: 4 }}>
        {stops.map((s, i) => {
          const isFirst = i === 0
          const isLast  = i === stops.length - 1
          const mc      = METHOD_COLORS[s.method] || METHOD_COLORS.new_registration
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', flexShrink: 0 }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 80 }}>
                {/* Dot */}
                <div style={{
                  width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
                  background: isFirst ? '#111' : isLast ? '#22c55e' : '#d1d5db',
                  border: '2px solid #fff',
                  boxShadow: '0 0 0 1.5px ' + (isFirst ? '#111' : isLast ? '#22c55e' : '#d1d5db'),
                  margin: '4px 0',
                }} />
                <div style={{ fontSize: 10, fontWeight: 500, color: '#111', textAlign: 'center', marginTop: 4 }}>
                  {s.zone_id || s.location_name}
                </div>
                <div style={{ fontSize: 10, color: '#aaa', marginTop: 2 }}>{fmtTime(s.seen_at)}</div>
                <span style={{
                  marginTop: 4, padding: '1px 6px', borderRadius: 4, fontSize: 9,
                  background: mc.bg, color: mc.color,
                }}>
                  {mc.label}
                </span>
              </div>
              {/* Connecting line */}
              {!isLast && (
                <div style={{ width: 24, height: 1, background: '#e8e8e8', marginTop: 8.5, flexShrink: 0 }} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── PhotoSearch page ────────────────────────────── */
export default function PhotoSearch() {
  const navigate = useNavigate()
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(null)
  const [faceOk, setFaceOk]       = useState(null)   // true | false | null
  const [scope, setScope]         = useState('live_and_history')
  const [checking, setChecking]   = useState(false)
  const [searching, setSearching] = useState(false)
  const [result, setResult]       = useState(null)
  const [idPerson, setIdPerson]   = useState(null)   // person resolved via the "find by ID" card
  const [filter, setFilter]       = useState('all')
  const [zoom, setZoom]           = useState(null)          // photo in lightbox
  const [suggestions, setSuggestions] = useState([])        // high-confidence dups for the matched person

  // Camera count from status
  /* ── Real camera roster + live states (polled every 5s) ── */
  const [camStatus, setCamStatus] = useState({ cameras: [], active: 0 })
  useEffect(() => {
    const poll = () => axios.get(`${API}/camera/status`)
      .then(r => setCamStatus({
        cameras: r.data?.cameras || [],
        active:  r.data?.active_cameras || 0,
      }))
      .catch(() => {})
    poll()
    const iv = setInterval(poll, 5000)
    return () => clearInterval(iv)
  }, [])
  const camCount = camStatus.cameras.length || 1

  /* ── Realtime: re-run the search every 5s after a match ── */
  useEffect(() => {
    if (!result?.matched || !file) return
    let busy = false
    const iv = setInterval(async () => {
      if (busy) return
      busy = true
      try {
        const b64 = await toBase64(file)
        const res = await axios.post(`${API}/search/by-photo`, { base64_image: b64, scope: 'live_only' })
        setResult(prev => ({
          ...prev,
          live_matches: res.data.matched ? res.data.live_matches : [],
          is_live_now:  res.data.matched ? res.data.is_live_now  : false,
        }))
      } catch { /* keep last result */ }
      finally { busy = false }
    }, 5000)
    return () => clearInterval(iv)
  }, [result?.matched, file])

  /* ── Upload handler ──────────────────────────────── */
  const handleFile = async (f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setFaceOk(null)
    const url = URL.createObjectURL(f)
    setPreview(url)

    // Check face
    setChecking(true)
    try {
      const b64 = await toBase64(f)
      const res = await axios.post(`${API}/search/by-photo`, { base64_image: b64, scope: 'live_only', check_only: true })
      setFaceOk(res.data.face_detected !== false)
    } catch {
      setFaceOk(false)
    } finally {
      setChecking(false)
    }
  }

  /* ── High-confidence duplicate suggestions for the matched person ──
     Fetched once per matched code (the endpoint is rate-limited). Only
     suggestions >= 85% are surfaced inline — prominence, never auto-merge. */
  useEffect(() => {
    if (!result?.matched || !result?.unique_code) {
      setSuggestions([])
      return
    }
    let cancelled = false
    axios.get(`${API}/persons/duplicate-suggestions`)
      .then(r => {
        if (cancelled) return
        setSuggestions((r.data || []).filter(p =>
          p.similarity >= 0.85 &&
          (p.code_a === result.unique_code || p.code_b === result.unique_code)))
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [result?.matched, result?.unique_code])

  /* After a merge the identity data changed — re-run the search so the
     result reflects the merged person (and any stale code is replaced). */
  const handleMerged = () => {
    if (file && faceOk) handleSearch()
  }

  /* ── ID-search person → same camera grid as photo search ── */
  const handleIdPerson = (person) => {
    setIdPerson(person)
    if (person) { setResult(null); setFilter('all') }
  }

  /* ── Search ──────────────────────────────────────── */
  const handleSearch = async () => {
    if (!file || !faceOk) return
    setSearching(true)
    setIdPerson(null)
    try {
      const b64 = await toBase64(file)
      const res = await axios.post(`${API}/search/by-photo`, { base64_image: b64, scope })
      setResult(res.data)
    } catch (err) {
      setResult({ error: err.response?.data?.detail || 'Search failed' })
    } finally {
      setSearching(false)
    }
  }

  /* ── Build camera grid ───────────────────────────── */
  const buildGrid = () => {
    if (!result) return []
    const activeById = Object.fromEntries(camStatus.cameras.map(c => [c.camera_id, c]))
    const live    = (result.live_matches    || []).map(c => ({ ...c, match_type: 'live' }))
    // One history card per camera — count every sighting, keep the newest time,
    // so ALL areas the person passed through are visible at a glance
    const byCam = {}
    for (const c of result.history_matches || []) {
      const k = c.camera_id
      if (!byCam[k]) byCam[k] = { ...c, match_type: 'history', count: 0 }
      byCam[k].count += 1
      if ((c.detected_at || '') > (byCam[k].detected_at || '')) {
        byCam[k].detected_at = c.detected_at
        byCam[k].confidence  = c.confidence
      }
    }
    // A camera that's also live shouldn't repeat as history
    const liveIds = new Set(live.map(c => c.camera_id))
    const history = Object.values(byCam).filter(c => !liveIds.has(c.camera_id))
    // Enrich matches with the camera's real live state + label
    const allMatch = [...live, ...history].map(c => ({
      ...c,
      is_active:   activeById[c.camera_id]?.is_active || false,
      camera_name: c.camera_name || activeById[c.camera_id]?.label || c.camera_id,
    }))

    // Every other real camera appears as a "no match here" card
    const knownIds = new Set(allMatch.map(c => c.camera_id))
    const others = camStatus.cameras
      .filter(c => !knownIds.has(c.camera_id))
      .map(c => ({
        camera_id:   c.camera_id,
        camera_name: c.label || c.camera_id,
        zone_id:     c.zone_id,
        match_type:  'none',
        is_active:   c.is_active,
      }))

    const grid = [...allMatch, ...others]

    if (filter === 'live')    return grid.filter(c => c.match_type === 'live')
    if (filter === 'history') return grid.filter(c => c.match_type === 'history')
    if (filter === 'none')    return grid.filter(c => c.match_type === 'none')
    return grid
  }

  /* ── Build the same camera grid from an ID-search result ──
     The person detail's appearances become history cards (one per camera,
     counting every sighting); every other camera shows as "not detected". */
  const buildIdGrid = () => {
    if (!idPerson) return []
    const activeById = Object.fromEntries(camStatus.cameras.map(c => [c.camera_id, c]))
    const byCam = {}
    for (const a of idPerson.appearances || []) {
      const k = a.camera_id
      if (!byCam[k]) byCam[k] = {
        camera_id: k, zone_id: a.zone_id, match_type: 'history', count: 0,
        detected_at: a.seen_at, confidence: a.confidence,
      }
      byCam[k].count += 1
      if ((a.seen_at || '') > (byCam[k].detected_at || '')) {
        byCam[k].detected_at = a.seen_at
        byCam[k].confidence  = a.confidence
      }
    }
    const allMatch = Object.values(byCam).map(c => ({
      ...c,
      is_active:   activeById[c.camera_id]?.is_active || false,
      camera_name: activeById[c.camera_id]?.label || c.camera_id,
    }))
    const knownIds = new Set(allMatch.map(c => c.camera_id))
    const others = camStatus.cameras
      .filter(c => !knownIds.has(c.camera_id))
      .map(c => ({
        camera_id: c.camera_id, camera_name: c.label || c.camera_id,
        zone_id: c.zone_id, match_type: 'none', is_active: c.is_active,
      }))
    const grid = [...allMatch, ...others]

    if (filter === 'history') return grid.filter(c => c.match_type === 'history')
    if (filter === 'none')    return grid.filter(c => c.match_type === 'none')
    return grid
  }

  const grid = result ? buildGrid() : buildIdGrid()
  const liveCount    = result?.live_matches?.length    || 0
  const historyCount = result?.history_matches?.length || 0
  const gridTotal    = result ? liveCount + historyCount : (idPerson?.total_sightings || 0)

  return (
    <div className="fade-in" style={{
      display: 'grid',
      gridTemplateColumns: 'minmax(240px, 280px) 1fr',
      gap: 20,
      height: '100%',
      overflow: 'hidden',
    }}>
    {/* Collapses to single-column below 600px via inline media query workaround */}
    <style>{`@media (max-width: 600px) { .fade-in { grid-template-columns: 1fr !important; } }`}</style>

      {/* LEFT PANEL — scrollable */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflowY: 'auto' }}>

        {/* Upload card */}
        <div className="card">
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 500 }}>Upload photo</div>
            <div style={{ fontSize: 11, color: '#aaa', marginTop: 2 }}>Any clear photo of the person</div>
          </div>

          {/* Drop zone */}
          <label htmlFor="photo-input" style={{ cursor: 'pointer', display: 'block' }}>
            <div style={{
              position: 'relative', borderRadius: 10, overflow: 'hidden',
              aspectRatio: '1/1', background: preview ? '#000' : '#f5f5f5',
              border: preview ? 'none' : '1px dashed #e0e0e0',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              {preview ? (
                <>
                  <img src={preview} alt="preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  {faceOk === true && <FaceBracket />}
                  {faceOk === true && (
                    <div style={{
                      position: 'absolute', bottom: 8, left: '50%', transform: 'translateX(-50%)',
                      background: 'rgba(22,163,74,0.9)', color: '#fff', borderRadius: 20,
                      fontSize: 10, fontWeight: 500, padding: '3px 10px', whiteSpace: 'nowrap',
                    }}>
                      Face detected ✓
                    </div>
                  )}
                  {faceOk === false && (
                    <div style={{
                      position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.55)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <span style={{ fontSize: 11, color: '#f87171', fontWeight: 500, textAlign: 'center', padding: '0 8px' }}>
                        No face found —<br />try another photo
                      </span>
                    </div>
                  )}
                  {checking && (
                    <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <div className="spinner spinner-white" />
                    </div>
                  )}
                </>
              ) : (
                <div style={{ textAlign: 'center', color: '#bbb' }}>
                  <svg style={{ width: 24, height: 24, margin: '0 auto 6px', display: 'block' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span style={{ fontSize: 11 }}>Click to upload</span>
                  <div style={{ fontSize: 10, marginTop: 3 }}>JPG · PNG · WEBP</div>
                </div>
              )}
            </div>
          </label>
          <input id="photo-input" type="file" accept="image/*" style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files?.[0])} />
        </div>

        {/* Scope toggle */}
        <div className="card" style={{ padding: 12 }}>
          <div style={{ fontSize: 11, color: '#aaa', marginBottom: 8 }}>Search scope</div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[['live_and_history', 'Live + History'], ['live_only', 'Live only']].map(([v, l]) => (
              <button key={v} onClick={() => setScope(v)} style={{
                flex: 1, padding: '6px 0', borderRadius: 7, border: 'none', cursor: 'pointer',
                fontFamily: 'inherit', fontSize: 11, fontWeight: 500,
                background: scope === v ? '#111' : '#f5f5f5',
                color: scope === v ? '#fff' : '#888',
                transition: 'all 0.12s',
              }}>{l}</button>
            ))}
          </div>
        </div>

        {/* Search button */}
        <button
          id="search-btn"
          className="btn btn-black"
          style={{ width: '100%', padding: '10px', fontSize: 13 }}
          onClick={handleSearch}
          disabled={!faceOk || searching}
        >
          {searching ? (
            <><div className="spinner spinner-white" />Searching {camCount} camera{camCount !== 1 ? 's' : ''}…</>
          ) : 'Search all cameras →'}
        </button>

        {/* Search by SDT code */}
        <SearchById navigate={navigate} onPerson={handleIdPerson} />

        {/* Result summary */}
        {result && result.matched !== undefined && (
          <div className="card fade-in" style={{ padding: 12 }}>
            {result.matched ? (
              <>
                <button
                  onClick={() => navigate(`/people?code=${encodeURIComponent(result.unique_code)}`)}
                  title="View this person's appearances"
                  style={{
                    display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8,
                    padding: 8, borderRadius: 10, border: '0.5px solid #e0e0e0',
                    background: '#f8f8f8', cursor: 'pointer', width: '100%', textAlign: 'left',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = '#eef2ff' }}
                  onMouseLeave={e => { e.currentTarget.style.background = '#f8f8f8' }}
                >
                  {/* Registration photo — always face-visible (enrolment is
                      face-gated), unlike a live/sighting crop which may not be */}
                  {result.photo_path && (
                    <img
                      src={mediaUrl(result.photo_path)} alt=""
                      onClick={e => { e.stopPropagation(); setZoom(result.photo_path) }}
                      title="Click to enlarge"
                      style={{
                        width: 44, height: 44, borderRadius: 8, objectFit: 'cover',
                        flexShrink: 0, border: '1px solid #e0e0e0', cursor: 'zoom-in',
                      }} />
                  )}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <span style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 600, color: '#111' }}>
                      {result.unique_code}
                    </span>
                    <span style={{ fontSize: 10, color: '#6366f1' }}>view appearances →</span>
                  </div>
                </button>
                <dl style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  {[
                    ['Match confidence', `${((result.confidence || 0) * 100).toFixed(1)}%`, '#22c55e'],
                    ['Cameras searched', result.cameras_searched ?? 1],
                    ['Live matches',     liveCount,    '#22c55e'],
                    ['History matches',  historyCount, '#3b82f6'],
                    ['First seen',       fmtTime(result.first_seen)],
                    ['Last seen',        result.is_live_now ? '🟢 Live now' : fmtTime(result.last_seen), result.is_live_now ? '#22c55e' : undefined],
                  ].map(([k, v, c]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <dt style={{ fontSize: 11, color: '#aaa' }}>{k}</dt>
                      <dd style={{ fontSize: 12, fontWeight: 500, color: c || '#111' }}>{v}</dd>
                    </div>
                  ))}
                </dl>

                {/* Field — how close the other candidates were, not just the winner */}
                {(() => {
                  const others = (result.candidates || [])
                    .filter(c => c.unique_code !== result.unique_code)
                  if (others.length === 0) return null
                  return (
                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
                        Field — other candidates
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                        {others.map(c => (
                          <div key={c.unique_code} style={{
                            display: 'flex', alignItems: 'center', gap: 6, fontSize: 10,
                            padding: '3px 6px', borderRadius: 6, background: '#f8fafc',
                          }}>
                            <div
                              onClick={() => c.photo_path && setZoom(c.photo_path)}
                              title={c.photo_path ? 'Click to enlarge' : undefined}
                              style={{
                                width: 22, height: 22, borderRadius: 5, overflow: 'hidden',
                                background: '#f0f0f0', flexShrink: 0,
                                cursor: c.photo_path ? 'zoom-in' : 'default',
                              }}>
                              {c.photo_path && (
                                <img src={mediaUrl(c.photo_path)} alt=""
                                     style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                     onError={e => { e.target.style.display = 'none' }} />
                              )}
                            </div>
                            <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#334155' }}>
                              {c.unique_code}
                            </span>
                            {c.display_name && <span style={{ color: '#94a3b8' }}>· {c.display_name}</span>}
                            <span style={{ marginLeft: 'auto', fontWeight: 600, color: '#475569' }}>
                              {(c.similarity * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                })()}

                {/* High-confidence duplicate suggestion — inline, click to merge */}
                {suggestions.map(p => (
                  <MergeSuggestionBanner key={p.code_a + p.code_b} pair={p}
                    code={result.unique_code} onMerged={handleMerged} />
                ))}
              </>
            ) : (
              <div style={{ color: '#aaa', fontSize: 12, textAlign: 'center' }}>
                {result.message || 'No match found'}
              </div>
            )}
          </div>
        )}

        {/* Clear */}
        {(result || idPerson || preview) && (
          <button className="btn btn-white" style={{ width: '100%' }}
            onClick={() => {
              setFile(null); setPreview(null); setFaceOk(null)
              setResult(null); setIdPerson(null); setSuggestions([])
            }}>
            Clear search
          </button>
        )}
      </div>

      {/* RIGHT PANEL — scrollable */}
      <div style={{ height: '100%', overflowY: 'auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 13, fontWeight: 500 }}>
            Camera grid {result || idPerson ? `— ${gridTotal} detection${gridTotal === 1 ? '' : 's'}` : ''}
          </div>
          <div style={{ fontSize: 11, color: '#aaa', marginTop: 2 }}>
            Green = live now · Blue = history · Gray = not detected
          </div>
        </div>

        {/* Filter tabs */}
        {(result || idPerson) && (
          <div className="tab-strip" style={{ marginBottom: 14, width: 'fit-content' }}>
            {[['all','All cameras'], ['live','Live matches'], ['history','History only'], ['none','Not detected']].map(([v, l]) => (
              <button key={v} className={`tab-item ${filter === v ? 'active' : ''}`} onClick={() => setFilter(v)}>{l}</button>
            ))}
          </div>
        )}

        {result || idPerson ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
            {grid.map((cam, i) => <CameraCard key={i} cam={cam} />)}
          </div>
        ) : (
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            height: 300, color: '#bbb', gap: 8,
          }}>
            <svg style={{ width: 40, height: 40 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2}
                d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span style={{ fontSize: 13 }}>Upload a photo — or search by ID — to see cameras</span>
          </div>
        )}

        {/* Timeline */}
        {result?.timeline && <Timeline stops={result.timeline} />}
      </div>

      {zoom && <Lightbox src={zoom} onClose={() => setZoom(null)} />}
    </div>
  )
}
