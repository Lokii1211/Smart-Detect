import React, { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/* ── Mini components ──────────────────────────────── */
function StatCard({ id, label, value, sub, subColor = '#22c55e', loading }) {
  return (
    <div id={id} className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span className="label">{label}</span>
      {loading ? (
        <div style={{ height: 32, width: 60, background: '#f0f0f0', borderRadius: 6 }} />
      ) : (
        <span style={{ fontSize: 26, fontWeight: 500, color: '#111', letterSpacing: '-0.03em', lineHeight: 1 }}>
          {value ?? '—'}
        </span>
      )}
      {sub && <span style={{ fontSize: 11, color: subColor }}>{sub}</span>}
    </div>
  )
}

function LiveFeedPreview() {
  const [status, setStatus] = useState(null)
  useEffect(() => {
    const fetch = () => axios.get(`${API}/camera/status`).then(r => setStatus(r.data)).catch(() => {})
    fetch()
    const iv = setInterval(fetch, 5000)
    return () => clearInterval(iv)
  }, [])

  const cameras = status?.cameras || []
  // Multiple cameras can run at once (webcam + uploaded videos) — prefer
  // whichever is actually active right now, not just the first one ever seen
  const preview = cameras.find(c => c.is_active) || cameras[0]
  const isConnected = !!preview?.is_active
  const streamUrl = `${API}/camera/stream/${preview?.camera_id || 'CAM-001'}`

  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
      <div style={{ padding: '12px 14px', borderBottom: '0.5px solid #f0f0f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 13, fontWeight: 500 }}>
          Live Feed{preview ? ` — ${preview.label || preview.camera_id}` : ''}
        </span>
        {isConnected
          ? <span className="badge badge-green"><span className="dot dot-green pulse" style={{ width: 5, height: 5 }} />LIVE</span>
          : <span className="badge badge-gray">Offline</span>
        }
      </div>
      <div className="camera-screen" style={{ borderRadius: 0, borderBottomLeftRadius: 11, borderBottomRightRadius: 11 }}>
        <div className="camera-grid-lines" />
        {isConnected ? (
          <img
            key={streamUrl}
            src={streamUrl} alt="Live camera"
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            onError={e => { e.target.style.display = 'none' }}
          />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 8 }}>
            <svg style={{ width: 32, height: 32, color: '#444' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
            </svg>
            <span style={{ fontSize: 11, color: '#555' }}>No camera connected</span>
          </div>
        )}
      </div>
    </div>
  )
}

function RecentDetections() {
  const [items, setItems] = useState([])
  useEffect(() => {
    const fetch = () => axios.get(`${API}/camera/detections/recent?limit=10`).then(r => setItems(r.data)).catch(() => {})
    fetch()
    const iv = setInterval(fetch, 5000)
    return () => clearInterval(iv)
  }, [])

  const METHOD_COLOR = {
    face:             '#3b82f6',
    dress_color:      '#22c55e',
    body_structure:   '#a855f7',
    multi_feature:    '#f59e0b',
    new_registration: '#6b7280',
  }

  return (
    <div className="card" style={{ overflow: 'hidden', padding: 0, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div style={{ padding: '12px 14px', borderBottom: '0.5px solid #f0f0f0', flexShrink: 0 }}>
        <span style={{ fontSize: 13, fontWeight: 500 }}>Recent Detections</span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {items.length === 0 ? (
          <div style={{ padding: '24px 16px', textAlign: 'center', color: '#bbb', fontSize: 12 }}>
            No detections yet
          </div>
        ) : items.map((d, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '9px 14px', borderBottom: '0.5px solid #f7f8fa',
          }}>
            <div style={{
              width: 28, height: 28, borderRadius: 8, flexShrink: 0,
              background: d.color_hex || '#f0f0f0',
              border: '0.5px solid #e8e8e8',
            }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 500, color: '#111', fontFamily: 'monospace' }}>
                {d.unique_code}
              </div>
              <div style={{ fontSize: 10, color: '#aaa', marginTop: 1 }}>
                {d.detected_at ? new Date(d.detected_at + 'Z').toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'}
                {d.zone_id && ` · ${d.zone_id}`}
              </div>
            </div>
            <span style={{
              padding: '2px 7px', borderRadius: 6,
              fontSize: 10, fontWeight: 500,
              background: `${METHOD_COLOR[d.method] || '#888'}18`,
              color: METHOD_COLOR[d.method] || '#888',
            }}>
              {d.method === 'new_registration' ? 'New' :
               d.method === 'face' ? 'Face' :
               d.method === 'dress_color' ? 'Color' :
               d.method === 'body_structure' ? 'Body' : 'Multi'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Dashboard page ──────────────────────────────── */
export default function Dashboard() {
  const [persons,   setPersons]   = useState(null)
  const [camStatus, setCamStatus] = useState(null)
  const [loading,   setLoading]   = useState(true)

  useEffect(() => {
    const fetch = async () => {
      const [p, c] = await Promise.allSettled([
        axios.get(`${API}/persons`),
        axios.get(`${API}/camera/status`),
      ])
      if (p.status === 'fulfilled') setPersons(p.value.data.length)
      if (c.status === 'fulfilled') setCamStatus(c.value.data)
      setLoading(false)
    }
    fetch()
    const iv = setInterval(fetch, 5000)
    return () => clearInterval(iv)
  }, [])

  const isLive = camStatus?.connected === true

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Stats row */}
      <div className="stats-row">
        <StatCard id="stat-registered"  label="Registered"    value={persons}                            loading={loading} />
        <StatCard id="stat-tracks"      label="Active Tracks" value={camStatus?.active_tracks ?? 0}      loading={loading} sub={isLive ? 'Live tracking' : 'No stream'} subColor={isLive ? '#22c55e' : '#aaa'} />
        <StatCard id="stat-today"       label="Sightings Today" value={camStatus?.persons_detected_today ?? 0} loading={loading} />
        <StatCard id="stat-cameras"     label="Cameras"       value={isLive ? '1/1' : '0/1'}             loading={loading} sub={isLive ? 'Connected' : 'Offline'} subColor={isLive ? '#22c55e' : '#f59e0b'} />
      </div>

      {/* Feed + detections */}
      <div className="two-col">
        <div className="two-col-left"><LiveFeedPreview /></div>
        <div className="two-col-right"><RecentDetections /></div>
      </div>
    </div>
  )
}
