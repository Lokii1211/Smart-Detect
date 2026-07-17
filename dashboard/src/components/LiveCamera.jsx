import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import CameraConnect from './CameraConnect.jsx'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const METHOD_META = {
  face:           { label: 'Face',      color: 'bg-green-500/20 text-green-300 border-green-500/40' },
  dress_color:    { label: 'Color',     color: 'bg-blue-500/20 text-blue-300 border-blue-500/40' },
  body_structure: { label: 'Structure', color: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40' },
  multi_feature:  { label: 'Multi',     color: 'bg-purple-500/20 text-purple-300 border-purple-500/40' },
  new_registration:{ label: 'New',      color: 'bg-slate-500/20 text-slate-300 border-slate-500/40' },
}

function MethodBadge({ method }) {
  const m = METHOD_META[method] || METHOD_META.new_registration
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${m.color}`}>
      {m.label}
    </span>
  )
}

function DetectionCard({ det }) {
  const conf = Math.round((det.confidence || 0) * 100)
  const time = det.detected_at
    ? new Date(det.detected_at + 'Z').toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '—'

  return (
    <div className="glass-card p-3 flex items-center gap-3 animate-fade-in hover:border-metro-accent/30 transition-colors">
      {/* Color swatch */}
      {det.color_hex ? (
        <div
          className="w-8 h-8 rounded-lg flex-shrink-0 border border-white/10"
          style={{ backgroundColor: det.color_hex }}
          title={`Dress color: ${det.color_hex}`}
        />
      ) : (
        <div className="w-8 h-8 rounded-lg bg-metro-700 flex items-center justify-center flex-shrink-0">
          <svg className="w-4 h-4 text-metro-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0zM12 14a7 7 0 0 0-7 7h14a7 7 0 0 0-7-7z" />
          </svg>
        </div>
      )}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="font-mono text-sm font-bold text-white">{det.unique_code}</span>
          <MethodBadge method={det.method} />
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>{conf}% confidence</span>
          <span>·</span>
          <span>{time}</span>
          {det.zone_id && <><span>·</span><span className="text-metro-teal">{det.zone_id}</span></>}
        </div>
      </div>
      <div className="text-right flex-shrink-0">
        <div className={`text-sm font-bold ${conf >= 80 ? 'text-metro-green' : conf >= 60 ? 'text-metro-amber' : 'text-metro-red'}`}>
          {conf}%
        </div>
      </div>
    </div>
  )
}

function LivePersonCard({ person }) {
  const m = METHOD_META[person.method] || METHOD_META.new_registration
  return (
    <div className="glass-card p-3 border border-metro-700/50 animate-slide-up">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-base font-bold text-white">{person.unique_code}</span>
        <MethodBadge method={person.method} />
      </div>
      <div className="flex items-center gap-2">
        {person.color_hex && (
          <div
            className="w-5 h-5 rounded border border-white/10 flex-shrink-0"
            style={{ backgroundColor: person.color_hex }}
          />
        )}
        <span className="text-xs text-slate-500">
          {person.total_sightings} sighting{person.total_sightings !== 1 ? 's' : ''}
        </span>
        {person.zone && <span className="text-xs text-metro-teal">{person.zone}</span>}
      </div>
    </div>
  )
}

export default function LiveCamera() {
  const [cameraStatus, setCameraStatus] = useState(null)   // null | {connected, camera_id, fps, ...}
  const [detections, setDetections]     = useState([])
  const [livePersons, setLivePersons]   = useState([])
  const [imgError, setImgError]         = useState(false)
  const [cameraId, setCameraId]         = useState('CAM-001')
  const imgRef = useRef(null)

  // ── Poll camera status every 3s ─────────────────────────────────────────────
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await axios.get(`${API}/camera/status`)
        setCameraStatus(res.data)
        if (res.data.camera_id) setCameraId(res.data.camera_id)
      } catch {
        setCameraStatus(null)
      }
    }
    fetchStatus()
    const iv = setInterval(fetchStatus, 3000)
    return () => clearInterval(iv)
  }, [])

  // ── Poll recent detections every 3s ─────────────────────────────────────────
  useEffect(() => {
    if (!cameraStatus?.connected) { setDetections([]); return }
    const fetchDetections = async () => {
      try {
        const res = await axios.get(`${API}/camera/detections/recent`)
        setDetections(Array.isArray(res.data) ? res.data : [])
      } catch { setDetections([]) }
    }
    fetchDetections()
    const iv = setInterval(fetchDetections, 3000)
    return () => clearInterval(iv)
  }, [cameraStatus?.connected])

  // ── Poll live persons every 2s ────────────────────────────────────────────
  useEffect(() => {
    if (!cameraStatus?.connected) { setLivePersons([]); return }
    const fetchLive = async () => {
      try {
        const res = await axios.get(`${API}/persons/live`)
        setLivePersons(Array.isArray(res.data) ? res.data : [])
      } catch { setLivePersons([]) }
    }
    fetchLive()
    const iv = setInterval(fetchLive, 2000)
    return () => clearInterval(iv)
  }, [cameraStatus?.connected])

  const streamUrl = `${API}/camera/stream/${cameraId}`
  const isConnected = cameraStatus?.connected === true

  return (
    <div className="space-y-6 animate-fade-in">
      {/* No camera connected — show connect panel */}
      {!isConnected && (
        <CameraConnect onStarted={(status) => setCameraStatus(status)} />
      )}

      {/* Camera active — show stream + detections */}
      {isConnected && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

          {/* ── Left: stream + recent detections ─── */}
          <div className="xl:col-span-2 space-y-4">

            {/* Stream player */}
            <div className="glass-card overflow-hidden p-0 relative">
              {/* Top overlay bar */}
              <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-2
                              bg-gradient-to-b from-black/70 to-transparent pointer-events-none">
                <span className="text-xs text-slate-300 font-mono">
                  {cameraStatus?.camera_id || cameraId}
                </span>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  <span className="text-xs font-bold text-red-400">LIVE</span>
                  {cameraStatus?.fps && (
                    <span className="text-xs text-slate-400 ml-2">{cameraStatus.fps.toFixed(1)} fps</span>
                  )}
                </div>
                <button
                  onClick={async () => {
                    await axios.post(`${API}/camera/stop`).catch(() => {})
                    setCameraStatus(null)
                  }}
                  className="pointer-events-auto text-xs px-2 py-1 rounded bg-red-600/70 hover:bg-red-600
                             text-white font-medium transition-colors"
                >
                  Stop
                </button>
              </div>

              {imgError ? (
                <div className="flex flex-col items-center justify-center h-72 bg-metro-900 text-slate-500 gap-3">
                  <svg className="w-12 h-12 text-metro-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M15 10l4.553-2.069A1 1 0 0 1 21 8.845v6.31a1 1 0 0 1-1.447.894L15 14M3 8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8z" />
                  </svg>
                  <p className="text-sm">Stream not available</p>
                  <button
                    onClick={() => { setImgError(false); if (imgRef.current) imgRef.current.src = streamUrl + '?t=' + Date.now() }}
                    className="text-xs px-3 py-1 rounded-lg bg-metro-700 hover:bg-metro-600 transition-colors"
                  >Retry</button>
                </div>
              ) : (
                <img
                  ref={imgRef}
                  src={streamUrl}
                  alt="Live camera stream"
                  className="w-full object-contain bg-black"
                  style={{ minHeight: '280px', maxHeight: '480px' }}
                  onError={() => setImgError(true)}
                />
              )}
            </div>

            {/* Recent detections list */}
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <svg className="w-4 h-4 text-metro-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 8v4l3 3m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
                </svg>
                Recent Detections (last 60s)
                <span className="ml-auto text-xs text-slate-500">{detections.length} entries</span>
              </h3>
              {detections.length === 0 ? (
                <p className="text-slate-600 text-sm text-center py-6">No detections yet — point camera at a person</p>
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                  {detections.map((d, i) => <DetectionCard key={i} det={d} />)}
                </div>
              )}
            </div>
          </div>

          {/* ── Right: Currently Visible panel ───── */}
          <aside className="space-y-4">
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-metro-green animate-pulse-slow" />
                Currently Visible
                <span className="ml-auto text-xs text-slate-500">{livePersons.length} person{livePersons.length !== 1 ? 's' : ''}</span>
              </h3>
              {livePersons.length === 0 ? (
                <p className="text-slate-600 text-sm text-center py-6">No one in frame</p>
              ) : (
                <div className="space-y-2">
                  {livePersons.map((p, i) => <LivePersonCard key={p.unique_code || i} person={p} />)}
                </div>
              )}
            </div>

            {/* Session stats */}
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Session Stats</h3>
              <dl className="space-y-2 text-sm">
                {[
                  { label: 'Persons Today',   value: cameraStatus?.persons_detected_today ?? '—' },
                  { label: 'Active Tracks',   value: cameraStatus?.active_tracks ?? '—' },
                  { label: 'Stream FPS',      value: cameraStatus?.fps ? cameraStatus.fps.toFixed(1) : '—' },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between">
                    <dt className="text-slate-500">{label}</dt>
                    <dd className="text-white font-mono font-semibold">{value}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* Method legend */}
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">ID Methods</h3>
              <div className="space-y-1.5">
                {[
                  { color: 'bg-green-500',  label: 'Face Recognition' },
                  { color: 'bg-blue-500',   label: 'Dress Color' },
                  { color: 'bg-yellow-400', label: 'Body Structure' },
                  { color: 'bg-purple-500', label: 'Multi-Feature' },
                  { color: 'bg-slate-500',  label: 'New Registration' },
                ].map(({ color, label }) => (
                  <div key={label} className="flex items-center gap-2 text-xs text-slate-400">
                    <span className={`w-2.5 h-2.5 rounded-full ${color} flex-shrink-0`} />
                    {label}
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  )
}
