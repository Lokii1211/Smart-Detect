import React, { useEffect, useState, useCallback } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const CONFIG_FIELDS = [
  { key: 'detection_confidence',     label: 'Detection Confidence',     type: 'range', min: 0.1, max: 0.9, step: 0.05, unit: '' },
  { key: 'nms_iou_threshold',       label: 'NMS IoU Threshold',        type: 'range', min: 0.1, max: 0.9, step: 0.05, unit: '' },
  { key: 'face_match_threshold',    label: 'Face Match Threshold',     type: 'range', min: 0.3, max: 0.95, step: 0.01, unit: '' },
  { key: 'reid_match_threshold',    label: 'Re-ID Match Threshold',    type: 'range', min: 0.3, max: 0.95, step: 0.01, unit: '' },
  { key: 'color_match_threshold',   label: 'Color Match Threshold',    type: 'number', min: 5, max: 100, step: 5, unit: 'HSV' },
  { key: 'sighting_cooldown_seconds', label: 'Sighting Cooldown',      type: 'number', min: 5, max: 300, step: 5, unit: 'sec' },
  { key: 'max_simultaneous_streams', label: 'Max Simultaneous Streams', type: 'number', min: 1, max: 8, step: 1, unit: '' },
  { key: 'yolo_input_size',         label: 'YOLO Input Size',          type: 'select', options: [320, 416, 640], unit: 'px' },
  { key: 'insightface_det_size',    label: 'InsightFace Det Size',     type: 'select', options: [128, 160, 320, 640], unit: 'px' },
]

function getToken() {
  return localStorage.getItem('smartdetect_token')
}

async function ensureToken() {
  let token = getToken()
  if (token) return token
  try {
    const res = await axios.post(`${API}/auth/login`, { username: 'admin', password: 'smartAdmin2024' })
    token = res.data.access_token
    localStorage.setItem('smartdetect_token', token)
    return token
  } catch {
    return null
  }
}

export default function Settings() {
  const [config, setConfig] = useState(null)
  const [info, setInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState(null)
  const [dirty, setDirty] = useState({})

  const fetchSettings = useCallback(async () => {
    try {
      const token = await ensureToken()
      const res = await axios.get(`${API}/settings`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
      setConfig(res.data.config)
      setInfo({ active_streams: res.data.active_streams, max_streams: res.data.max_streams, db_type: res.data.database_url_type })
      setError(null)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load settings')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchSettings() }, [fetchSettings])

  const handleChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    setDirty(prev => ({ ...prev, [key]: true }))
    setSaved(false)
  }

  const handleSave = async () => {
    setSaving(true)
    setSaved(false)
    try {
      const token = await ensureToken()
      const changedKeys = Object.keys(dirty)
      const payload = {}
      changedKeys.forEach(k => { payload[k] = config[k] })
      await axios.put(`${API}/settings`, payload, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
      setDirty({})
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  const hasDirty = Object.keys(dirty).length > 0

  if (loading) {
    return (
      <div className="fade-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 300 }}>
        <div className="spinner" />
      </div>
    )
  }

  if (error && !config) {
    return (
      <div className="card fade-in" style={{ maxWidth: 480, textAlign: 'center', padding: 30 }}>
        <div style={{ fontSize: 13, color: '#ef4444', marginBottom: 8 }}>{error}</div>
        <button className="btn btn-black" onClick={fetchSettings}>Retry</button>
      </div>
    )
  }

  return (
    <div className="fade-in" style={{ maxWidth: 640, display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* System info */}
      {info && (
        <div className="card" style={{ padding: '14px 18px' }}>
          <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 10, color: '#111' }}>System Info</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            {[
              ['Database', info.db_type === 'sqlite' ? 'SQLite' : 'PostgreSQL'],
              ['Active Streams', `${info.active_streams} / ${info.max_streams}`],
              ['API', API],
            ].map(([label, val]) => (
              <div key={label}>
                <div style={{ fontSize: 10, color: '#aaa', textTransform: 'uppercase', fontWeight: 600 }}>{label}</div>
                <div style={{ fontSize: 12, color: '#111', marginTop: 2, fontFamily: 'monospace' }}>{val}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detection thresholds */}
      <div className="card" style={{ padding: '14px 18px' }}>
        <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 14, color: '#111' }}>Detection & Matching Thresholds</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {CONFIG_FIELDS.filter(f => f.key.includes('threshold') || f.key.includes('confidence')).map(field => (
            <ConfigRow key={field.key} field={field} value={config[field.key]} isDirty={dirty[field.key]} onChange={v => handleChange(field.key, v)} />
          ))}
        </div>
      </div>

      {/* Camera & processing */}
      <div className="card" style={{ padding: '14px 18px' }}>
        <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 14, color: '#111' }}>Camera & Processing</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {CONFIG_FIELDS.filter(f => !f.key.includes('threshold') && !f.key.includes('confidence')).map(field => (
            <ConfigRow key={field.key} field={field} value={config[field.key]} isDirty={dirty[field.key]} onChange={v => handleChange(field.key, v)} />
          ))}
        </div>
      </div>

      {/* Save button */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <button
          className="btn btn-black"
          style={{ padding: '10px 24px', fontSize: 13, opacity: hasDirty ? 1 : 0.4 }}
          disabled={!hasDirty || saving}
          onClick={handleSave}
        >
          {saving ? <><div className="spinner spinner-white" /> Saving...</> : 'Save Changes'}
        </button>
        {saved && <span style={{ fontSize: 12, color: '#22c55e', fontWeight: 500 }}>Saved successfully</span>}
        {error && config && <span style={{ fontSize: 12, color: '#ef4444' }}>{error}</span>}
      </div>
    </div>
  )
}

function ConfigRow({ field, value, isDirty, onChange }) {
  const displayVal = field.type === 'range' ? value?.toFixed(2) : value

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, color: '#111', fontWeight: isDirty ? 600 : 400 }}>
          {field.label}
          {isDirty && <span style={{ color: '#f59e0b', marginLeft: 4, fontSize: 10 }}>modified</span>}
        </div>
      </div>

      {field.type === 'range' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            type="range" min={field.min} max={field.max} step={field.step}
            value={value ?? field.min}
            onChange={e => onChange(parseFloat(e.target.value))}
            style={{ width: 120, accentColor: '#111' }}
          />
          <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#555', minWidth: 36, textAlign: 'right' }}>
            {displayVal}
          </span>
        </div>
      )}

      {field.type === 'number' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <input
            type="number" min={field.min} max={field.max} step={field.step}
            value={value ?? ''}
            onChange={e => onChange(parseFloat(e.target.value))}
            style={{
              width: 72, padding: '4px 8px', borderRadius: 6,
              border: '1px solid #e0e0e0', fontSize: 12, fontFamily: 'monospace',
              textAlign: 'right',
            }}
          />
          {field.unit && <span style={{ fontSize: 10, color: '#aaa' }}>{field.unit}</span>}
        </div>
      )}

      {field.type === 'select' && (
        <select
          value={value ?? ''}
          onChange={e => onChange(parseInt(e.target.value))}
          style={{
            padding: '4px 8px', borderRadius: 6,
            border: '1px solid #e0e0e0', fontSize: 12, fontFamily: 'monospace',
            background: '#fff',
          }}
        >
          {field.options.map(o => (
            <option key={o} value={o}>{o}{field.unit ? ` ${field.unit}` : ''}</option>
          ))}
        </select>
      )}
    </div>
  )
}
