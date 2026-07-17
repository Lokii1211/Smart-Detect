import React, { useEffect, useState, useCallback } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const SEVERITY_STYLE = {
  critical: { bg: '#fef2f2', border: '#fca5a5', color: '#dc2626', icon: '!!' },
  warning:  { bg: '#fffbeb', border: '#fcd34d', color: '#b45309', icon: '!' },
  info:     { bg: '#f0f9ff', border: '#93c5fd', color: '#2563eb', icon: 'i' },
}

const TYPE_LABELS = {
  watchlist:       'Watchlist',
  camera_offline:  'Camera',
  crowd:           'Crowd',
}

function fmtTime(iso) {
  if (!iso) return '—'
  const d = new Date(iso + (iso.endsWith('Z') ? '' : 'Z'))
  const now = new Date()
  const diffMin = Math.floor((now - d) / 60000)
  if (diffMin < 1) return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffMin < 1440) return `${Math.floor(diffMin / 60)}h ago`
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

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

export default function Alerts() {
  const [alerts, setAlerts] = useState([])
  const [watchlist, setWatchlist] = useState([])
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('alerts')
  const [addCode, setAddCode] = useState('')
  const [addReason, setAddReason] = useState('')
  const [adding, setAdding] = useState(false)

  const fetchData = useCallback(async () => {
    try {
      const token = await ensureToken()
      const h = authHeaders(token)
      const [aRes, wRes] = await Promise.all([
        axios.get(`${API}/alerts`, { params: { limit: 100 }, headers: h }),
        axios.get(`${API}/watchlist`, { headers: h }),
      ])
      setAlerts(aRes.data)
      setWatchlist(wRes.data)
    } catch {}
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchData()
    const iv = setInterval(fetchData, 8000)
    return () => clearInterval(iv)
  }, [fetchData])

  const markRead = async (id) => {
    const token = await ensureToken()
    await axios.put(`${API}/alerts/${id}/read`, {}, { headers: authHeaders(token) }).catch(() => {})
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, is_read: true } : a))
  }

  const markAllRead = async () => {
    const token = await ensureToken()
    await axios.put(`${API}/alerts/read-all`, {}, { headers: authHeaders(token) }).catch(() => {})
    setAlerts(prev => prev.map(a => ({ ...a, is_read: true })))
  }

  const addToWatchlist = async () => {
    if (!addCode.trim()) return
    setAdding(true)
    try {
      const token = await ensureToken()
      await axios.post(`${API}/watchlist`, { unique_code: addCode.trim(), reason: addReason || null }, { headers: authHeaders(token) })
      setAddCode('')
      setAddReason('')
      fetchData()
    } catch {}
    setAdding(false)
  }

  const removeFromWatchlist = async (id) => {
    const token = await ensureToken()
    await axios.delete(`${API}/watchlist/${id}`, { headers: authHeaders(token) }).catch(() => {})
    fetchData()
  }

  const unreadCount = alerts.filter(a => !a.is_read).length

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16, height: '100%', overflow: 'hidden' }}>

      {/* Tab bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="tab-strip" style={{ width: 'fit-content' }}>
          <button className={`tab-item ${tab === 'alerts' ? 'active' : ''}`} onClick={() => setTab('alerts')}>
            Alerts {unreadCount > 0 && <span style={{
              marginLeft: 4, padding: '1px 6px', borderRadius: 10,
              fontSize: 10, fontWeight: 600, background: '#ef4444', color: '#fff',
            }}>{unreadCount}</span>}
          </button>
          <button className={`tab-item ${tab === 'watchlist' ? 'active' : ''}`} onClick={() => setTab('watchlist')}>
            Watchlist ({watchlist.filter(w => w.is_active).length})
          </button>
        </div>
        {tab === 'alerts' && unreadCount > 0 && (
          <button className="btn btn-white" style={{ fontSize: 11, padding: '4px 10px' }} onClick={markAllRead}>
            Mark all read
          </button>
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
            <div className="spinner" />
          </div>
        ) : tab === 'alerts' ? (
          /* ── Alerts list ── */
          alerts.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', color: '#aaa', padding: 40 }}>
              <div style={{ fontSize: 32, marginBottom: 8 }}>🔔</div>
              <div style={{ fontSize: 13 }}>No alerts yet</div>
              <div style={{ fontSize: 11, marginTop: 4 }}>Add persons to the watchlist to get notified when they're detected</div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {alerts.map(alert => {
                const s = SEVERITY_STYLE[alert.severity] || SEVERITY_STYLE.info
                return (
                  <div key={alert.id} className="card" style={{
                    padding: '10px 14px', display: 'flex', gap: 10, alignItems: 'flex-start',
                    borderLeft: `3px solid ${s.border}`,
                    background: alert.is_read ? '#fff' : s.bg,
                    opacity: alert.is_read ? 0.7 : 1,
                  }}>
                    <div style={{
                      width: 24, height: 24, borderRadius: 6,
                      background: s.color + '18', color: s.color,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 11, fontWeight: 700, flexShrink: 0,
                    }}>
                      {s.icon}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 12, fontWeight: 500, color: '#111' }}>{alert.title}</span>
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
                          <span style={{
                            fontSize: 9, padding: '1px 5px', borderRadius: 4,
                            background: s.color + '18', color: s.color, fontWeight: 600,
                          }}>
                            {TYPE_LABELS[alert.alert_type] || alert.alert_type}
                          </span>
                          <span style={{ fontSize: 10, color: '#aaa' }}>{fmtTime(alert.created_at)}</span>
                        </div>
                      </div>
                      {alert.message && (
                        <div style={{ fontSize: 11, color: '#666', marginTop: 3 }}>{alert.message}</div>
                      )}
                      <div style={{ display: 'flex', gap: 8, marginTop: 4, fontSize: 10, color: '#aaa' }}>
                        {alert.unique_code && <span>Person: <b style={{ color: '#111' }}>{alert.unique_code}</b></span>}
                        {alert.camera_id && <span>Camera: {alert.camera_id}</span>}
                        {!alert.is_read && (
                          <button onClick={() => markRead(alert.id)} style={{
                            marginLeft: 'auto', border: 'none', background: 'none',
                            cursor: 'pointer', color: '#3b82f6', fontSize: 10, fontFamily: 'inherit',
                          }}>Mark read</button>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )
        ) : (
          /* ── Watchlist tab ── */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Add form */}
            <div className="card" style={{ padding: '14px 16px' }}>
              <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 10 }}>Add to Watchlist</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  type="text" placeholder="SDT-XXXX" value={addCode}
                  onChange={e => setAddCode(e.target.value)}
                  style={{
                    flex: 1, padding: '6px 10px', borderRadius: 6,
                    border: '1px solid #e0e0e0', fontSize: 12, fontFamily: 'monospace',
                  }}
                />
                <input
                  type="text" placeholder="Reason (optional)" value={addReason}
                  onChange={e => setAddReason(e.target.value)}
                  style={{
                    flex: 2, padding: '6px 10px', borderRadius: 6,
                    border: '1px solid #e0e0e0', fontSize: 12,
                  }}
                />
                <button className="btn btn-black" style={{ fontSize: 12, padding: '6px 14px' }}
                  onClick={addToWatchlist} disabled={!addCode.trim() || adding}>
                  {adding ? '...' : 'Add'}
                </button>
              </div>
            </div>

            {/* Watchlist entries */}
            {watchlist.filter(w => w.is_active).length === 0 ? (
              <div className="card" style={{ textAlign: 'center', color: '#aaa', padding: 30 }}>
                <div style={{ fontSize: 13 }}>Watchlist is empty</div>
                <div style={{ fontSize: 11, marginTop: 4 }}>Add a person code above to monitor</div>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 8 }}>
                {watchlist.filter(w => w.is_active).map(entry => (
                  <div key={entry.id} className="card" style={{ padding: '12px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 500, fontFamily: 'monospace', color: '#111' }}>
                        {entry.unique_code}
                      </div>
                      {entry.reason && <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>{entry.reason}</div>}
                      <div style={{ fontSize: 10, color: '#bbb', marginTop: 2 }}>Added {fmtTime(entry.created_at)}</div>
                    </div>
                    <button onClick={() => removeFromWatchlist(entry.id)} style={{
                      border: 'none', background: '#fee2e2', color: '#dc2626',
                      borderRadius: 6, padding: '4px 8px', cursor: 'pointer',
                      fontSize: 10, fontWeight: 500, fontFamily: 'inherit',
                    }}>Remove</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
