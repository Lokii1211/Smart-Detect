import React, { useEffect, useState, useCallback } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const TYPE_ICONS = {
  backpack:   { icon: '🎒', color: '#f59e0b' },
  handbag:    { icon: '👜', color: '#ec4899' },
  suitcase:   { icon: '🧳', color: '#8b5cf6' },
  car:        { icon: '🚗', color: '#3b82f6' },
  motorcycle: { icon: '🏍', color: '#ef4444' },
  truck:      { icon: '🚛', color: '#6b7280' },
  bus:        { icon: '🚌', color: '#10b981' },
}

function fmtTime(iso) {
  if (!iso) return '—'
  return new Date(iso + (iso.endsWith('Z') ? '' : 'Z'))
    .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function fmtDate(iso) {
  if (!iso) return '—'
  return new Date(iso + (iso.endsWith('Z') ? '' : 'Z'))
    .toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function ObjectFeed() {
  const [objects, setObjects] = useState([])
  const [stats, setStats] = useState(null)
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      const [objRes, statsRes] = await Promise.all([
        axios.get(`${API}/objects/recent`, { params: { limit: 100 } }),
        axios.get(`${API}/objects/stats`),
      ])
      setObjects(objRes.data)
      setStats(statsRes.data)
    } catch {
      // backend may not have objects yet
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const iv = setInterval(fetchData, 8000)
    return () => clearInterval(iv)
  }, [fetchData])

  const types = [...new Set(objects.map(o => o.object_type))]
  const filtered = filter === 'all' ? objects : objects.filter(o => o.object_type === filter)

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16, height: '100%', overflow: 'hidden' }}>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: 10, color: '#aaa', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.05em' }}>Total Today</div>
          <div style={{ fontSize: 22, fontWeight: 600, color: '#111', marginTop: 4 }}>
            {loading ? '—' : (stats?.total_today ?? 0)}
          </div>
        </div>
        {Object.entries(stats?.by_type ?? {}).map(([type, count]) => {
          const info = TYPE_ICONS[type] || { icon: '📦', color: '#6b7280' }
          return (
            <div key={type} className="card" style={{ padding: '14px 16px' }}>
              <div style={{ fontSize: 10, color: '#aaa', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.05em' }}>
                {info.icon} {type}
              </div>
              <div style={{ fontSize: 22, fontWeight: 600, color: info.color, marginTop: 4 }}>{count}</div>
            </div>
          )
        })}
      </div>

      {/* Filter tabs */}
      <div className="tab-strip" style={{ width: 'fit-content' }}>
        <button className={`tab-item ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>
          All ({objects.length})
        </button>
        {types.map(t => {
          const info = TYPE_ICONS[t] || { icon: '📦' }
          const count = objects.filter(o => o.object_type === t).length
          return (
            <button key={t} className={`tab-item ${filter === t ? 'active' : ''}`} onClick={() => setFilter(t)}>
              {info.icon} {t} ({count})
            </button>
          )
        })}
      </div>

      {/* Object list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
            <div className="spinner" />
          </div>
        ) : filtered.length === 0 ? (
          <div className="card" style={{ textAlign: 'center', color: '#aaa', padding: 40 }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>📦</div>
            <div style={{ fontSize: 13 }}>No object detections yet</div>
            <div style={{ fontSize: 11, marginTop: 4 }}>Objects like bags, vehicles will appear here when detected by cameras</div>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
            {filtered.map(obj => {
              const info = TYPE_ICONS[obj.object_type] || { icon: '📦', color: '#6b7280' }
              return (
                <div key={obj.id} className="card" style={{ padding: '12px 14px', display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: 8,
                    background: info.color + '18',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 18, flexShrink: 0,
                  }}>
                    {info.icon}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 13, fontWeight: 500, color: '#111', textTransform: 'capitalize' }}>
                        {obj.object_type}
                      </span>
                      <span style={{
                        fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 4,
                        background: info.color + '18', color: info.color,
                      }}>
                        {Math.round(obj.confidence * 100)}%
                      </span>
                    </div>
                    <div style={{ fontSize: 11, color: '#888', marginTop: 3 }}>
                      {obj.camera_id} · {obj.zone_id || 'main'}
                    </div>
                    <div style={{ fontSize: 10, color: '#bbb', marginTop: 2 }}>
                      {fmtDate(obj.detected_at)} {fmtTime(obj.detected_at)}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
