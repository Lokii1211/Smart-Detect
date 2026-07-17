import React, { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Locations() {
  const [locations, setLocations] = useState([])
  const [loading,   setLoading]   = useState(true)
  const [showForm,  setShowForm]  = useState(false)
  const [form,      setForm]      = useState({ id: '', name: '', type: 'other', address: '' })
  const [saving,    setSaving]    = useState(false)
  const [error,     setError]     = useState(null)

  const fetch = () => {
    axios.get(`${API}/locations`)
      .then(r => setLocations(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetch() }, [])

  const handleSave = async () => {
    if (!form.id || !form.name) { setError('ID and Name are required.'); return }
    setSaving(true); setError(null)
    try {
      await axios.post(`${API}/locations`, form, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token') || ''}` },
      })
      setShowForm(false)
      setForm({ id: '', name: '', type: 'other', address: '' })
      fetch()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create location.')
    } finally { setSaving(false) }
  }

  const TYPE_COLORS = { entrance: '#f0fdf4', exit: '#fef2f2', lobby: '#eff6ff', corridor: '#fefce8', parking: '#f5f3ff', office: '#ecfdf5', warehouse: '#fff7ed', outdoor: '#f0f9ff', other: '#f9fafb' }

  return (
    <div className="fade-in">
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 500 }}>{locations.length} locations registered</div>
          <div style={{ fontSize: 11, color: '#aaa', marginTop: 1 }}>Entrances, exits, lobbies, corridors, and zones</div>
        </div>
        <button className="btn btn-black" onClick={() => setShowForm(!showForm)}>
          {showForm ? 'Cancel' : '+ Add Location'}
        </button>
      </div>

      {/* Add form */}
      {showForm && (
        <div className="card fade-in" style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>New location</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 10 }}>
            <div>
              <label style={{ fontSize: 11, color: '#aaa', display: 'block', marginBottom: 4 }}>Location ID *</label>
              <input value={form.id} onChange={e => setForm(p => ({ ...p, id: e.target.value }))}
                placeholder="LOC-001" />
            </div>
            <div>
              <label style={{ fontSize: 11, color: '#aaa', display: 'block', marginBottom: 4 }}>Name *</label>
              <input value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                placeholder="City Campus" />
            </div>
            <div>
              <label style={{ fontSize: 11, color: '#aaa', display: 'block', marginBottom: 4 }}>Type</label>
              <select value={form.type} onChange={e => setForm(p => ({ ...p, type: e.target.value }))}>
                {['entrance', 'exit', 'lobby', 'corridor', 'parking', 'office', 'warehouse', 'outdoor', 'other'].map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 11, color: '#aaa', display: 'block', marginBottom: 4 }}>Address</label>
              <input value={form.address} onChange={e => setForm(p => ({ ...p, address: e.target.value }))}
                placeholder="Optional" />
            </div>
          </div>
          {error && <div style={{ fontSize: 11, color: '#dc2626', marginBottom: 8 }}>{error}</div>}
          <button className="btn btn-black" style={{ width: '100%' }} onClick={handleSave} disabled={saving}>
            {saving ? 'Saving…' : 'Save Location'}
          </button>
        </div>
      )}

      {/* Locations grid */}
      {loading ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
          {[1,2,3].map(i => (
            <div key={i} className="card" style={{ height: 90, background: '#f8f8f8' }} />
          ))}
        </div>
      ) : locations.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '40px 16px', color: '#bbb' }}>
          <div style={{ fontSize: 13, marginBottom: 4 }}>No locations yet</div>
          <div style={{ fontSize: 11 }}>Click "+ Add Location" to register your first location</div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
          {locations.map(loc => (
            <div key={loc.id} className="card" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'monospace', color: '#111' }}>{loc.id}</span>
                <span style={{
                  padding: '2px 8px', borderRadius: 6, fontSize: 10, fontWeight: 500,
                  background: TYPE_COLORS[loc.type] || '#f9fafb', color: '#555',
                }}>{loc.type}</span>
              </div>
              <div style={{ fontSize: 13, fontWeight: 500, color: '#111' }}>{loc.name}</div>
              {loc.address && <div style={{ fontSize: 11, color: '#aaa' }}>{loc.address}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
