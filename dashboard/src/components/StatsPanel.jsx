import React, { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function StatCard({ id, icon, label, value, color = 'text-metro-accent', loading, dot }) {
  return (
    <div id={id} className="glass-card p-5 flex items-center gap-4">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center bg-metro-700/60 ${color}`}>
        {icon}
      </div>
      <div>
        {loading ? (
          <div className="h-8 w-16 bg-metro-700 rounded animate-pulse mb-1" />
        ) : (
          <div className="flex items-center gap-2">
            <p className={`text-3xl font-bold tabular-nums ${color}`}>{value}</p>
            {dot && <span className={`w-2 h-2 rounded-full ${dot}`} />}
          </div>
        )}
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</p>
      </div>
    </div>
  )
}

export default function StatsPanel() {
  const [persons,   setPersons]   = useState(0)
  const [camStatus, setCamStatus] = useState(null)
  const [loading,   setLoading]   = useState(true)

  const fetchStats = async () => {
    try {
      const [camRes, personsRes] = await Promise.allSettled([
        axios.get(`${API}/camera/status`),
        axios.get(`${API}/persons`),
      ])
      if (camRes.status === 'fulfilled') setCamStatus(camRes.value.data)
      if (personsRes.status === 'fulfilled') setPersons(personsRes.value.data.length ?? 0)
    } catch {
      // Silently fail — stats are non-critical
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
    const iv = setInterval(fetchStats, 5_000)
    return () => clearInterval(iv)
  }, [])

  const iconPerson = (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M17 20h5v-2a3 3 0 0 0-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 0 1 5.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 0 1 9.288 0M15 7a3 3 0 1 1-6 0 3 3 0 0 1 6 0z" />
    </svg>
  )

  const iconCamera = (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M15 10l4.553-2.069A1 1 0 0 1 21 8.845v6.31a1 1 0 0 1-1.447.894L15 14M3 8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8z" />
    </svg>
  )

  const iconToday = (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M9 19v-6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2zm0 0V9a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v10m-6 0a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2m0 0V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-2a2 2 0 0 1-2-2z" />
    </svg>
  )

  const iconShield = (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0 1 12 2.944a11.955 11.955 0 0 1-8.618 3.04A12.02 12.02 0 0 0 3 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  )

  const isConnected = camStatus?.connected === true
  const activeCams  = isConnected ? 1 : 0
  const personsToday = camStatus?.persons_detected_today ?? '—'

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard id="stat-persons"    icon={iconPerson} label="Registered Persons" value={persons}      color="text-metro-accent"  loading={loading} />
      <StatCard id="stat-cameras"    icon={iconCamera} label="Active Cameras"     value={activeCams}   color="text-metro-teal"    loading={loading}
        dot={isConnected ? 'bg-metro-green animate-pulse' : undefined} />
      <StatCard id="stat-today"      icon={iconToday}  label="Persons Today"      value={personsToday} color="text-metro-amber"    loading={loading} />
      <StatCard id="stat-status"     icon={iconShield} label="System Status"      value="Online"       color="text-metro-green"   loading={false} />
    </div>
  )
}
