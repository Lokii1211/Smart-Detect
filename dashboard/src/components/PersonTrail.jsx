import React, { useEffect, useState } from 'react'
import axios from 'axios'
import SightingCard from './SightingCard.jsx'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * PersonTrail — Vertical timeline of all camera sightings for a person.
 *
 * Props:
 *   unique_code {string}  The SDT-XXXX tracking code to display.
 */
export default function PersonTrail({ unique_code }) {
  const [trail, setTrail] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!unique_code) return

    const fetchTrail = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await axios.get(`${API}/person/${encodeURIComponent(unique_code)}/trail`)
        setTrail(res.data)
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to fetch trail. Is the backend running?')
        setTrail([])
      } finally {
        setLoading(false)
      }
    }

    fetchTrail()
  }, [unique_code])

  // ─── Loading ──────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4 animate-fade-in">
        <div className="w-12 h-12 border-4 border-metro-600 border-t-metro-accent rounded-full animate-spin" />
        <p className="text-slate-400 text-sm">Fetching movement trail…</p>
      </div>
    )
  }

  // ─── Error ────────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="glass-card p-6 border-red-500/30 animate-fade-in">
        <div className="flex items-start gap-3">
          <svg className="w-5 h-5 text-metro-red flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
          </svg>
          <p className="text-metro-red text-sm">{error}</p>
        </div>
      </div>
    )
  }

  // ─── Empty ────────────────────────────────────────────────────────────────
  if (trail.length === 0 && unique_code) {
    return (
      <div className="glass-card p-12 text-center animate-fade-in">
        <svg className="w-16 h-16 mx-auto text-metro-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9.172 16.172a4 4 0 0 1 5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
        </svg>
        <h3 className="text-lg font-semibold text-slate-300 mb-1">No Sightings Found</h3>
        <p className="text-slate-500 text-sm">
          No camera sightings recorded for <span className="font-mono text-slate-300">{unique_code}</span>.
        </p>
      </div>
    )
  }

  if (!unique_code) return null

  // ─── Summary strip ────────────────────────────────────────────────────────
  const firstSeen = new Date(trail[0]?.seen_at)
  const lastSeen  = new Date(trail[trail.length - 1]?.seen_at)
  const durationMs = lastSeen - firstSeen
  const durationMin = Math.round(durationMs / 60000)

  return (
    <div className="animate-fade-in">
      {/* Summary bar */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {[
          { label: 'Total Sightings', value: trail.length, color: 'text-metro-accent' },
          { label: 'Locations',       value: new Set(trail.map(s => s.location_id)).size, color: 'text-metro-teal' },
          { label: 'Journey Duration', value: durationMin < 1 ? '<1 min' : `${durationMin} min`, color: 'text-metro-amber' },
        ].map(({ label, value, color }) => (
          <div key={label} className="glass-card p-3 text-center">
            <p className={`text-2xl font-bold ${color}`}>{value}</p>
            <p className="text-xs text-slate-500 mt-0.5">{label}</p>
          </div>
        ))}
      </div>

      {/* Timeline */}
      <div>
        {trail.map((sighting, idx) => (
          <SightingCard
            key={idx}
            sighting={sighting}
            index={idx}
            isFirst={idx === 0}
            isLast={idx === trail.length - 1}
          />
        ))}
      </div>
    </div>
  )
}
