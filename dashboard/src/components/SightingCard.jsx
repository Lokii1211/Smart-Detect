import React from 'react'

/**
 * SightingCard — A single timeline entry for one person sighting.
 *
 * Props:
 *   sighting   {object}  { station_name, camera_id, seen_at, confidence, frame_snapshot_path }
 *   isFirst    {boolean} style as entry event
 *   isLast     {boolean} style as last-seen event
 *   index      {number}  position in trail
 */
export default function SightingCard({ sighting, isFirst = false, isLast = false, index }) {
  const { station_name, camera_id, seen_at, confidence } = sighting

  // ─── Helpers ──────────────────────────────────────────────────────────────
  const formatTime = (iso) => {
    const d = new Date(iso)
    return {
      date: d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }),
      time: d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true }),
    }
  }

  const confPercent = Math.round(confidence * 100)
  const confColor = confPercent >= 80 ? 'text-metro-green' : confPercent >= 60 ? 'text-metro-amber' : 'text-metro-red'
  const { date, time } = formatTime(seen_at)

  const dotColor = isFirst
    ? 'bg-metro-green border-metro-green/40'
    : isLast
    ? 'bg-metro-accent border-metro-accent/40'
    : 'bg-metro-500 border-metro-500/40'

  const labelBadge = isFirst
    ? <span className="badge-green ml-2">ENTRY</span>
    : isLast
    ? <span className="badge-blue ml-2">LAST SEEN</span>
    : null

  return (
    <div className="flex gap-4 animate-slide-up" style={{ animationDelay: `${index * 60}ms` }}>
      {/* Timeline dot + line */}
      <div className="flex flex-col items-center">
        <div className={`w-3.5 h-3.5 rounded-full border-2 mt-1.5 flex-shrink-0 ${dotColor} shadow-lg`} />
        {!isLast && <div className="w-px flex-1 bg-metro-600/50 mt-1" />}
      </div>

      {/* Card */}
      <div className={`glass-card p-4 mb-4 flex-1 transition-all hover:border-metro-accent/30 hover:shadow-lg hover:shadow-metro-accent/5
        ${isFirst ? 'border-l-2 border-l-metro-green' : isLast ? 'border-l-2 border-l-metro-accent' : ''}`}
      >
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="flex items-center flex-wrap gap-1.5">
              {/* Station icon */}
              <svg className="w-4 h-4 text-metro-teal flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M17.657 16.657L13.414 20.9a1.998 1.998 0 0 1-2.827 0l-4.244-4.243a8 8 0 1 1 11.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 1 1-6 0 3 3 0 0 1 6 0z" />
              </svg>
              <span className="font-semibold text-white">{station_name}</span>
              {labelBadge}
            </div>

            <div className="flex items-center gap-4 mt-2 text-sm text-slate-400">
              {/* Camera */}
              <span className="flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M15 10l4.553-2.069A1 1 0 0 1 21 8.845v6.31a1 1 0 0 1-1.447.894L15 14M3 8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8z" />
                </svg>
                {camera_id}
              </span>

              {/* Date */}
              <span className="flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2z" />
                </svg>
                {date}
              </span>

              {/* Time */}
              <span className="font-mono">{time}</span>
            </div>
          </div>

          {/* Confidence badge */}
          <div className="flex flex-col items-end gap-1">
            <span className={`text-xl font-bold tabular-nums ${confColor}`}>{confPercent}%</span>
            <span className="text-xs text-slate-500">confidence</span>
          </div>
        </div>
      </div>
    </div>
  )
}
