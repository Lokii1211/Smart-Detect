import React from 'react'

export default function TopBar({ title }) {
  return (
    <header style={{
      height: 52, background: '#fff',
      borderBottom: '0.5px solid #e8e8e8',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 24px', flexShrink: 0,
    }}>
      {/* Page title */}
      <span style={{ fontSize: 14, fontWeight: 500, color: '#111' }}>{title}</span>

      {/* Right side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {/* System online badge */}
        <span className="badge badge-green" style={{ fontSize: 11 }}>
          <span className="dot dot-green" style={{ width: 5, height: 5 }} />
          System online
        </span>

        {/* Operator avatar */}
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: '#111', color: '#fff',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 10, fontWeight: 600, userSelect: 'none',
        }}>
          OP
        </div>
      </div>
    </header>
  )
}
