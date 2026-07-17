import React, { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const NAV = [
  { path: '/',          label: 'Dashboard',    icon: '⊞', dot: null },
  { path: '/search',    label: 'Photo Search', icon: '⊕', dot: 'amber' },
  { path: '/live',      label: 'Live Camera',  icon: '▶', dot: 'live' },
  { path: '/people',    label: 'People',       icon: '👤', dot: null },
  { path: '/locations', label: 'Locations',    icon: '📍', dot: null },
  { path: '/objects',   label: 'Object Feed',  icon: '📦', dot: null },
  { path: '/alerts',    label: 'Alerts',       icon: '🔔', dot: 'alert' },
  { path: '/settings',  label: 'Settings',     icon: '⚙', dot: null },
]

function NavIcon({ path }) {
  const icons = {
    '/':          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>,
    '/search':    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><circle cx="12" cy="13" r="3" strokeWidth={1.8} /></svg>,
    '/live':      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15 10l4.553-2.069A1 1 0 0121 8.845v6.31a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" /></svg>,
    '/people':    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M17 20h5v-2a4 4 0 00-3-3.87M9 20H4v-2a4 4 0 013-3.87m6-1.13a4 4 0 10-4-6.93M16 7a4 4 0 11-8 0 4 4 0 018 0z" /></svg>,
    '/locations': <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
    '/objects':   <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" /></svg>,
    '/alerts':    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" /></svg>,
    '/settings':  <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
  }
  return <span style={{ width: 14, height: 14, display: 'flex', flexShrink: 0 }}>{icons[path] || null}</span>
}

export default function Sidebar({ activePath, onNavigate }) {
  const [camStatus, setCamStatus] = useState(null)
  const [unreadAlerts, setUnreadAlerts] = useState(0)

  useEffect(() => {
    const fetchStatus = () => {
      axios.get(`${API}/camera/status`).then(r => setCamStatus(r.data)).catch(() => {})
      axios.get(`${API}/alerts/unread-count`).then(r => setUnreadAlerts(r.data?.count || 0)).catch(() => {})
    }
    fetchStatus()
    const iv = setInterval(fetchStatus, 5000)
    return () => clearInterval(iv)
  }, [])

  const isLive = camStatus?.connected === true

  return (
    <aside style={{
      width: 200, height: '100vh', background: '#fff',
      borderRight: '0.5px solid #e8e8e8',
      display: 'flex', flexDirection: 'column',
      flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{ padding: '18px 18px 14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 28, height: 28, background: '#111', borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>
            <div style={{ width: 8, height: 8, background: '#fff', borderRadius: '50%' }} />
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#111', lineHeight: 1.2 }}>SmartDetect</div>
            <div style={{ fontSize: 10, color: '#aaa', lineHeight: 1.2 }}>Detection System</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '4px 12px', overflowY: 'auto' }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: '#bbb', letterSpacing: '0.08em', textTransform: 'uppercase', padding: '8px 8px 6px' }}>
          Main Menu
        </div>
        {NAV.map(item => {
          const active = activePath === item.path
          const liveNow = item.dot === 'live' && isLive
          return (
            <button
              key={item.path}
              id={`nav-${item.path.replace('/', '') || 'home'}`}
              onClick={() => onNavigate(item.path)}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                width: '100%', padding: '7px 10px', borderRadius: 8,
                border: 'none', cursor: 'pointer', marginBottom: 2,
                background: active ? '#f0f0f0' : 'transparent',
                color: active ? '#111' : '#555',
                fontFamily: 'inherit', fontSize: 13,
                fontWeight: active ? 500 : 400,
                textAlign: 'left',
                transition: 'background 0.12s',
              }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.background = '#f8f8f8' }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent' }}
            >
              <NavIcon path={item.path} />
              <span style={{ flex: 1 }}>{item.label}</span>
              {item.dot === 'amber' && (
                <span className="dot" style={{ width: 6, height: 6, background: '#f59e0b' }} />
              )}
              {item.dot === 'alert' && unreadAlerts > 0 && (
                <span style={{
                  minWidth: 16, height: 16, borderRadius: 8,
                  background: '#ef4444', color: '#fff',
                  fontSize: 9, fontWeight: 700,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  padding: '0 4px',
                }}>{unreadAlerts > 99 ? '99+' : unreadAlerts}</span>
              )}
              {item.dot === 'live' && (
                <span
                  className={`dot ${liveNow ? 'dot-green' : 'dot-gray'}`}
                  style={{ width: 6, height: 6 }}
                />
              )}
            </button>
          )
        })}
      </nav>

      {/* Bottom camera status */}
      <div style={{ borderTop: '0.5px solid #f0f0f0', padding: '14px 18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            className={`dot ${isLive ? 'dot-green pulse' : 'dot-gray'}`}
            style={{ width: 7, height: 7 }}
          />
          <span style={{ fontSize: 11, color: '#666' }}>
            {isLive
              ? `${camStatus?.active_tracks ?? 0} tracks live`
              : 'No camera active'}
          </span>
        </div>
      </div>
    </aside>
  )
}
