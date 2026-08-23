import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import Login from './pages/Login.jsx'
import { isAuthenticated, installAuthInterceptor, logout, primeStreamToken, getStreamToken } from './auth'
import Sidebar from './components/Sidebar.jsx'
import TopBar  from './components/TopBar.jsx'
import Dashboard   from './pages/Dashboard.jsx'
import PhotoSearch from './pages/PhotoSearch.jsx'
import LiveCamera  from './pages/LiveCamera.jsx'
import People      from './pages/People.jsx'
import Locations   from './pages/Locations.jsx'
import ObjectFeed  from './pages/ObjectFeed.jsx'
import Alerts      from './pages/Alerts.jsx'
import Settings    from './pages/Settings.jsx'

const PAGE_TITLES = {
  '/':          'Dashboard',
  '/search':    'Photo Search',
  '/live':      'Live Camera',
  '/people':    'People',
  '/locations': 'Locations',
  '/objects':   'Object Feed',
  '/alerts':    'Alerts',
  '/settings':  'Settings',
}

function AppShell() {
  const location = useLocation()
  const navigate  = useNavigate()
  const title = PAGE_TITLES[location.pathname] || 'SmartDetect'

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: '#f7f8fa' }}>
      <Sidebar activePath={location.pathname} onNavigate={navigate} />

      <div className="layout-body">
        <TopBar title={title} />

        <main className="layout-main">
          <Routes>
            <Route path="/"          element={<Dashboard />} />
            <Route path="/search"    element={<PhotoSearch />} />
            <Route path="/live"      element={<LiveCamera />} />
            <Route path="/people"    element={<People />} />
            <Route path="/locations" element={<Locations />} />
            <Route path="/objects"   element={<ObjectFeed />} />
            <Route path="/alerts"    element={<Alerts />} />
            <Route path="/settings"  element={<Settings />} />
            <Route path="*" element={
              <div className="card fade-in" style={{ maxWidth: 320 }}>
                <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 6 }}>404 — Page not found</div>
              </div>
            } />
          </Routes>
        </main>

      </div>
    </div>
  )
}

// Any 401 anywhere clears the session and returns to sign-in.
installAuthInterceptor(axios)

export default function App() {
  const [authed, setAuthed] = useState(isAuthenticated())

  useEffect(() => {
    const onExpired = () => setAuthed(false)
    window.addEventListener('sd-auth-expired', onExpired)
    return () => window.removeEventListener('sd-auth-expired', onExpired)
  }, [])

  // <img> tags cannot send Authorization headers, so mint the short-lived
  // stream token once per session before any media renders — then refresh
  // it well before the 60-min expiry. Without this, every MJPEG stream and
  // snapshot <img> 401s after an hour in a long-lived tab ("Reconnecting
  // in 3s…" loops). mediaUrl() reads the token synchronously from
  // sessionStorage, so a background refresh is all it takes.
  useEffect(() => {
    if (!authed) return
    primeStreamToken()
    const iv = setInterval(() => getStreamToken(true).catch(() => {}), 25 * 60 * 1000)
    return () => clearInterval(iv)
  }, [authed])

  if (!authed) return <Login onSuccess={() => setAuthed(true)} />

  return (
    <BrowserRouter>
      <AppShell onSignOut={logout} />
    </BrowserRouter>
  )
}
