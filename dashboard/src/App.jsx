import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom'
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

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}
