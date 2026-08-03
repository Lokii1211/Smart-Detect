import React, { useState } from 'react'
import { login } from '../auth'

/*
 * Sign-in screen. Replaces the hardcoded auto-login that previously shipped
 * operator and admin passwords inside the client bundle.
 */
export default function Login({ onSuccess }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if (!username || !password) return
    setBusy(true); setError(null)
    try {
      await login(username, password)
      onSuccess?.()
    } catch (err) {
      setError(err.message || 'Sign-in failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center',
      justifyContent: 'center', background: '#f8fafc', padding: 20,
    }}>
      <form onSubmit={submit} style={{
        width: '100%', maxWidth: 360, background: '#fff', borderRadius: 14,
        padding: 28, border: '1px solid #e2e8f0',
        boxShadow: '0 4px 24px rgba(15,23,42,0.06)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8, background: '#0f172a',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#fff' }} />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#0f172a' }}>SmartDetect</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>Detection System</div>
          </div>
        </div>

        <div style={{ fontSize: 12, color: '#64748b', margin: '18px 0 14px', lineHeight: 1.5 }}>
          Sign in to access camera streams and person records.
        </div>

        <label style={{ fontSize: 11, color: '#64748b', display: 'block', marginBottom: 4 }}>
          Username
        </label>
        <input
          value={username} onChange={e => setUsername(e.target.value)}
          autoFocus autoComplete="username"
          style={inputStyle}
        />

        <label style={{ fontSize: 11, color: '#64748b', display: 'block', margin: '12px 0 4px' }}>
          Password
        </label>
        <input
          type="password" value={password} onChange={e => setPassword(e.target.value)}
          autoComplete="current-password"
          style={inputStyle}
        />

        {error && (
          <div style={{
            marginTop: 12, padding: '8px 10px', borderRadius: 8, fontSize: 12,
            background: '#fef2f2', border: '1px solid #fecaca', color: '#b91c1c',
          }}>{error}</div>
        )}

        <button type="submit" disabled={busy || !username || !password} style={{
          marginTop: 18, width: '100%', padding: '10px 0', borderRadius: 9,
          border: 'none', cursor: busy ? 'default' : 'pointer',
          background: busy ? '#475569' : '#0f172a', color: '#fff',
          fontSize: 13, fontWeight: 600, fontFamily: 'inherit',
          opacity: (!username || !password) ? 0.5 : 1,
        }}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>

        <div style={{ marginTop: 14, fontSize: 10, color: '#94a3b8', lineHeight: 1.6 }}>
          Credentials are set in the server's <code>.env</code>. Run
          {' '}<code>python scripts/generate_env.py</code> to create them.
        </div>
      </form>
    </div>
  )
}

const inputStyle = {
  width: '100%', padding: '9px 11px', borderRadius: 9, fontSize: 13,
  border: '1px solid #e2e8f0', outline: 'none', fontFamily: 'inherit',
  background: '#fafafa', boxSizing: 'border-box',
}
