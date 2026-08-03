/*
 * dashboard/src/auth.js
 * ─────────────────────
 * Single source of truth for dashboard authentication.
 *
 * SECURITY: hardcoded operator/admin credentials were previously embedded in
 * four page components and auto-submitted on mount. Anyone who opened
 * DevTools — or read the bundled JS, which is served to every visitor — had
 * working credentials for a live biometric system. They are gone; the user
 * now logs in and the token is held in sessionStorage.
 *
 * sessionStorage (not localStorage): the token dies with the tab, limiting
 * the window on a shared or unattended machine. It remains readable by any
 * script on the origin, so this is XSS-exposed by design — the real fix is
 * an httpOnly cookie, which needs a same-site deployment and CSRF handling.
 * Documented in docs/DEPLOYMENT_TLS.md as a known gap.
 */

export const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const TOKEN_KEY = 'sd_token'
const ROLE_KEY = 'sd_role'

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY)
}

export function getRole() {
  return sessionStorage.getItem(ROLE_KEY)
}

export function isAuthenticated() {
  return Boolean(getToken())
}

/*
 * Many components call bare axios without passing headers. Setting the
 * default here means every request is authenticated once signed in, instead
 * of each call site having to remember — which is how routes ended up
 * effectively public in the first place.
 */
let _axios = null
export function bindAxios(axiosInstance) {
  _axios = axiosInstance
  applyAuthHeader()
}

function applyAuthHeader() {
  if (!_axios) return
  const t = getToken()
  if (t) _axios.defaults.headers.common['Authorization'] = `Bearer ${t}`
  else delete _axios.defaults.headers.common['Authorization']
}

export function setSession(token, role) {
  sessionStorage.setItem(TOKEN_KEY, token)
  if (role) sessionStorage.setItem(ROLE_KEY, role)
  applyAuthHeader()
}

export function clearSession() {
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(ROLE_KEY)
  sessionStorage.removeItem('sd_stream_token')
  applyAuthHeader()
}

/** Authorization header for axios/fetch. Empty when signed out. */
export function authHeaders() {
  const t = getToken()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

export async function login(username, password) {
  const res = await fetch(`${API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Sign-in failed (${res.status})`)
  }
  const data = await res.json()
  setSession(data.access_token, data.role)
  return data
}

export function logout() {
  clearSession()
  window.location.reload()
}

/*
 * MJPEG streams and snapshot <img> tags cannot send an Authorization header,
 * so the backend also accepts ?token= carrying a short-lived, 'stream'-scoped
 * token that the JSON API rejects. Cached for the session and refreshed on
 * demand.
 */
let _streamTokenPromise = null

export async function getStreamToken(force = false) {
  const cached = sessionStorage.getItem('sd_stream_token')
  if (cached && !force) return cached
  if (_streamTokenPromise && !force) return _streamTokenPromise

  _streamTokenPromise = fetch(`${API}/auth/stream-token`, {
    method: 'POST',
    headers: authHeaders(),
  })
    .then(r => {
      if (!r.ok) throw new Error(`stream token failed (${r.status})`)
      return r.json()
    })
    .then(d => {
      sessionStorage.setItem('sd_stream_token', d.stream_token)
      _streamTokenPromise = null
      return d.stream_token
    })
    .catch(e => {
      _streamTokenPromise = null
      throw e
    })
  return _streamTokenPromise
}

/** Append a stream token to a media URL (MJPEG stream or snapshot image). */
export function withStreamToken(url, streamToken) {
  if (!streamToken) return url
  return url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(streamToken)
}

/**
 * Install a global 401 handler: any expired/invalid token drops the session
 * and returns the user to the sign-in screen instead of leaving pages stuck
 * on silent failures.
 */
export function installAuthInterceptor(axios) {
  bindAxios(axios)
  axios.interceptors.response.use(
    r => r,
    err => {
      if (err?.response?.status === 401) {
        clearSession()
        window.dispatchEvent(new CustomEvent('sd-auth-expired'))
      }
      return Promise.reject(err)
    },
  )
}


/*
 * URL for an authenticated media resource (snapshot image or MJPEG stream).
 * <img> cannot send headers, so the short-lived stream token rides in the
 * query string. Call primeStreamToken() once after sign-in; mediaUrl() is
 * then synchronous, which keeps it usable directly in JSX src attributes.
 */
export function primeStreamToken() {
  return getStreamToken().catch(() => null)
}

export function mediaUrl(pathOrUrl) {
  const base = pathOrUrl.startsWith('http') ? pathOrUrl : `${API}/${pathOrUrl.replace(/^\//, '')}`
  const t = sessionStorage.getItem('sd_stream_token')
  return t ? base + (base.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(t) : base
}
