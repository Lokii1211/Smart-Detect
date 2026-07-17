import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * RegisterPerson — Image upload + location selector to register a new person.
 *
 * Props:
 *   onRegistered {function(unique_code: string)} - called after successful registration
 */
export default function RegisterPerson({ onRegistered }) {
  const [locations, setLocations] = useState([])
  const [locationId, setLocationId] = useState('')
  const [zoneId, setZoneId] = useState('Entrance')
  const [personType, setPersonType] = useState('visitor')
  const [imageFile, setImageFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)
  const fileInputRef = useRef(null)

  // ── Fetch locations on mount ──────────────────────────────────────────────
  useEffect(() => {
    axios.get(`${API}/locations`)
      .then(res => {
        setLocations(res.data)
        if (res.data.length > 0) setLocationId(res.data[0].id)
      })
      .catch(() => {})
  }, [])

  // ── Image selection ────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImageFile(file)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(file)
  }

  // ── Submit ─────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!imageFile || !locationId) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const reader = new FileReader()
      reader.onload = async (ev) => {
        const base64 = ev.target.result.split(',')[1]   // strip data:image/...;base64, prefix
        try {
          const res = await axios.post(`${API}/register`, {
            base64_image: base64,
            location_id: locationId,
            zone_id: zoneId,
            person_type: personType,
          })
          setResult(res.data)
          if (onRegistered) onRegistered(res.data.unique_code)
        } catch (err) {
          setError(err.response?.data?.detail || 'Registration failed.')
        } finally {
          setLoading(false)
        }
      }
      reader.readAsDataURL(imageFile)
    } catch (err) {
      setError('Failed to read image file.')
      setLoading(false)
    }
  }

  // ── Copy to clipboard ──────────────────────────────────────────────────────
  const handleCopy = () => {
    if (!result?.unique_code) return
    navigator.clipboard.writeText(result.unique_code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const isNew = result?.is_new_registration

  return (
    <div className="space-y-5">
      {/* Upload area */}
      <div
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200
          ${preview
            ? 'border-metro-accent/50 bg-metro-accent/5'
            : 'border-metro-600 hover:border-metro-accent/50 hover:bg-metro-accent/5'
          }`}
      >
        {preview ? (
          <div className="flex flex-col items-center gap-3">
            <img src={preview} alt="Preview" className="h-40 max-w-full object-contain rounded-xl shadow-lg" />
            <p className="text-sm text-metro-accent font-medium">{imageFile?.name}</p>
            <p className="text-xs text-slate-500">Click to change image</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-slate-400">
            <svg className="w-14 h-14 text-metro-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 0 1 2.828 0L16 16m-2-2l1.586-1.586a2 2 0 0 1 2.828 0L20 14m-6-6h.01M6 20h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2z" />
            </svg>
            <div>
              <p className="font-semibold text-slate-300">Click to upload a face photo</p>
              <p className="text-sm mt-1">JPG, PNG, WEBP — clear frontal face preferred</p>
            </div>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          id="image-upload"
        />
      </div>

      {/* Location selector */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1.5" htmlFor="location-select">
          Location
        </label>
        <select
          id="location-select"
          value={locationId}
          onChange={e => setLocationId(e.target.value)}
          className="input-field"
          disabled={loading}
        >
          {locations.length === 0 && (
            <option value="">No locations available</option>
          )}
          {locations.map(l => (
            <option key={l.id} value={l.id}>{l.name} ({l.type})</option>
          ))}
        </select>
      </div>

      {/* Zone input */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1.5" htmlFor="zone-input">
          Zone / Area
        </label>
        <input
          id="zone-input"
          type="text"
          value={zoneId}
          onChange={e => setZoneId(e.target.value)}
          placeholder="e.g. Entrance, Gate A, Parking"
          className="input-field"
          disabled={loading}
        />
      </div>

      {/* Person type selector */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1.5">Person Type</label>
        <div className="flex gap-2">
          {['visitor', 'staff', 'unknown'].map(type => (
            <button
              key={type}
              onClick={() => setPersonType(type)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium capitalize border transition-all
                ${personType === type
                  ? 'bg-metro-accent text-white border-metro-accent shadow-lg shadow-blue-500/20'
                  : 'text-slate-400 border-metro-600 hover:border-metro-accent/50 hover:text-white'
                }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!imageFile || !locationId || loading}
        className="btn-primary w-full flex items-center justify-center gap-2 py-3"
        id="register-btn"
      >
        {loading ? (
          <>
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Registering…
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 1 1-8 0 4 4 0 0 1 8 0zM3 20a6 6 0 0 1 12 0v1H3v-1z" />
            </svg>
            Register Person
          </>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="glass-card p-4 border-red-500/30 animate-fade-in">
          <p className="text-metro-red text-sm flex items-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
            </svg>
            {error}
          </p>
        </div>
      )}

      {/* Success result */}
      {result && (
        <div className={`glass-card p-5 border animate-slide-up ${isNew ? 'border-metro-green/40' : 'border-metro-accent/40'}`}>
          <div className="flex items-center gap-2 mb-3">
            {isNew ? (
              <>
                <span className="badge-green text-sm">✓ New Registration</span>
                <span className="text-slate-400 text-xs">Person added to database</span>
              </>
            ) : (
              <>
                <span className="badge-blue text-sm">↩ Returning Person</span>
                <span className="text-slate-400 text-xs">Existing record matched</span>
              </>
            )}
          </div>

          <p className="text-xs text-slate-500 mb-1">Tracking Code</p>
          <div className="flex items-center gap-3">
            <code className={`text-2xl font-bold font-mono tracking-widest ${isNew ? 'text-metro-green' : 'text-metro-accent'}`}>
              {result.unique_code}
            </code>
            <button
              onClick={handleCopy}
              className={`btn-secondary flex items-center gap-1.5 text-sm py-1.5 px-3 ${copied ? 'text-metro-green' : ''}`}
              id="copy-code-btn"
            >
              {copied ? (
                <><svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg> Copied!</>
              ) : (
                <><svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M8 16H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v2m-6 12h8a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2z" />
                </svg> Copy</>
              )}
            </button>
          </div>
          <p className="text-slate-400 text-sm mt-2">{result.message}</p>
        </div>
      )}
    </div>
  )
}
