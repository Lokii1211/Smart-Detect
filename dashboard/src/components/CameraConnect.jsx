import React, { useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const CAMERA_OPTIONS = [
  {
    id: 'usb0',
    source: 0,
    label: 'USB Webcam',
    subtitle: 'Laptop or USB camera (index 0)',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M15 10l4.553-2.069A1 1 0 0 1 21 8.845v6.31a1 1 0 0 1-1.447.894L15 14M3 8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8z" />
      </svg>
    ),
  },
  {
    id: 'usb1',
    source: 1,
    label: 'Second Webcam',
    subtitle: 'USB camera (index 1)',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M3 9a2 2 0 0 1 2-2h.93a2 2 0 0 0 1.664-.89l.812-1.22A2 2 0 0 1 10.07 4h3.86a2 2 0 0 1 1.664.89l.812 1.22A2 2 0 0 0 18.07 7H19a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 13a3 3 0 1 1-6 0 3 3 0 0 1 6 0z" />
      </svg>
    ),
  },
  {
    id: 'rtsp',
    source: null,   // filled from input
    label: 'IP Camera',
    subtitle: 'RTSP stream URL',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M8.111 16.404a5.5 5.5 0 0 1 7.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
      </svg>
    ),
  },
  {
    id: 'file',
    source: null,   // filled from input
    label: 'Video File',
    subtitle: 'For testing with recorded footage',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1z" />
      </svg>
    ),
  },
]

export default function CameraConnect({ onStarted }) {
  const [selected, setSelected] = useState('usb0')
  const [rtspUrl, setRtspUrl]   = useState('')
  const [filePath, setFilePath] = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  const getSource = () => {
    if (selected === 'usb0') return 0
    if (selected === 'usb1') return 1
    if (selected === 'rtsp') return rtspUrl.trim()
    if (selected === 'file') return filePath.trim()
    return 0
  }

  const handleStart = async () => {
    const source = getSource()
    if ((selected === 'rtsp' || selected === 'file') && !source) {
      setError('Please enter a URL or file path.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API}/camera/start`, { camera_source: source })
      if (onStarted) onStarted(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to connect to camera. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto animate-fade-in">
      <div className="glass-card p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-metro-accent/20 border border-metro-accent/30
                          flex items-center justify-center mx-auto mb-4">
            <svg className="w-7 h-7 text-metro-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M15 10l4.553-2.069A1 1 0 0 1 21 8.845v6.31a1 1 0 0 1-1.447.894L15 14M3 8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8z" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-white mb-1">Connect a Camera</h2>
          <p className="text-sm text-slate-400">
            SmartDetect will automatically identify and track persons in real-time
          </p>
        </div>

        {/* Camera type selector */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          {CAMERA_OPTIONS.map(opt => (
            <button
              key={opt.id}
              id={`camera-opt-${opt.id}`}
              onClick={() => { setSelected(opt.id); setError(null) }}
              className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-150 text-center
                ${selected === opt.id
                  ? 'bg-metro-accent/10 border-metro-accent/50 text-metro-accent shadow-lg shadow-blue-500/10'
                  : 'border-metro-700/60 text-slate-400 hover:border-metro-accent/30 hover:text-slate-300'
                }`}
            >
              {opt.icon}
              <div>
                <div className="text-sm font-semibold">{opt.label}</div>
                <div className="text-xs text-slate-500 mt-0.5">{opt.subtitle}</div>
              </div>
            </button>
          ))}
        </div>

        {/* RTSP input */}
        {selected === 'rtsp' && (
          <div className="mb-4 animate-fade-in">
            <label className="block text-sm font-medium text-slate-300 mb-1.5">RTSP Stream URL</label>
            <input
              id="rtsp-url"
              type="text"
              value={rtspUrl}
              onChange={e => setRtspUrl(e.target.value)}
              placeholder="rtsp://192.168.1.x:554/stream"
              className="input-field font-mono text-sm"
              disabled={loading}
            />
          </div>
        )}

        {/* File path input */}
        {selected === 'file' && (
          <div className="mb-4 animate-fade-in">
            <label className="block text-sm font-medium text-slate-300 mb-1.5">Video File Path</label>
            <input
              id="file-path"
              type="text"
              value={filePath}
              onChange={e => setFilePath(e.target.value)}
              placeholder="C:\videos\test.mp4 or /home/user/test.mp4"
              className="input-field font-mono text-sm"
              disabled={loading}
            />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="glass-card p-3 border-red-500/30 mb-4 animate-fade-in">
            <p className="text-metro-red text-sm flex items-center gap-2">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
              </svg>
              {error}
            </p>
          </div>
        )}

        {/* Start button */}
        <button
          id="start-camera-btn"
          onClick={handleStart}
          disabled={loading}
          className="btn-primary w-full py-3.5 text-base flex items-center justify-center gap-3"
        >
          {loading ? (
            <>
              <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Connecting…
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M14.752 11.168l-3.197-2.132A1 1 0 0 0 10 9.87v4.263a1 1 0 0 0 1.555.832l3.197-2.132a1 1 0 0 0 0-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />
              </svg>
              Start Live Detection
            </>
          )}
        </button>

        {/* Info */}
        <p className="text-xs text-slate-600 text-center mt-4">
          SmartDetect automatically identifies persons using face, dress color, and body structure.
          Unknown persons are registered instantly with an SDT-XXXX ID.
        </p>
      </div>
    </div>
  )
}
