import React from 'react'

/**
 * SearchBar — Accepts a person tracking code and triggers a search.
 *
 * Props:
 *   value        {string}   controlled input value
 *   onChange     {function} called with new string value
 *   onSearch     {function} called when user submits (button click or Enter)
 *   placeholder  {string}
 *   loading      {boolean}  disables input + button during fetch
 */
export default function SearchBar({ value, onChange, onSearch, placeholder = 'Search SDT-XXXX…', loading = false }) {
  const handleKey = (e) => {
    if (e.key === 'Enter') onSearch()
  }

  return (
    <div className="flex gap-3 w-full">
      <div className="relative flex-1">
        {/* Search icon */}
        <svg
          className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none"
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
        </svg>
        <input
          type="text"
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={handleKey}
          placeholder={placeholder}
          disabled={loading}
          className="input-field pl-10 font-mono tracking-wide"
          aria-label="Search person by tracking code"
          id="search-input"
        />
      </div>

      <button
        onClick={onSearch}
        disabled={loading || !value.trim()}
        className="btn-primary flex items-center gap-2 min-w-[110px] justify-center"
        id="search-btn"
      >
        {loading ? (
          <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        ) : (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
          </svg>
        )}
        {loading ? 'Searching…' : 'Search'}
      </button>
    </div>
  )
}
