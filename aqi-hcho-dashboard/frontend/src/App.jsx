import { useEffect, useState } from 'react'
import axios from 'axios'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { Wind, FlaskConical, MapPin, BrainCircuit, Upload, Activity, Leaf, RefreshCw } from 'lucide-react'

const API = 'http://127.0.0.1:8000'
const initialForm = { pm25: 42, pm10: 76, no2: 24, temperature: 29, humidity: 58, hcho: 0.012 }

function Stat({ icon: Icon, label, value, sub, color }) {
  return <div className="stat card"><div className="stat-top"><span>{label}</span><span className="icon" style={{color}}><Icon size={20}/></span></div><div className="stat-value">{value}</div><div className="muted">{sub}</div></div>
}

export default function App() {
  const [summary, setSummary] = useState(null)
  const [forecast, setForecast] = useState([])
  const [hotspots, setHotspots] = useState([])
  const [form, setForm] = useState(initialForm)
  const [prediction, setPrediction] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function loadDashboard() {
    setError('')
    try {
      const [s, f, h] = await Promise.all([
        axios.get(`${API}/api/summary`), axios.get(`${API}/api/forecast`), axios.get(`${API}/api/hotspots`)
      ])
      setSummary(s.data); setForecast(f.data.forecast); setHotspots(h.data.hotspots)
    } catch {
      setError('Backend no connect')
    }
  }
  useEffect(() => { loadDashboard() }, [])

  async function predict(e) {
    e.preventDefault(); setLoading(true); setError(''); setPrediction(null)
    try { const res = await axios.post(`${API}/api/predict`, Object.fromEntries(Object.entries(form).map(([k,v]) => [k, Number(v)]))); setPrediction(res.data) }
    catch { setError('Prediction fail') }
    finally { setLoading(false) }
  }

  async function uploadCsv(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const data = new FormData(); data.append('file', file)
    setError(''); setUploadResult(null)
    try { const res = await axios.post(`${API}/api/upload-csv`, data); setUploadResult(res.data) }
    catch (err) { setError(err.response?.data?.detail || 'CSV no upload. Required') }
    e.target.value = ''
  }

  const setField = (key, value) => setForm(prev => ({...prev, [key]: value}))
  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Wind size={23}/></div><div><b>AQI Prediction</b><small>ENVIRONMENT INTELLIGENCE</small></div></div>
      <div className="nav-label">WORKSPACE</div>
      <a className="nav active" href="#overview"><Activity size={18}/> Overview</a>
      <a className="nav" href="#prediction"><BrainCircuit size={18}/> AQI Prediction</a>
      <a className="nav" href="#hotspots"><MapPin size={18}/> HCHO Hotspots</a>
      <a className="nav" href="#upload"><Upload size={18}/> Dataset Upload</a>
      <div className="sidebar-bottom"><div className="pulse"></div><div><b>AI Engine</b><small>Demo model ready</small></div></div>
    </aside>

    <main className="main">
      <header className="header"><div><div className="eyebrow">ENVIRONMENT MONITORING</div><h1>Air Quality Dashboard</h1><p className="muted">Surface AQI prediction & HCHO Hotspot Detection</p></div><button className="refresh" onClick={loadDashboard}><RefreshCw size={16}/> Refresh</button></header>
      <div className="notice"><span></span><div><b></b><p></p></div></div>
      {error && <div className="error">{error}</div>}

      <section id="overview" className="stats-grid">
        <Stat icon={Wind} label="Surface AQI" value={summary?.current_aqi ?? '—'} sub="Moderate · demo reading" color="#0d9488"/>
        <Stat icon={FlaskConical} label="HCHO Concentration" value={summary ? `${summary.hcho_ppm} ppm` : '—'} sub="Illustrative demo reading" color="#d97706"/>
        <Stat icon={MapPin} label="Detected Hotspots" value={summary?.hotspots ?? '—'} sub="Fictional demo zones" color="#6366f1"/>
        <Stat icon={BrainCircuit} label="Model MAE" value={summary ? summary.model_mae_demo : '—'} sub="Synthetic test data only" color="#0891b2"/>
      </section>

      <section className="content-grid">
        <div className="card chart-card">
          <div className="section-heading"><div><h2>AQI Forecast</h2><p className="muted">Illustrative next 24-hour trend</p></div><span className="pill">DEMO FORECAST</span></div>
          <div className="chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={forecast} margin={{top: 10, right: 8, left: -18, bottom: 0}}><defs><linearGradient id="aqiFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#0d9488" stopOpacity={0.25}/><stop offset="100%" stopColor="#0d9488" stopOpacity={0.01}/></linearGradient></defs><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf2"/><XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill:'#718096',fontSize:12}}/><YAxis axisLine={false} tickLine={false} tick={{fill:'#718096',fontSize:12}}/><Tooltip/><Area type="monotone" dataKey="aqi" stroke="#0d9488" strokeWidth={3} fill="url(#aqiFill)" /></AreaChart></ResponsiveContainer></div>
          <div className="chart-foot"><span><i className="legend-dot"></i> Predicted AQI</span><span className="muted">Replace with real time-series data</span></div>
        </div>
        <div className="card model-card" id="prediction">
          <div className="section-heading"><div><h2>Predict AQI</h2><p className="muted">Enter environmental readings</p></div><BrainCircuit size={22} color="#0d9488"/></div>
          <form onSubmit={predict}>
            <div className="input-grid">
              {[['pm25','PM2.5 (µg/m³)'],['pm10','PM10 (µg/m³)'],['no2','NO₂ (µg/m³)'],['temperature','Temperature (°C)'],['humidity','Humidity (%)'],['hcho','HCHO (ppm)']].map(([key,label])=><label key={key}>{label}<input type="number" step="any" value={form[key]} onChange={e=>setField(key,e.target.value)} required min="0"/></label>)}
            </div>
            <button className="primary" disabled={loading}>{loading ? 'Predicting…' : 'Run AI Prediction'}</button>
          </form>
          {prediction && <div className="prediction-result"><div><small>Predicted AQI</small><strong>{prediction.predicted_aqi}</strong><span>{prediction.category}</span></div><div><small>HCHO band</small><b>{prediction.hcho_risk}</b></div><p>{prediction.note}</p></div>}
        </div>
      </section>

      <section id="hotspots" className="card table-card">
        <div className="section-heading"><div><h2>HCHO Hotspot Detection</h2><p className="muted">Zones ranked by illustrative concentration</p></div><span className="pill amber">FORMALDEHYDE</span></div>
        <div className="table-wrap"><table><thead><tr><th>Zone</th><th>HCHO (ppm)</th><th>Demo risk band</th><th>Status</th></tr></thead><tbody>{hotspots.map(h=><tr key={h.id}><td><span className="zone"><MapPin size={16}/>{h.location}</span></td><td>{h.hcho.toFixed(3)}</td><td>{h.risk}</td><td><span className={`status ${h.risk.startsWith('High')?'high':h.risk.startsWith('Medium')?'medium':'low'}`}>{h.risk.split(' ')[0]}</span></td></tr>)}</tbody></table></div>
        <p className="table-note">Hotspot bands are illustrative software placeholders, not official safety thresholds.</p>
      </section>

      <section id="upload" className="card upload-card">
        <div className="upload-icon"><Upload size={22}/></div><div className="upload-copy"><h2>Upload your dataset</h2><p className="muted">CSV must include: pm25, pm10, no2, temperature, humidity, hcho</p></div><label className="upload-button">Choose CSV<input type="file" accept=".csv,text/csv" onChange={uploadCsv}/></label>
        {uploadResult && <div className="upload-result"><b>Processed {uploadResult.rows} rows</b><p>{uploadResult.note}</p><pre>{JSON.stringify(uploadResult.preview.slice(0,3), null, 2)}</pre></div>}
      </section>
      <footer>AirScope AI · Academic prototype · Validate with real datasets before practical use</footer>
    </main>
  </div>
}
