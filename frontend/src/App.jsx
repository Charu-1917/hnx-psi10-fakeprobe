import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import './App.css';
import demoVideo from './demo/video.json';
import demoImage from './demo/image.json';
import demoAudio from './demo/audio.json';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const BUNDLED = { video: demoVideo, image: demoImage, audio: demoAudio };

const LABELS = {
  AI_GENERATED: ['⚠ AI-GENERATED / EDITED', 'bad'],
  POSSIBLY_AI: ['❓ POSSIBLY AI — review', 'warn'],
  LIKELY_REAL: ['✅ LOOKS REAL', 'ok'],
  NOT_ENOUGH_INFO: ['➖ NOT ENOUGH INFORMATION', 'warn'],
};
const DOT = { bad: '🔴', warn: '🟠', ok: '🟢' };
const COLOR = { bad: '#ff5d73', warn: '#ffb84d', ok: '#3ddc97' };
const CHIPS = [
  'Summarise the verdict',
  'Why is the peak at the start?',
  'When are the suspicious intervals?',
  'Do audio and lip-sync agree?',
  'Why is this UNCERTAIN?',
  'How reliable is this?',
];

const kindOf = (f) => (f.type.startsWith('image') ? 'image' : f.type.startsWith('audio') ? 'audio' : 'video');
const areaColor = (p) => (p >= 60 ? '#ff5d73' : p >= 45 ? '#ffb84d' : '#ffd966');
const fmt = (n, d = 2) => (typeof n === 'number' ? n.toFixed(d) : '—');

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let detail = null;
    try { detail = (await r.json()).detail; } catch { /* not json */ }
    const err = new Error((detail && detail.message) || (typeof detail === 'string' ? detail : `HTTP ${r.status}`));
    err.status = r.status;
    err.runDemo = r.status === 503;
    throw err;
  }
  return r.json();
}

function Badge({ r }) {
  if (!r) return null;
  if (r.sample_data) return <span className="badge sample">Sample result (cached demo) · synthetic sample data, not a live analysis</span>;
  if (r.cached_real_run) return <span className="badge cached">Sample result (cached demo) · cached real run, not a live analysis</span>;
  return <span className="badge live">Live analysis · {r.engine}</span>;
}

export default function App() {
  const [engine, setEngine] = useState('demo');
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [kind, setKind] = useState('video');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [offerDemo, setOfferDemo] = useState(false);
  const [sel, setSel] = useState(0); // index into frame_regions
  const [showAll, setShowAll] = useState(false);
  const [highlight, setHighlight] = useState(null); // area number
  const [chat, setChat] = useState([{ who: 'a', text: "Hi! I answer only from the result above. I call small tools (summary, frame, intervals, modalities, audit) and explain in plain words. Try a suggestion." }]);
  const [q, setQ] = useState('');
  const [audioBuf, setAudioBuf] = useState(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const audioRef = useRef(null);
  const waveRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    api('/api/v1/health').then(setHealth).catch(() => setHealth({ offline: true }));
  }, []);

  const loadDemo = useCallback(async (type) => {
    setError(null); setOfferDemo(false); setBusy(true);
    let r;
    try { r = await api(`/api/v1/demo/${type}`); } catch { r = BUNDLED[type]; } // API down: bundled copy
    setKind(type); setResult(r); setSel(0); setShowAll(false); setHighlight(null); setBusy(false);
  }, []);

  const onFile = async (f) => {
    if (!f) return;
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    const k = kindOf(f);
    setFile(f); setKind(k); setFileUrl(URL.createObjectURL(f)); setResult(null); setAudioBuf(null);
    imgRef.current = null;
    if (k === 'image') {
      const im = new Image();
      im.onload = () => { imgRef.current = im; setSel((s) => s); };
      im.src = URL.createObjectURL(f);
    }
    if (k !== 'image') {
      try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        setAudioBuf(await new Ctx().decodeAudioData(await f.arrayBuffer()));
      } catch { setAudioBuf(null); }
    }
  };

  const run = async () => {
    if (!file) { loadDemo(kind); return; }
    setBusy(true); setError(null); setOfferDemo(false);
    try {
      const fd = new FormData(); fd.append('file', file);
      const r = await api(`/api/v1/analyze?engine=${engine}`, { method: 'POST', body: fd });
      setResult(r); setSel(0); setShowAll(false); setHighlight(null);
    } catch (e) {
      if (e.message === 'Failed to fetch') { setError('The API is not running. Start it with: uvicorn api:app --port 8000'); setOfferDemo(true); }
      else { setError(e.message); setOfferDemo(!!e.runDemo || engine !== 'demo'); }
    } finally { setBusy(false); }
  };

  const regions = result?.frame_regions || [];
  const withBoxes = useMemo(() => regions.map((f, i) => ({ ...f, i })).filter((f) => f.regions.length), [regions]);
  const fs = result?.frame_size;
  const usingUpload = !!file && !result?.sample_data && !result?.demo;
  const W = fs?.w || 640, H = fs?.h || 360;

  // ---- Step 3 canvas
  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c || !result) return;
    const x = c.getContext('2d');
    c.width = W; c.height = H;
    const bg = () => {
      x.fillStyle = '#0a0e15'; x.fillRect(0, 0, W, H);
      let drawn = false;
      if (kind === 'image' && imgRef.current && usingUpload) { x.drawImage(imgRef.current, 0, 0, W, H); drawn = true; }
      const v = videoRef.current;
      if (kind === 'video' && usingUpload && v && v.videoWidth && v.readyState > 1) { x.drawImage(v, 0, 0, W, H); drawn = true; }
      if (!drawn) {
        x.fillStyle = '#9fb0cc'; x.font = `${Math.max(14, W / 50)}px sans-serif`;
        x.fillText(usingUpload ? 'Frame loading…' : 'Original frame not available for this cached result — boxes shown on a blank frame.', 12, 24);
      }
    };
    const box = (r, n, col, label) => {
      const lw = Math.max(2, W / 300);
      x.strokeStyle = col; x.lineWidth = lw; x.fillStyle = col + '2e';
      x.fillRect(r.x, r.y, r.w, r.h); x.strokeRect(r.x, r.y, r.w, r.h);
      const fsz = Math.max(12, W / 55);
      x.font = `bold ${fsz}px sans-serif`;
      const t = label;
      const tw = x.measureText(t).width + 10;
      const ty = r.y - fsz - 6 < 0 ? r.y + 2 : r.y - fsz - 6;
      x.fillStyle = col; x.fillRect(r.x, ty, tw, fsz + 6);
      x.fillStyle = '#000'; x.fillText(t, r.x + 5, ty + fsz);
    };
    bg();
    if (showAll) {
      let n = 0;
      regions.forEach((fr) => fr.regions.forEach((r) => {
        n += 1;
        const pct = Math.round((r.mean_anomaly ?? fr.score ?? 0) * 100);
        box(r, n, areaColor(pct), `#${n}`);
      }));
    } else if (regions[sel]) {
      const fr = regions[sel];
      fr.regions.forEach((r, i) => {
        const pct = Math.round((r.mean_anomaly ?? fr.score ?? 0) * 100);
        box(r, i + 1, areaColor(pct), `${r.source === 'face_box' ? 'Face area (approx.)' : `Region ${i + 1}`} · ${pct}%`);
      });
      x.fillStyle = '#fff'; x.font = `${Math.max(12, W / 55)}px sans-serif`;
      if (fr.timestamp_sec != null) x.fillText(`t = ${fr.timestamp_sec}s`, 8, H - 10);
    }
  }, [result, regions, sel, showAll, kind, usingUpload, W, H]);

  useEffect(() => {
    if (!result || kind === 'audio') return;
    const v = videoRef.current;
    const fr = regions[sel];
    if (kind === 'video' && usingUpload && v && fr?.timestamp_sec != null && isFinite(v.duration)) {
      v.onseeked = draw; v.currentTime = fr.timestamp_sec;
    }
    draw();
  }, [draw, result, kind, sel, regions, usingUpload]);

  // ---- waveform
  const segs = result?.audio_segments || [];
  const dur = audioBuf?.duration || result?.audio_duration_sec || result?.video_duration_sec || 20;
  useEffect(() => {
    const c = waveRef.current;
    if (!c || !result || kind === 'image') return;
    const x = c.getContext('2d'), w = c.width, h = c.height;
    x.clearRect(0, 0, w, h);
    segs.forEach((g) => { x.fillStyle = 'rgba(255,93,115,.3)'; x.fillRect((g.start / dur) * w, 0, ((g.end - g.start) / dur) * w, h); });
    const d = audioBuf && usingUpload ? audioBuf.getChannelData(0) : null;
    x.fillStyle = '#4cc9f0';
    for (let i = 0; i < w; i += 2) {
      let a;
      if (d) {
        let m = 0; const st = Math.floor((i / w) * d.length), n = Math.floor((d.length / w) * 2);
        for (let k = 0; k < n; k += Math.ceil(n / 20)) m = Math.max(m, Math.abs(d[st + k] || 0));
        a = m;
      } else a = 0.12 + 0.4 * Math.abs(Math.sin(i * 0.05) * Math.sin(i * 0.013 + 1));
      x.fillRect(i, h / 2 - a * h * 0.45, 1.5, a * h * 0.9);
    }
    x.fillStyle = '#fff'; x.font = '11px sans-serif';
    segs.forEach((g, n) => x.fillText(`#${n + 1}`, (g.start / dur) * w + 3, 12));
  }, [result, kind, segs, dur, audioBuf, usingUpload]);

  const seekAudio = (e) => {
    const c = waveRef.current; const t = (e.nativeEvent.offsetX / c.clientWidth) * dur;
    const a = audioRef.current;
    if (a && a.src) { a.currentTime = t; a.play(); }
  };

  const jumpTo = (a) => {
    setHighlight(a.n); setShowAll(false);
    const idx = regions.findIndex((fr) => fr.frame_index === a.frame_index && fr.regions.length);
    if (idx >= 0) setSel(idx);
    canvasRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  // ---- chat
  const ask = async (text) => {
    if (!text.trim()) return;
    setQ('');
    setChat((c) => [...c, { who: 'u', text }]);
    if (!result) { setChat((c) => [...c, { who: 'a', text: 'Run a check or a demo first, so I have a result to read.' }]); return; }
    try {
      const r = await api('/api/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, result }) });
      setChat((c) => [...c, { who: 'a', text: r.answer, tools: r.tools_used, llm: r.llm_used }]);
    } catch {
      setChat((c) => [...c, { who: 'a', text: 'The chat service is offline (start the API). The result and evidence above are still valid.' }]);
    }
  };

  const [label, sev] = result ? LABELS[result.verdict_label] || LABELS.NOT_ENOUGH_INFO : [];
  const areas = result?.all_detected_areas || [];
  const kindWord = { video: 'video', image: 'image', audio: 'audio clip' }[kind];

  return (
    <>
      <header>
        <h1>🔍 DeepTrace — Deepfake Forensics</h1>
        <span className="tag">Upload → Check → See where → Ask why</span>
        {health && (
          <span className="tag" title={health.fast_reason || health.deep_reason || ''}>
            {health.offline ? 'API offline · bundled demo only' : `demo ✔ · fast ${health.fast_ready ? '✔' : '✖'} · deep ${health.deep_ready ? '✔' : '✖'}`}
          </span>
        )}
      </header>
      <main>
        {/* STEP 1 */}
        <section className="card">
          <h2>Step 1 · Upload a video, image or audio</h2>
          <p className="mu">Choose a file and press the button. The AI will tell you if it looks real or fake.</p>
          {fileUrl && kind === 'video' && <video ref={videoRef} src={fileUrl} controls />}
          {fileUrl && kind === 'audio' && <audio ref={audioRef} src={fileUrl} controls style={{ width: '100%' }} />}
          <div className="row">
            <label className="drop" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files[0]); }}>
              {file ? `📎 ${file.name}` : 'Drag & drop a file here, or click to choose'}
              <input type="file" accept="video/*,image/*,audio/*" hidden onChange={(e) => onFile(e.target.files[0])} />
            </label>
          </div>
          <div className="row">
            <span className="mu">Engine:</span>
            {[['demo', 'Demo (cached)'], ['fast', 'Fast (live)'], ['deep', 'Deep']].map(([k, t]) => (
              <button key={k} className={'g' + (engine === k ? ' on' : '')} onClick={() => setEngine(k)}>{t}</button>
            ))}
            <button className="run" disabled={busy} onClick={run}>{busy ? 'Checking…' : '🔍 Check if it is AI'}</button>
          </div>
          <div className="row">
            <span className="mu">Try a demo:</span>
            {['video', 'image', 'audio'].map((t) => (
              <button key={t} className="g" onClick={() => { setFile(null); setFileUrl(null); setAudioBuf(null); imgRef.current = null; loadDemo(t); }}>
                {{ video: '🎬 Video', image: '🖼 Image', audio: '🎧 Audio' }[t]}
              </button>
            ))}
          </div>
          {engine === 'demo' && file && <p className="mu small">Demo engine replays a cached result for this file type; your file is not analysed.</p>}
          {error && (
            <div className="err">⚠ {error}
              {offerDemo && <button className="run" onClick={() => { setEngine('demo'); loadDemo(kind); }}>Run demo instead</button>}
            </div>
          )}
        </section>

        {result && (
          <>
            {/* STEP 2 */}
            <section className="card">
              <h2>Step 2 · Result</h2>
              <Badge r={result} />
              <div className="verdict">
                <div className="big" style={{ color: COLOR[sev] }}>{label}</div>
                <div>
                  <div className="pct">{result.ai_probability_pct != null ? `${result.ai_probability_pct}%` : '—'}</div>
                  <div className="mu small">AI-probability estimate for this {kindWord} · certainty: <b>{result.certainty}</b></div>
                </div>
              </div>
              {result.provenance && <div className="mu small">{result.provenance}</div>}
              <b>Why we say this:</b>
              {(result.reasons || []).map((r, i) => <div className="reason" key={i}>{DOT[r.severity]} {r.text}</div>)}
              <div className="mu small">Estimate from pretrained models — not legal proof. Several independent signals agreeing means a stronger conclusion.</div>
            </section>

            {/* STEP 3 */}
            <section className="card">
              <h2>Step 3 · Where is it fake?</h2>
              {kind !== 'audio' && (
                regions.length === 0 ? <p className="mu">This result contains no position data, so no boxes are drawn.</p> : (
                  <>
                    {fs?.estimated && <p className="mu small">Frame size is estimated from the box positions, so placement is approximate.</p>}
                    <div className="row">
                      {kind === 'video' && withBoxes.map((f) => (
                        <button key={f.i} className={'g' + (!showAll && sel === f.i ? ' on' : '')} onClick={() => { setShowAll(false); setSel(f.i); }}>{f.timestamp_sec}s</button>
                      ))}
                      <button className={'g' + (showAll ? ' on' : '')} onClick={() => setShowAll(true)}>Show ALL on one image</button>
                    </div>
                    <canvas ref={canvasRef} className="ev" />
                    {!showAll && regions[sel] && (
                      <div className="mu small">
                        {regions[sel].timestamp_sec != null && <b>t={regions[sel].timestamp_sec}s · </b>}score {fmt(regions[sel].score)} · {regions[sel].regions.length} box(es)
                        {regions[sel].regions.some((r) => r.source === 'face_box') && ' · approximate (face area)'}
                      </div>
                    )}
                    {showAll && <div className="mu small">All detected areas from every checked moment. Overlapping boxes = same spot flagged repeatedly.</div>}
                  </>
                )
              )}
              {kind === 'video' && (result.timeline || []).length > 0 && <Timeline tl={result.timeline} regions={regions} onPick={(i) => { setShowAll(false); setSel(i); }} sel={sel} />}
              {kind !== 'image' && (
                <div className="sub">
                  <b>🎧 Voice evidence</b>
                  {segs.length === 0 ? (
                    <p className="mu small">{kind === 'audio' ? 'The voice checker gives one score for the whole clip, so it cannot say which seconds are fake. ' : 'No voice time segments are available for this result. '}Whole-clip result is in “Why we say this”.</p>
                  ) : (
                    <>
                      {result.sample_data && <div className="mu small">Segment times in this demo are sample data.</div>}
                      <canvas ref={waveRef} width="640" height="110" className="wave" onClick={seekAudio} />
                      <div className="mu small">Red blocks = parts the AI believes are synthetic. Click the wave to play from there.</div>
                    </>
                  )}
                </div>
              )}
              <div className="sub">
                <b>Every place the AI found</b>
                {areas.length === 0 ? <p className="mu small">Nothing to list: this result has no detected areas.</p> : (
                  <>
                    <div style={{ margin: '6px 0' }}>{areas.length} area(s) found</div>
                    {areas.map((a) => (
                      <div key={a.n} className={'area' + (highlight === a.n ? ' hl' : '')} onClick={() => a.kind === 'region' && jumpTo(a)}>
                        <span className="num" style={{ background: areaColor(a.percent_fake) }}>{a.n}</span>
                        {a.kind === 'audio' ? <>Voice at <b>{a.time}</b> — </> : a.time !== '-' ? <>At <b>{a.time}</b> · </> : null}
                        <b>{a.where}</b> — {a.what}. <span style={{ color: areaColor(a.percent_fake) }}>{a.label} ({a.percent_fake}%)</span>
                      </div>
                    ))}
                  </>
                )}
              </div>
            </section>
          </>
        )}

        {/* STEP 4 */}
        <section className="card">
          <h2>Step 4 · 🤖 Ask the AI assistant</h2>
          <div className="chat">
            {chat.map((m, i) => (
              <div key={i} className={'m ' + m.who}>
                {m.tools && <div className="tool">🔧 {m.tools.map((t) => t + '()').join(' → ')}{m.llm ? ' · LLM reworded' : ' · rule-based'}</div>}
                {m.text}
              </div>
            ))}
          </div>
          <div className="row">{CHIPS.map((c) => <span key={c} className="chip" onClick={() => ask(c)}>{c}</span>)}</div>
          <div className="row">
            <input type="text" value={q} placeholder="Ask about the evidence…" onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && ask(q)} />
            <button onClick={() => ask(q)}>Ask</button>
          </div>
        </section>

        {result && (
          <details className="card">
            <summary>⚙ Technical details (for experts)</summary>
            <div className="tech">
              <h3>Scores</h3>
              <Bars result={result} />
              {result.reliability_level && <p className="mu small">Reliability: {result.reliability_level} · confidence {fmt(result.confidence)}</p>}
              <h3>Timings</h3>
              <pre>{JSON.stringify(result.processing || 'not recorded', null, 1)}</pre>
              {result.sync_evidence?.offset_frames != null && <><h3>Lip-sync</h3><pre>{JSON.stringify(result.sync_evidence, null, 1)}</pre></>}
              {(result.evidence_list || result.evidence) && <><h3>Evidence</h3><pre>{JSON.stringify(result.evidence_list || result.evidence, null, 1)}</pre></>}
              {regions.length > 0 && (
                <>
                  <h3>Boxes (selected moment)</h3>
                  <table>
                    <thead><tr><th>x,y</th><th>w×h</th><th>mean</th><th>max</th><th>reliab.</th><th>source</th></tr></thead>
                    <tbody>{(regions[sel]?.regions || []).map((r, i) => (
                      <tr key={i}><td>{r.x},{r.y}</td><td>{r.w}×{r.h}</td><td>{fmt(r.mean_anomaly)}</td><td>{fmt(r.max_anomaly)}</td><td>{fmt(r.reliability)}</td><td>{r.source}</td></tr>
                    ))}</tbody>
                  </table>
                </>
              )}
              {result.visual_evidence?.anomaly_map_path && <img className="heat" alt="anomaly heatmap" src={API + result.visual_evidence.anomaly_map_path} />}
              <h3>Raw JSON</h3>
              <pre className="raw">{JSON.stringify(result, null, 1)}</pre>
            </div>
          </details>
        )}
      </main>
    </>
  );
}

function Bars({ result }) {
  const m = result.modalities || {};
  const rows = [['Visual', m.visual_score], ['Audio', m.audio_score], ['Lip-sync mismatch', m.sync_desync_score], ['Overall', result.manipulation_score ?? result.final_fake_probability]]
    .filter(([, v]) => typeof v === 'number');
  if (!rows.length) return <p className="mu small">No raw scores in this result.</p>;
  return rows.map(([k, v]) => (
    <div className="mod" key={k}><span>{k}</span><div className="bar"><i style={{ width: `${Math.round(v * 100)}%`, background: v > 0.5 ? '#ff5d73' : v > 0.35 ? '#ffb84d' : '#3ddc97' }} /></div><span>{fmt(v)}</span></div>
  ));
}

function Timeline({ tl, regions, onPick, sel }) {
  const w = 640, h = 160, p = 24;
  const tmax = Math.max(...tl.map((d) => d.t), 1);
  const X = (t) => p + (t / tmax) * (w - 2 * p), Y = (v) => h - p - v * (h - 2 * p);
  const pick = (t) => { const i = regions.findIndex((f) => f.timestamp_sec === t); if (i >= 0) onPick(i); };
  return (
    <div className="sub">
      <b>Score over time</b>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%">
        <line x1={p} x2={w - p} y1={Y(0.5)} y2={Y(0.5)} stroke="#ffb84d" strokeDasharray="4 4" />
        <text x={w - p - 60} y={Y(0.5) - 4}>0.5 line</text>
        <polyline fill="none" stroke="#4cc9f0" strokeWidth="2" points={tl.map((d) => `${X(d.t)},${Y(d.fake_prob)}`).join(' ')} />
        {tl.map((d, i) => (
          <circle key={i} cx={X(d.t)} cy={Y(d.fake_prob)} r={regions[sel]?.timestamp_sec === d.t ? 7 : 4.5} fill={d.suspicious ? '#ff5d73' : '#4cc9f0'} style={{ cursor: 'pointer' }} onClick={() => pick(d.t)}>
            <title>{`${d.t}s · ${fmt(d.fake_prob)}`}</title>
          </circle>
        ))}
        <text x={p} y={h - 6}>0s</text><text x={w - p - 20} y={h - 6}>{tmax}s</text>
      </svg>
      <div className="mu small">Red dot = suspicious moment. Click a dot to see its boxes.</div>
    </div>
  );
}
