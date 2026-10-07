import { useState, useRef } from 'react';
import './App.css';

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze media.");
    } finally {
      setLoading(false);
    }
  };

  const resolveArtifactUrl = (path) => {
    if (!path) return null;
    if (path.startsWith("/")) return `${API_BASE_URL}${path}`;
    return path;
  };

  const formatScore = (val) => {
    if (val === null || val === undefined) return "Not available";
    return `${(val * 100).toFixed(1)}%`;
  };

  return (
    <div className="container">
      {/* SECTION 1 - HEADER */}
      <header className="header">
        <h1>Multimodal Deepfake & Digital Forensics</h1>
        <p>AI-powered forensic analysis for image, video and audio authenticity.</p>
      </header>

      {/* SECTION 2 - UPLOAD */}
      {!loading && !result && (
        <section className="card upload-section">
          <h2 className="card-title">Upload Media</h2>
          <div className="file-input-wrapper">
            <label htmlFor="media-upload">Select Image, Audio, or Video</label>
            <input
              id="media-upload"
              type="file"
              className="file-input"
              accept="image/*,video/*,audio/*"
              onChange={handleFileChange}
              ref={fileInputRef}
            />
          </div>
          {file && <p>Selected: {file.name}</p>}
          <div className="btn-group">
            <button className="btn" onClick={handleAnalyze} disabled={!file || loading}>
              Analyze
            </button>
            <button className="btn btn-secondary" onClick={handleReset} disabled={loading}>
              Clear
            </button>
          </div>
        </section>
      )}

      {/* SECTION 3 - PROCESSING STATE */}
      {loading && (
        <section className="card loading-state">
          <div className="spinner"></div>
          <h2>Running multimodal forensic analysis...</h2>
          <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>
            Depending on media type, video analysis may take approximately 1–2 minutes on this local CPU prototype.
          </p>
          <ul style={{ listStyle: 'none', marginTop: '1.5rem', display: 'inline-block', textAlign: 'left', color: 'var(--text-muted)' }}>
            <li>• Visual forensic analysis</li>
            <li>• Audio forensic analysis</li>
            <li>• Audio-video synchronization</li>
            <li>• Evidence fusion</li>
          </ul>
        </section>
      )}

      {/* SECTION 4 - ERROR STATE */}
      {error && !loading && (
        <section className="card error-card">
          <h2>Analysis failed</h2>
          <p>{error}</p>
          <button className="btn" onClick={handleReset} style={{ marginTop: '1rem' }}>
            Try Again
          </button>
        </section>
      )}

      {/* RESULT DASHBOARD */}
      {result && !loading && !error && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem' }}>
             {/* SECTION 13 - RESET */}
             <button className="btn btn-secondary" onClick={handleReset}>
               Analyze Another File
             </button>
          </div>

          {/* SECTION 5 - MAIN VERDICT */}
          <section className="card main-verdict">
            <div className="card-title" style={{ border: 'none', marginBottom: '0.5rem' }}>Forensic Verdict</div>
            <div className={`verdict-decision decision-${result.decision}`}>
              {result.decision.replace("_", " ")}
            </div>
            <div className="verdict-score">
              Manipulation Score: {formatScore(result.manipulation_score)}
            </div>
          </section>

          {/* SECTION 6 - MODALITY EVIDENCE */}
          <section className="grid-3">
            <div className="card modality-card">
              <div className="card-title">Visual Forensics</div>
              <div className="score">{formatScore(result.modalities?.visual_score)}</div>
              <p className="metric-label">Model Score</p>
            </div>
            <div className="card modality-card">
              <div className="card-title">Audio Forensics</div>
              <div className="score">{formatScore(result.modalities?.audio_score)}</div>
              <p className="metric-label">Model Score</p>
            </div>
            <div className="card modality-card">
              <div className="card-title">Audio-Video Sync</div>
              <div className="score">{formatScore(result.modalities?.sync_desync_score)}</div>
              <p className="metric-label">Fusion Evidence</p>
            </div>
          </section>

          {/* SECTION 7 - SYNC EVIDENCE */}
          <section className="card">
            <h2 className="card-title">Synchronization Evidence</h2>
            {result.sync_evidence && Object.keys(result.sync_evidence).length > 0 ? (
              <div className="metrics-list">
                <div className="metric-item">
                  <span className="metric-label">Offset</span>
                  <span className="metric-value">{result.sync_evidence.offset_ms !== null ? `${result.sync_evidence.offset_ms} ms` : 'N/A'}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Offset Frames</span>
                  <span className="metric-value">{result.sync_evidence.offset_frames !== null ? result.sync_evidence.offset_frames : 'N/A'}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Confidence</span>
                  <span className="metric-value">{result.sync_evidence.confidence !== null ? result.sync_evidence.confidence : 'N/A'}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Reliable</span>
                  <span className="metric-value">{result.sync_evidence.reliable !== null ? (result.sync_evidence.reliable ? 'Yes' : 'No') : 'N/A'}</span>
                </div>
              </div>
            ) : (
              <p>Synchronization analysis unavailable.</p>
            )}
          </section>

          {/* SECTION 8 - FORENSIC EXPLANATION */}
          <section className="card">
            <h2 className="card-title">Deterministic Evidence</h2>
            {result.evidence && result.evidence.length > 0 ? (
              <ul className="evidence-list">
                {result.evidence.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            ) : (
              <p>No deterministic evidence returned.</p>
            )}
          </section>

          {/* SECTION 9 - VISUAL FORENSIC ARTIFACTS */}
          <section className="card">
            <h2 className="card-title">Visual Forensic Artifacts</h2>
            {result.visual_evidence && (result.visual_evidence.anomaly_map_path || result.visual_evidence.reliability_map_path) ? (
              <div className="artifacts-grid">
                {result.visual_evidence.anomaly_map_path && (
                  <div className="artifact-item">
                    <h4>Anomaly Map</h4>
                    <img src={resolveArtifactUrl(result.visual_evidence.anomaly_map_path)} alt="Anomaly Map" className="artifact-img" />
                  </div>
                )}
                {result.visual_evidence.reliability_map_path && (
                  <div className="artifact-item">
                    <h4>Reliability Map</h4>
                    <img src={resolveArtifactUrl(result.visual_evidence.reliability_map_path)} alt="Reliability Map" className="artifact-img" />
                  </div>
                )}
              </div>
            ) : (
              <p>Visual forensic artifact unavailable.</p>
            )}
          </section>

          {/* SECTION 10 - SUSPICIOUS REGIONS */}
          <section className="card">
            <h2 className="card-title">Suspicious Regions</h2>
            {result.suspicious_regions && result.suspicious_regions.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Area</th>
                      <th>Bounding Box (X, Y, W, H)</th>
                      <th>Mean Anomaly</th>
                      <th>Max Anomaly</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.suspicious_regions.map((region, idx) => (
                      <tr key={idx}>
                        <td>{idx + 1}</td>
                        <td>{region.area}</td>
                        <td>{region.x}, {region.y}, {region.width}, {region.height}</td>
                        <td>{region.mean_anomaly ? region.mean_anomaly.toFixed(3) : 'N/A'}</td>
                        <td>{region.max_anomaly ? region.max_anomaly.toFixed(3) : 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>No suspicious regions were identified by the current forensic analysis.</p>
            )}
          </section>

          {/* SECTION 11 - SUSPICIOUS TIMELINE */}
          <section className="card">
            <h2 className="card-title">Suspicious Timeline</h2>
            {result.suspicious_timeline && result.suspicious_timeline.length > 0 ? (
              <ul className="evidence-list">
                {result.suspicious_timeline.map((item, idx) => (
                  <li key={idx}>{JSON.stringify(item)}</li>
                ))}
              </ul>
            ) : (
              <p>No suspicious timeline segments were returned.</p>
            )}
          </section>

          {/* SECTION 12 - PROCESSING METRICS */}
          <section className="card">
            <h2 className="card-title">Processing Metrics (Local Prototype)</h2>
            {result.processing && Object.keys(result.processing).length > 0 ? (
              <div className="metrics-list">
                <div className="metric-item">
                  <span className="metric-label">Visual Time</span>
                  <span className="metric-value">{result.processing.visual_time_ms ? `${(result.processing.visual_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Audio Time</span>
                  <span className="metric-value">{result.processing.audio_time_ms ? `${(result.processing.audio_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Sync Time</span>
                  <span className="metric-value">{result.processing.sync_time_ms ? `${(result.processing.sync_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Total Time</span>
                  <span className="metric-value">{result.processing.total_time_ms ? `${(result.processing.total_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                </div>
              </div>
            ) : (
              <p>Metrics unavailable.</p>
            )}
          </section>

        </div>
      )}
    </div>
  );
}

export default App;
