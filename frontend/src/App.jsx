import { useState, useRef } from 'react';
import './App.css';

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
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
    setShowAdvanced(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setResult(null);
    setShowAdvanced(false);

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
      setError(err.message || "Analysis could not be completed.");
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
    if (val === null || val === undefined) return "N/A";
    return val;
  };

  const formatPercentage = (val) => {
    if (val === null || val === undefined) return "Not available";
    return `${Math.round(val * 100)}%`;
  };
  
  const getVerdictInfo = (decision) => {
    switch (decision) {
      case "LIKELY_AUTHENTIC":
        return {
          title: "🟢 Likely Authentic",
          description: "We did not find strong signs that this media was created or changed using AI.",
          className: "verdict-LIKELY_AUTHENTIC"
        };
      case "LIKELY_MANIPULATED":
        return {
          title: "🔴 Likely AI-generated or manipulated",
          description: "We found several signs that suggest this media may have been created or changed using AI or other digital manipulation.",
          className: "verdict-LIKELY_MANIPULATED"
        };
      case "UNCERTAIN":
      default:
        return {
          title: "🟡 We can't be sure",
          description: "Some signs look normal, while other signs need further attention. We don't have enough evidence to confidently say whether this media is AI-generated or authentic.",
          className: "verdict-UNCERTAIN"
        };
    }
  };

  const calculateConfidence = (res) => {
    let completedChecks = 0;
    if (res.modalities?.visual_score !== null) completedChecks++;
    if (res.modalities?.audio_score !== null) completedChecks++;
    if (res.sync_evidence?.confidence !== null) completedChecks++;
    
    if (completedChecks === 0) return "Limited";
    
    // Simplistic heuristic for confidence wording
    const score = res.manipulation_score;
    if (score > 0.8 || score < 0.2) {
      if (completedChecks >= 2) return "High";
      return "Moderate";
    } else if (score > 0.4 && score < 0.6) {
      return "Mixed";
    } else {
      return "Moderate";
    }
  };

  const getOverallTakeaway = (decision, res) => {
    if (decision === "LIKELY_AUTHENTIC") {
      return "The available checks mostly support that this media is authentic. Because the available checks mostly support the same conclusion, we classified this media as Likely Authentic.";
    } else if (decision === "LIKELY_MANIPULATED") {
      return "The available checks mostly suggest that this media has been generated or manipulated. Because multiple signs point to manipulation, we classified this media as Likely AI-generated or manipulated.";
    } else {
      return "We found mixed signals during the analysis. Without stronger evidence pointing clearly in one direction, we cannot make a confident classification.";
    }
  };

  const getMeaningText = (decision) => {
    if (decision === "LIKELY_AUTHENTIC") {
      return "We found few or no signs that need attention. The overall evidence strongly points to authentic media.";
    } else if (decision === "LIKELY_MANIPULATED") {
      return "We found significant signs that need attention, and the overall evidence is strong enough to say that this media is likely AI-generated or manipulated.";
    } else {
      return "We found some signs that need attention, but the overall evidence is not strong enough to say that this media is AI-generated or manipulated.";
    }
  };

  const isVideo = file && file.type.startsWith("video/");

  return (
    <div className="app-container">
      {/* SECTION 12 - UPLOAD SCREEN */}
      {!loading && !result && (
        <>
        <header className="page-header">
          <h1>🔍 Is this AI-generated?</h1>
          <p>Upload a photo, video, or audio file and we'll check it for signs of AI generation or manipulation.</p>
        </header>

        <section className="card upload-card">
          <div className="upload-instruction">
            Drag & drop your file here
          </div>
          
          <div className="upload-supported">
            <div>
              <span>Photos</span>
              <small>JPG, JPEG, PNG</small>
            </div>
            <div>
              <span>Videos</span>
              <small>MP4, MOV, AVI</small>
            </div>
            <div>
              <span>Audio</span>
              <small>WAV, MP3, M4A</small>
            </div>
          </div>

          <div style={{ marginTop: '1rem', width: '100%', maxWidth: '400px' }}>
            {file ? (
              <div className="selected-file-info">
                <p style={{ fontWeight: '600', marginBottom: '0.25rem' }}>📄 File information</p>
                <p><strong>Name:</strong> {file.name}</p>
                <p><strong>Size:</strong> {(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            ) : (
              <label className="file-picker-label">
                Browse Files
                <input
                  type="file"
                  className="file-input"
                  accept="image/*,video/*,audio/*"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                />
              </label>
            )}
          </div>

          <div className="actions-row">
            <button className="btn-primary" onClick={handleAnalyze} disabled={!file || loading}>
              Analyze Media
            </button>
            {file && (
              <button className="btn-secondary" onClick={handleReset} disabled={loading}>
                Clear
              </button>
            )}
          </div>
        </section>

        <section className="capabilities-grid">
          <div className="capability-card">
            <h4>AI Image Detection</h4>
            <p>Detect suspicious AI-generated or manipulated imagery.</p>
          </div>
          <div className="capability-card">
            <h4>AI Video & Deepfake Detection</h4>
            <p>Analyze video for visual manipulation and audio-video inconsistencies.</p>
          </div>
          <div className="capability-card">
            <h4>AI Audio Detection</h4>
            <p>Identify suspicious synthetic or AI-generated speech.</p>
          </div>
          <div className="capability-card">
            <h4>Audio–Video Consistency</h4>
            <p>Check whether audio and visual speech are aligned.</p>
          </div>
          <div className="capability-card">
            <h4>Forensic Localization</h4>
            <p>Highlight areas that deserve further investigation.</p>
          </div>
          <div className="capability-card">
            <h4>Evidence-Based Scoring</h4>
            <p>Combine multiple forensic signals into one understandable assessment.</p>
          </div>
        </section>
        </>
      )}

      {/* SECTION 11 - PROCESSING SCREEN */}
      {loading && (
        <section className="card processing-card">
          <div className="spinner-container">
            <div className="spinner"></div>
          </div>
          <h2>Analyzing your file</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
            We're checking your file for signs of AI generation, editing, or other manipulation.
          </p>
          
          <ul className="processing-steps">
            <li>✓ Checking the image</li>
            <li>✓ Checking the audio</li>
            <li>✓ Checking whether audio and video match</li>
            <li>✓ Comparing all the results</li>
            <li>✓ Preparing your explanation</li>
          </ul>

          {isVideo && (
            <div className="processing-disclaimer">
              Video analysis may take 1–2 minutes on this local CPU prototype.
            </div>
          )}
        </section>
      )}

      {/* ERROR STATE */}
      {error && !loading && (
        <section className="card error-card">
          <h2>Analysis could not be completed.</h2>
          <p style={{ marginBottom: '1rem' }}><strong>What happened:</strong> {error}</p>
          <button className="btn-secondary" onClick={handleReset}>
            Try another file
          </button>
        </section>
      )}

      {/* RESULT DASHBOARD */}
      {result && !loading && !error && (
        <div>
          <div className="flex-between">
            <h2 className="section-title">Forensic Report</h2>
            <button className="btn-secondary" onClick={handleReset}>
              Analyze Another File
            </button>
          </div>

          {/* SECTION 2 - MAIN VERDICT */}
          <section className="card verdict-card">
            <div className={`verdict-status ${getVerdictInfo(result.decision).className}`}>
              {getVerdictInfo(result.decision).title}
            </div>
            <p className="verdict-explanation">
              {getVerdictInfo(result.decision).description}
            </p>
            
            <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
              <div className="score-display">
                <div className="score-value">{formatPercentage(result.manipulation_score)}</div>
                <div className="score-label">AI-generated signs detected</div>
              </div>

              <div className="score-display" style={{ backgroundColor: 'transparent', border: '1px solid var(--border-color)' }}>
                <div className="score-value" style={{ color: 'var(--accent-light)' }}>{calculateConfidence(result)}</div>
                <div className="score-label">🎯 How sure are we?</div>
              </div>
            </div>

            <p className="score-disclaimer" style={{ marginTop: '1.5rem', fontWeight: 'bold' }}>
              What does this mean?
            </p>
            <p className="score-disclaimer" style={{ maxWidth: '600px', margin: '0.5rem auto' }}>
              {getMeaningText(result.decision)} This is an AI detection score, not a guarantee.
            </p>
          </section>

          {/* SECTION 4 - WHY DID WE REACH THIS RESULT? */}
          <section className="card">
            <h3 className="section-title">🔎 Why do we think this?</h3>
            
            <div className="info-block">
              <div className="info-title" style={{ fontSize: '1rem', color: '#fff', textTransform: 'none', fontWeight: 'bold' }}>🖼️ Image check</div>
              <div className="info-content" style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>
                {result.modalities?.visual_score !== null && result.modalities?.visual_score !== undefined
                  ? (result.modalities.visual_score > 0.5 ? "Some unusual signs found." : "Looks mostly normal.")
                  : "Visual analysis was not performed or was unavailable."}
                {result.modalities?.visual_score !== null && (
                  <div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>We checked the image for signs of AI generation or digital editing. {isVideo ? "(Visual analysis currently uses a representative frame from the video.)" : ""}</div>
                )}
              </div>
            </div>

            <div className="info-block">
              <div className="info-title" style={{ fontSize: '1rem', color: '#fff', textTransform: 'none', fontWeight: 'bold' }}>🎙️ Voice check</div>
              <div className="info-content" style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>
                {result.modalities?.audio_score !== null && result.modalities?.audio_score !== undefined
                  ? (result.modalities.audio_score > 0.5 ? "Some signs of AI-generated speech." : "No strong signs of AI-generated speech.")
                  : "Audio could not be checked."}
                {result.modalities?.audio_score !== null && (
                  <div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>We checked the audio for patterns associated with AI-generated or cloned speech.</div>
                )}
              </div>
            </div>

            <div className="info-block">
              <div className="info-title" style={{ fontSize: '1rem', color: '#fff', textTransform: 'none', fontWeight: 'bold' }}>🔊 Audio & video check</div>
              <div className="info-content" style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>
                {result.sync_evidence && Object.keys(result.sync_evidence).length > 0
                  ? (result.sync_evidence.reliable 
                      ? "🟢 Audio and video match well." 
                      : (result.sync_evidence.confidence !== null ? "Audio and video may not match or could not be verified." : "Could not determine."))
                  : "Could not determine."}
                {result.sync_evidence && Object.keys(result.sync_evidence).length > 0 && (
                  <>
                  <div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>We checked whether the audio and video are closely aligned.</div>
                  {result.sync_evidence.offset_ms !== null && (
                    <div style={{ fontSize: '0.85rem', marginTop: '0.5rem', fontFamily: 'monospace' }}>Timing difference: {result.sync_evidence.offset_ms} ms</div>
                  )}
                  </>
                )}
              </div>
            </div>
          </section>

          {/* SECTION 6 - SUSPICIOUS AREA UX */}
          <section className="card">
            <h3 className="section-title">
              {result.suspicious_regions && result.suspicious_regions.length > 0 
                ? "🔍 Something unusual was found here"
                : "✅ Nothing unusual was found"}
            </h3>
            
            <p style={{ marginBottom: '1.5rem', color: 'var(--text-main)' }}>
              {result.suspicious_regions && result.suspicious_regions.length > 0 
                ? "Our image check found a small area that looks different from the surrounding image. This does not automatically mean the image is fake. It shows where our system noticed something unusual."
                : "Our image check did not find any clear areas that require attention."}
            </p>

            {/* Visual Evidence (Artifacts) */}
            {result.visual_evidence && (result.visual_evidence.anomaly_map_path || result.visual_evidence.reliability_map_path) ? (
              <div className="artifacts-grid">
                {result.visual_evidence.anomaly_map_path && (
                  <div className="artifact-item">
                    <div className="artifact-header">
                      <h4>Anomaly Map</h4>
                      <p>Highlights areas where the visual forensic model detected stronger manipulation-related signals.</p>
                    </div>
                    <img src={resolveArtifactUrl(result.visual_evidence.anomaly_map_path)} alt="Anomaly Map" className="artifact-img" />
                  </div>
                )}
                {result.visual_evidence.reliability_map_path && (
                  <div className="artifact-item">
                    <div className="artifact-header">
                      <h4>Reliability Map</h4>
                      <p>Shows how reliable the visual forensic analysis is across the image.</p>
                    </div>
                    <img src={resolveArtifactUrl(result.visual_evidence.reliability_map_path)} alt="Reliability Map" className="artifact-img" />
                  </div>
                )}
              </div>
            ) : null}
          </section>

          {/* SECTION 7 - OVERALL EXPLANATION */}
          <section className="card">
            <h3 className="section-title">💡 What should I take from this?</h3>
            <p style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>
              {getOverallTakeaway(result.decision, result)}
            </p>
          </section>

          {/* SECTION 14 - TECHNICAL DETAILS */}
          <div className="collapsible">
            <div 
              className="collapsible-header" 
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span>⚙️ Technical Details</span>
              <span>{showAdvanced ? "▲" : "▼"}</span>
            </div>
            {showAdvanced && (
              <div className="collapsible-content">
                <div className="tech-grid">
                  
                  <div className="tech-group">
                    <h4>IMAGE</h4>
                    <div className="tech-row">
                      <span className="tech-label">Model</span>
                      <span className="tech-value">TruFor</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Detection Score</span>
                      <span className="tech-value">{result.modalities?.visual_score !== null ? result.modalities.visual_score.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Number of unusual areas</span>
                      <span className="tech-value">{result.suspicious_regions ? result.suspicious_regions.length : 0}</span>
                    </div>
                  </div>

                  <div className="tech-group">
                    <h4>AUDIO</h4>
                    <div className="tech-row">
                      <span className="tech-label">Model</span>
                      <span className="tech-value">AASIST</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">AI speech/spoof score</span>
                      <span className="tech-value">{result.modalities?.audio_score !== null ? result.modalities.audio_score.toFixed(4) : 'N/A'}</span>
                    </div>
                  </div>

                  <div className="tech-group">
                    <h4>AUDIO-VIDEO</h4>
                    <div className="tech-row">
                      <span className="tech-label">Model</span>
                      <span className="tech-value">SyncNet</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Timing difference</span>
                      <span className="tech-value">{result.sync_evidence?.offset_ms !== null ? `${result.sync_evidence.offset_ms} ms` : 'N/A'}</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Technical confidence</span>
                      <span className="tech-value">{result.sync_evidence?.confidence !== null ? result.sync_evidence.confidence.toFixed(3) : 'N/A'}</span>
                    </div>
                  </div>

                  <div className="tech-group">
                    <h4>COMBINED</h4>
                    <div className="tech-row">
                      <span className="tech-label">Combined detection score</span>
                      <span className="tech-value">{result.manipulation_score !== null ? result.manipulation_score.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Final internal decision</span>
                      <span className="tech-value">{result.decision}</span>
                    </div>
                  </div>

                  <div className="tech-group">
                    <h4>PROCESSING TIME</h4>
                    <div className="tech-row">
                      <span className="tech-label">Visual processing time</span>
                      <span className="tech-value">{result.processing?.visual_time_ms ? `${(result.processing.visual_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Audio processing time</span>
                      <span className="tech-value">{result.processing?.audio_time_ms ? `${(result.processing.audio_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Sync processing time</span>
                      <span className="tech-value">{result.processing?.sync_time_ms ? `${(result.processing.sync_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                    </div>
                    <div className="tech-row">
                      <span className="tech-label">Total processing time</span>
                      <span className="tech-value">{result.processing?.total_time_ms ? `${(result.processing.total_time_ms / 1000).toFixed(2)} s` : 'N/A'}</span>
                    </div>
                  </div>

                  {result.suspicious_regions && result.suspicious_regions.length > 0 && (
                    <div className="tech-group" style={{ gridColumn: '1 / -1' }}>
                      <h4>REGION TECHNICAL DETAILS</h4>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>X</th>
                            <th>Y</th>
                            <th>Width</th>
                            <th>Height</th>
                            <th>Area</th>
                            <th>Mean Anomaly</th>
                            <th>Max Anomaly</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.suspicious_regions.map((r, i) => (
                            <tr key={i}>
                              <td>{formatScore(r.x)}</td>
                              <td>{formatScore(r.y)}</td>
                              <td>{formatScore(r.width)}</td>
                              <td>{formatScore(r.height)}</td>
                              <td>{formatScore(r.area)}</td>
                              <td>{r.mean_anomaly ? r.mean_anomaly.toFixed(3) : 'N/A'}</td>
                              <td>{r.max_anomaly ? r.max_anomaly.toFixed(3) : 'N/A'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                </div>
              </div>
            )}
          </div>
          
        </div>
      )}
    </div>
  );
}

export default App;
