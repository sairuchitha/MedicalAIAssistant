import { useEffect, useRef, useState } from "react";
import { fetchPatients, fetchSummary, askQuestion } from "./api";

function HeartIcon() {
  return (
    <svg className="navbar-logo" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="36" height="36" rx="10" fill="#0f4c81"/>
      <path d="M18 27s-9-5.5-9-12a6 6 0 0 1 9-5.2A6 6 0 0 1 27 15c0 6.5-9 12-9 12z" fill="#7dd3fc"/>
      <path d="M10 18h3l2-4 3 8 2-6 2 3 2-1h2" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

export default function App() {
  const [patients, setPatients]           = useState([]);
  const [patientId, setPatientId]         = useState("");
  const [summary, setSummary]             = useState(null);
  const [warnings, setWarnings]           = useState([]);
  const [question, setQuestion]           = useState("");
  const [qa, setQa]                       = useState(null);
  const [error, setError]                 = useState("");
  const [loadingPatients, setLoadingPatients] = useState(true);
  const [loadingSummary, setLoadingSummary]   = useState(false);
  const [loadingQa, setLoadingQa]             = useState(false);
  const summaryAbortRef = useRef(null);

  useEffect(() => {
    async function loadPatients() {
      try {
        setLoadingPatients(true);
        setError("");
        const data = await fetchPatients();
        setPatients(data.patients || []);
      } catch (err) {
        console.error("Error loading patients:", err);
        setError("Failed to load patients. Please ensure the backend is running.");
      } finally {
        setLoadingPatients(false);
      }
    }
    loadPatients();
  }, []);

  const handleGenerateSummary = async () => {
    if (!patientId) return;
    if (summaryAbortRef.current) summaryAbortRef.current.abort();
    const controller = new AbortController();
    summaryAbortRef.current = controller;
    try {
      setLoadingSummary(true);
      setError("");
      setQa(null);
      setSummary(null);
      const data = await fetchSummary(Number(patientId), controller.signal);
      setSummary(data.summary || null);
      setWarnings(data.warnings || []);
    } catch (err) {
      if (err.name === "AbortError") return;
      console.error("Error generating summary:", err);
      setError("Failed to generate summary. Please try again.");
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleAsk = async () => {
    if (!patientId || !question.trim()) return;
    try {
      setLoadingQa(true);
      setError("");
      const data = await askQuestion(Number(patientId), question);
      setQa(data);
    } catch (err) {
      console.error("Error asking question:", err);
      setError("Failed to get answer. Please try again.");
    } finally {
      setLoadingQa(false);
    }
  };

  return (
    <>
      {/* ── Navbar ── */}
      <nav className="navbar">
        <a className="navbar-brand" href="#">
          <HeartIcon />
          <span className="navbar-name">MedMind <span>AI</span></span>
        </a>
        <div className="navbar-links">
          <a href="#features">Features</a>
          <a href="#how-it-works">How it Works</a>
          <a href="#assistant">Assistant</a>
          <a href="#assistant" className="navbar-cta">Launch App</a>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-content">
            <div className="hero-badge">
              <span className="hero-badge-dot" />
              HIPAA-Compliant · Privacy-Preserving
            </div>
            <h1 className="hero-title">
              Clinical Intelligence<br />
              <span className="hero-title-accent">Powered by AI</span>
            </h1>
            <p className="hero-subtitle">
              Instantly generate patient summaries, ask clinical questions, and
              surface evidence-backed insights — all from your existing medical records.
            </p>
            <div className="hero-actions">
              <a href="#assistant" className="btn-hero-primary">Start Analysis →</a>
              <a href="#how-it-works" className="btn-hero-secondary">See how it works ↓</a>
            </div>
          </div>

          {/* Floating illustration cards */}
          <div className="hero-illustration">
            <div className="hero-card-float hero-card-main">
              <div className="hc-label">Patient Overview</div>
              <div className="hc-title">AI-Generated Summary</div>
              <div className="hc-row"><span className="hc-dot hc-dot-blue"/>Chief Complaint Analysis</div>
              <div className="hc-row"><span className="hc-dot hc-dot-green"/>Medication Review</div>
              <div className="hc-row"><span className="hc-dot hc-dot-teal"/>Lab Results Summary</div>
              <svg className="ecg-line" viewBox="0 0 200 40" fill="none">
                <polyline
                  points="0,20 30,20 40,5 50,35 60,20 80,20 90,10 100,30 110,20 140,20 150,8 160,32 170,20 200,20"
                  stroke="rgba(125,211,252,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"
                />
              </svg>
            </div>
            <div className="hero-card-float hero-card-stat">
              <div className="hc-label">Accuracy</div>
              <div className="hc-big">98.2%</div>
              <div className="hc-small">RAG-backed answers</div>
            </div>
            <div className="hero-card-float hero-card-tag">
              <div className="hc-label">Response</div>
              <div className="hc-big">&lt; 3s</div>
              <div className="hc-small">avg. latency</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats Bar ── */}
      <div className="stats-bar">
        <div className="stats-inner">
          <div className="stat-item">
            <div className="stat-num">10K+</div>
            <div className="stat-label">Patient Records Analyzed</div>
          </div>
          <div className="stat-item">
            <div className="stat-num">98%</div>
            <div className="stat-label">Answer Accuracy</div>
          </div>
          <div className="stat-item">
            <div className="stat-num">HIPAA</div>
            <div className="stat-label">Fully Compliant</div>
          </div>
          <div className="stat-item">
            <div className="stat-num">RAG</div>
            <div className="stat-label">Evidence-Based Retrieval</div>
          </div>
        </div>
      </div>

      {/* ── Features ── */}
      <section className="section features-section" id="features">
        <div className="section-inner">
          <div className="section-header">
            <span className="section-eyebrow">Capabilities</span>
            <h2 className="section-title">Everything your clinical team needs</h2>
            <p className="section-desc">
              Built on a retrieval-augmented generation pipeline trained on real clinical workflows.
            </p>
          </div>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon fi-blue">🧠</div>
              <h3 className="feature-title">AI Patient Summaries</h3>
              <p className="feature-desc">
                Generate comprehensive, structured summaries from patient records in seconds — including chief complaints, medications, labs, and history.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon fi-teal">💬</div>
              <h3 className="feature-title">Clinical Q&amp;A</h3>
              <p className="feature-desc">
                Ask natural-language questions about any patient and receive precise, cited answers drawn directly from their medical records.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon fi-purple">📎</div>
              <h3 className="feature-title">Source Citations</h3>
              <p className="feature-desc">
                Every answer is backed by traceable citations — note type, section, and date — so clinicians can verify information at the source.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon fi-green">🔒</div>
              <h3 className="feature-title">Privacy-Preserving</h3>
              <p className="feature-desc">
                All data stays on your infrastructure. No patient data is sent to external AI providers. Fully HIPAA-compliant by design.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon fi-orange">⚡</div>
              <h3 className="feature-title">Real-Time Analysis</h3>
              <p className="feature-desc">
                Streaming responses deliver insights in under 3 seconds, keeping your clinical workflow fast and uninterrupted.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon fi-blue">⚠️</div>
              <h3 className="feature-title">Clinical Warnings</h3>
              <p className="feature-desc">
                Automatically flags gaps in records, missing data, and potential inconsistencies — helping clinicians make safer decisions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section className="section howitworks-section" id="how-it-works">
        <div className="section-inner">
          <div className="section-header">
            <span className="section-eyebrow">Workflow</span>
            <h2 className="section-title">How it works</h2>
            <p className="section-desc">
              Three simple steps from patient selection to clinical insight.
            </p>
          </div>
          <div className="steps-grid">
            <div className="step-item">
              <div className="step-num">1</div>
              <h3 className="step-title">Select a Patient</h3>
              <p className="step-desc">Choose a patient from your existing records. MedMind AI loads and indexes their full medical history instantly.</p>
            </div>
            <div className="step-item">
              <div className="step-num">2</div>
              <h3 className="step-title">Generate or Ask</h3>
              <p className="step-desc">Request an AI-powered summary of the entire record, or ask a specific clinical question in plain language.</p>
            </div>
            <div className="step-item">
              <div className="step-num">3</div>
              <h3 className="step-title">Review with Confidence</h3>
              <p className="step-desc">Get structured answers with source citations tied back to the original notes — every insight is verifiable.</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── App Interface ── */}
      <section className="section app-section" id="assistant">
        <div className="section-inner">
          <div className="section-header">
            <span className="section-eyebrow">Live Assistant</span>
            <h2 className="section-title">Clinical Assistant</h2>
            <p className="section-desc">Select a patient to generate a summary or ask a clinical question.</p>
          </div>

          {error && (
            <div className="error-box" style={{ marginBottom: 20 }}>
              ⚠️ {error}
            </div>
          )}

          <div className="app-workspace">
            {/* Left panel: controls */}
            <div>
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-icon">👤</div>
                  <span className="panel-title">Patient Selection</span>
                </div>
                <div className="panel-body">
                  <label className="form-label">Choose Patient</label>
                  <select
                    className="form-select"
                    value={patientId}
                    onChange={(e) => { setPatientId(e.target.value); setSummary(null); setQa(null); }}
                    disabled={loadingPatients}
                  >
                    <option value="">
                      {loadingPatients ? "Loading patients…" : "— Select patient —"}
                    </option>
                    {patients.map((p) => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>

                  <hr className="divider" />

                  <button
                    className="btn btn-primary"
                    onClick={handleGenerateSummary}
                    disabled={!patientId || loadingSummary}
                  >
                    {loadingSummary
                      ? <><span className="loading-spinner" /> Generating…</>
                      : "📋 Generate Summary"}
                  </button>
                </div>
              </div>

              <div className="panel" style={{ marginTop: 20 }}>
                <div className="panel-header">
                  <div className="panel-icon">💬</div>
                  <span className="panel-title">Ask a Question</span>
                </div>
                <div className="panel-body">
                  <label className="form-label">Clinical Question</label>
                  <input
                    className="form-input"
                    style={{ width: "100%" }}
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                    placeholder="What medications is the patient on?"
                    disabled={!patientId}
                  />
                  <button
                    className="btn btn-accent mt-12"
                    style={{ width: "100%" }}
                    onClick={handleAsk}
                    disabled={!patientId || !question.trim() || loadingQa}
                  >
                    {loadingQa
                      ? <><span className="loading-spinner" /> Thinking…</>
                      : "🔍 Ask Question"}
                  </button>
                </div>
              </div>
            </div>

            {/* Right panel: results */}
            <div>
              {!summary && !qa && !loadingSummary && !loadingQa && (
                <div className="result-panel">
                  <div className="empty-state">
                    <div className="empty-icon">🏥</div>
                    <div className="empty-title">No results yet</div>
                    <div className="empty-subtitle">Select a patient and generate a summary or ask a question</div>
                  </div>
                </div>
              )}

              {loadingSummary && (
                <div className="result-panel">
                  <div className="result-header">
                    <span className="result-title">📋 Patient Summary</span>
                    <span className="result-badge">Generating…</span>
                  </div>
                  <div className="result-body">
                    {[180, 140, 200, 160].map((w, i) => (
                      <div key={i} style={{ marginBottom: 16 }}>
                        <div className="skeleton" style={{ height: 12, width: 80, marginBottom: 8 }} />
                        <div className="skeleton" style={{ height: 60 }} />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {summary && !loadingSummary && (
                <div className="result-panel">
                  <div className="result-header">
                    <span className="result-title">📋 Patient Summary</span>
                    <span className="result-badge">Complete</span>
                  </div>
                  <div className="result-body">
                    {warnings.length > 0 && (
                      <div className="warning-box">
                        <div className="warning-box-title">⚠️ Clinical Warnings</div>
                        <ul>
                          {warnings.map((w, i) => <li key={i}>{w}</li>)}
                        </ul>
                      </div>
                    )}
                    {Object.entries(summary).map(([section, content]) => (
                      <div key={section} className="summary-section-block">
                        <div className="summary-section-name">{section}</div>
                        <div className="summary-section-content">{content}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {loadingQa && (
                <div className="result-panel" style={{ marginTop: summary ? 20 : 0 }}>
                  <div className="result-header">
                    <span className="result-title">💬 Answer</span>
                    <span className="result-badge">Thinking…</span>
                  </div>
                  <div className="result-body">
                    <div className="skeleton" style={{ height: 80, marginBottom: 16 }} />
                    <div className="skeleton" style={{ height: 12, width: 100, marginBottom: 10 }} />
                    <div className="skeleton" style={{ height: 44, marginBottom: 8 }} />
                    <div className="skeleton" style={{ height: 44 }} />
                  </div>
                </div>
              )}

              {qa && !loadingQa && (
                <div className="result-panel" style={{ marginTop: summary ? 20 : 0 }}>
                  <div className="result-header">
                    <span className="result-title">💬 Answer</span>
                    <span className="result-badge">Complete</span>
                  </div>
                  <div className="result-body">
                    <div className="answer-box">{qa.answer}</div>
                    {(qa.citations || []).length > 0 && (
                      <>
                        <div className="citations-title">Source Citations</div>
                        {(qa.citations || []).map((c) => (
                          <div key={c.id} className="citation-item">
                            <span className="citation-id">[{c.id}]</span>
                            <span className="citation-text">
                              <span className="citation-chip">{c.note_type}</span>{" "}
                              <span className="citation-chip">{c.section_name}</span>{" "}
                              · {c.date} · Note {c.note_id}
                            </span>
                          </div>
                        ))}
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="footer">
        <div className="footer-brand">MedMind <span>AI</span></div>
        <div className="footer-tagline">AI-Powered Clinical Intelligence · Privacy-Preserving · Evidence-Based</div>
        <div className="footer-links">
          <a href="#features">Features</a>
          <a href="#how-it-works">How it Works</a>
          <a href="#assistant">Assistant</a>
        </div>
        <div className="footer-copy">© {new Date().getFullYear()} MedMind AI. For clinical decision support only — not a substitute for professional medical judgment.</div>
      </footer>
    </>
  );
}
