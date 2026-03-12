import { useEffect, useRef, useState } from "react";
import { fetchPatients, fetchSummary, askQuestion } from "./api";

export default function App() {
  const [patients, setPatients] = useState([]);
  const [patientId, setPatientId] = useState("");
  const [summary, setSummary] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [question, setQuestion] = useState("");
  const [qa, setQa] = useState(null);
  const [error, setError] = useState("");
  const [loadingPatients, setLoadingPatients] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingQa, setLoadingQa] = useState(false);
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
        setError("Failed to load patients");
      } finally {
        setLoadingPatients(false);
      }
    }

    loadPatients();
  }, []);

  const handleGenerateSummary = async () => {
    if (!patientId) return;

    // Cancel any in-flight summary request
    if (summaryAbortRef.current) {
      summaryAbortRef.current.abort();
    }
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
      if (err.name === "AbortError") return; // cancelled — ignore
      console.error("Error generating summary:", err);
      setError("Failed to generate summary");
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
      setError("Failed to get answer");
    } finally {
      setLoadingQa(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", padding: 20 }}>
      <h1>Vision AI</h1>
      <p>
        Privacy-preserving medical AI assistant for summaries and patient-specific
        RAG QA.
      </p>

      {error && <p style={{ color: "red" }}>{error}</p>}

      <div style={{ marginTop: 20, marginBottom: 20 }}>
        <label>
          <b>Select patient</b>
        </label>
        <br />
        <select
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          style={{ width: "100%", padding: 12, marginTop: 8 }}
          disabled={loadingPatients}
        >
          <option value="">
            {loadingPatients ? "Loading patients..." : "-- Select --"}
          </option>

          {patients.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: 20 }}>
        <button onClick={handleGenerateSummary} disabled={!patientId || loadingSummary}>
          {loadingSummary ? "Generating Summary..." : "Generate Summary"}
        </button>
      </div>

      {summary && (
        <div style={{ marginBottom: 30 }}>
          <h2>Summary</h2>

          {warnings.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <h4>Warnings</h4>
              <ul>
                {warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          {Object.entries(summary).map(([section, content]) => (
            <div key={section} style={{ marginBottom: 16 }}>
              <h3>{section}</h3>
              <p>{content}</p>
            </div>
          ))}
        </div>
      )}

      <div>
        <h2>Ask a Question</h2>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What medications is the patient currently on?"
          style={{ width: "80%", padding: 12, marginRight: 8 }}
        />
        <button onClick={handleAsk} disabled={!patientId || !question.trim() || loadingQa}>
          {loadingQa ? "Asking..." : "Ask"}
        </button>
      </div>

      {qa && (
        <div style={{ marginTop: 20 }}>
          <h3>Answer</h3>
          <p>{qa.answer}</p>

          <h4>Citations</h4>
          <ul>
            {(qa.citations || []).map((c) => (
              <li key={c.id}>
                [{c.id}] {c.date} | {c.note_type} | {c.section_name} | Note{" "}
                {c.note_id}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}