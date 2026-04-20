import { useState, useEffect, useRef } from "react";
import { askQuestion } from "../api";
import CitationList from "./CitationList";

const REASONING_KEYWORDS = ["trend", "over time", "history", "compare", "progression", "worsening", "changing", "evolv", "fluctuat", "how did", "how has"];

function isLikelyReasoning(q) {
  const lower = q.toLowerCase();
  return REASONING_KEYWORDS.some((kw) => lower.includes(kw));
}

export default function QAChat({ patientId }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState([]);
  const [questionType, setQuestionType] = useState("");
  const [error, setError] = useState("");
  const [blocked, setBlocked] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (loading) {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  const handleAsk = async () => {
    if (!patientId || !question.trim()) return;
    setLoading(true);
    setError("");
    setBlocked(false);
    setAnswer("");
    setCitations([]);
    setQuestionType("");
    try {
      const data = await askQuestion(Number(patientId), question);
      setAnswer(data.answer);
      setCitations(data.citations || []);
      setQuestionType(data.question_type || "");
    } catch (e) {
      if (e.status === 400) {
        setBlocked(true);
        setError(e.message);
      } else {
        setError(e.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Ask a Question</h2>
      <div className="row">
        <input
          className="input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What medications is the patient currently on?"
        />
        <button className="button" onClick={handleAsk} disabled={!patientId || loading}>
          {loading ? `Asking... (${elapsed}s)` : "Ask Question"}
        </button>
      </div>

      {loading && isLikelyReasoning(question) && (
        <p style={{ fontSize: "0.82rem", color: "#6c757d", marginTop: "6px" }}>
          Reasoning questions analyze multiple notes — this may take 60–90 seconds.
        </p>
      )}

      {blocked && (
        <div className="error" style={{ display: "flex", alignItems: "center", gap: "8px", background: "#fff3cd", border: "1px solid #ffc107", borderRadius: "6px", padding: "10px 14px", color: "#856404" }}>
          <span>&#9888;</span>
          <span>Query blocked: {error}</span>
        </div>
      )}
      {!blocked && error && <p className="error">{error}</p>}

      {answer && !error && (
        <>
          <div className="answer-box">
            <div className="pill">{questionType}</div>
            <p>{answer}</p>
          </div>
          <CitationList citations={citations} />
        </>
      )}
    </div>
  );
}