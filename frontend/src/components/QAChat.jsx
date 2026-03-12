import { useState } from "react";
import { askQuestion } from "../api";
import CitationList from "./CitationList";

export default function QAChat({ patientId }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState([]);
  const [questionType, setQuestionType] = useState("");
  const [error, setError] = useState("");

  const handleAsk = async () => {
    if (!patientId || !question.trim()) return;
    setLoading(true);
    setError("");
    try {
      const data = await askQuestion(Number(patientId), question);
      setAnswer(data.answer);
      setCitations(data.citations || []);
      setQuestionType(data.question_type || "");
    } catch (e) {
      setError(e.message);
      setAnswer("");
      setCitations([]);
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
          {loading ? "Asking..." : "Ask"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {answer && (
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
