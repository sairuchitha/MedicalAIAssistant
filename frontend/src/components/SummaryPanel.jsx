export default function SummaryPanel({ summary, warnings }) {
  if (!summary) return null;

  return (
    <div className="card">
      <h2>Structured Summary</h2>
      {warnings?.length > 0 && (
        <div className="warning-box">
          <strong>Warnings</strong>
          <ul>
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}
      {Object.entries(summary).map(([k, v]) => (
        <div key={k} className="section">
          <h3>{k}</h3>
          <p>{v}</p>
        </div>
      ))}
    </div>
  );
}
