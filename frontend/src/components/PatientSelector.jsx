export default function PatientSelector({ patients, value, onChange }) {
  return (
    <div className="card">
      <label className="label">Select patient</label>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">-- Select --</option>
        {patients.map((p) => (
          <option key={p} value={p}>{p}</option>
        ))}
      </select>
    </div>
  );
}
