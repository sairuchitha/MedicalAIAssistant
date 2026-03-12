const BASE_URL = "http://127.0.0.1:8000";

export async function fetchPatients() {
  const res = await fetch(`${BASE_URL}/api/patients`);
  if (!res.ok) {
    throw new Error("Failed to fetch patients");
  }
  return res.json();
}

export async function fetchSummary(patientId) {
  const res = await fetch(`${BASE_URL}/api/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: patientId }),
  });
  if (!res.ok) {
    throw new Error("Failed to fetch summary");
  }
  return res.json();
}

export async function askQuestion(patientId, question) {
  const res = await fetch(`${BASE_URL}/api/qa`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: patientId, question }),
  });
  if (!res.ok) {
    throw new Error("Failed to fetch QA");
  }
  return res.json();
}