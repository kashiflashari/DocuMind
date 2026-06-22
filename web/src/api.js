// Thin client for the DocuMind FastAPI backend.
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export function health() {
  return request("/health");
}

export function query(question, topK) {
  return request("/query", {
    method: "POST",
    body: JSON.stringify({ question, top_k: topK ?? null }),
  });
}

export function ingest(text, source) {
  return request("/ingest", {
    method: "POST",
    body: JSON.stringify({ documents: [{ text, source: source || "pasted" }] }),
  });
}
