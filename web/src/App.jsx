import { useEffect, useRef, useState } from "react";
import { health, ingest, query } from "./api";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [docText, setDocText] = useState("");
  const [docSource, setDocSource] = useState("");
  const [toast, setToast] = useState("");
  const scrollRef = useRef(null);

  const refreshStatus = () => health().then(setStatus).catch(() => setStatus(null));
  useEffect(() => { refreshStatus(); }, []);
  useEffect(() => { scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight); }, [messages]);

  const flash = (msg) => { setToast(msg); setTimeout(() => setToast(""), 2500); };

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setLoading(true);
    try {
      const res = await query(question);
      setMessages((m) => [...m, { role: "assistant", text: res.answer, citations: res.citations }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "error", text: String(err.message || err) }]);
    } finally {
      setLoading(false);
    }
  }

  async function addDoc() {
    if (!docText.trim()) return;
    try {
      const res = await ingest(docText, docSource);
      flash(`Indexed ${res.ingested_chunks} chunk(s) · ${res.chunks_indexed} total`);
      setDocText(""); setDocSource("");
      refreshStatus();
    } catch (err) {
      flash(`Ingest failed: ${err.message || err}`);
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">◆</span>
          <div>
            <h1>DocuMind</h1>
            <p>RAG knowledge assistant</p>
          </div>
        </div>

        <div className="panel">
          <h2>Add knowledge</h2>
          <input
            className="field"
            placeholder="Source name (optional)"
            value={docSource}
            onChange={(e) => setDocSource(e.target.value)}
          />
          <textarea
            className="field textarea"
            placeholder="Paste document text to index…"
            value={docText}
            onChange={(e) => setDocText(e.target.value)}
          />
          <button className="btn" onClick={addDoc} disabled={!docText.trim()}>
            Ingest document
          </button>
        </div>

        {status && (
          <div className="status">
            <Row k="Store" v={status.store} />
            <Row k="Embeddings" v={status.embeddings} />
            <Row k="LLM" v={status.llm_provider} />
            <Row k="Re-ranker" v={status.rerank} />
            <Row k="Chunks" v={status.chunks_indexed} />
          </div>
        )}
      </aside>

      <main className="chat">
        <div className="messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="empty">
              <h3>Ask your knowledge base anything</h3>
              <p>Answers are grounded in your indexed documents, with inline citations.</p>
            </div>
          )}
          {messages.map((m, i) => (
            <Message key={i} m={m} />
          ))}
          {loading && <div className="bubble assistant typing">Thinking…</div>}
        </div>

        <form className="composer" onSubmit={send}>
          <input
            className="field"
            placeholder="Ask a question…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button className="btn primary" disabled={loading || !input.trim()}>Send</button>
        </form>
      </main>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function Message({ m }) {
  return (
    <div className={`bubble ${m.role}`}>
      <div className="bubble-text">{m.text}</div>
      {m.citations?.length > 0 && (
        <div className="citations">
          {m.citations.map((c) => (
            <details key={c.n} className="citation">
              <summary>[{c.n}] {c.source}</summary>
              <p>{c.snippet}</p>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="status-row">
      <span>{k}</span>
      <code>{String(v)}</code>
    </div>
  );
}
