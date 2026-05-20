package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"rag-hybrid/pkg/client"
	"rag-hybrid/pkg/config"
	"rag-hybrid/pkg/search"
	"rag-hybrid/pkg/storage"
)

var (
	engine *search.Engine
	cfg    config.Config
)

func main() {
	cfg = config.Load()

	cl := client.New(cfg.Ollama.Host, cfg.Ollama.Port)
	if err := cl.Health(); err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  Ollama inaccessible (%s): %v\n", cfg.OllamaURL(), err)
		os.Exit(1)
	}

	store, err := storage.New(cfg.RAG.DBPath, storage.StoreConfig{
		RRFConstant:       cfg.RAG.RRFConstant,
		FTSWeight:         cfg.RAG.FTSWeight,
		FTSTechWeight:     cfg.RAG.FTSTechWeight,
		RerankerTimeoutMs: cfg.RAG.RerankerTimeoutMs,
		RerankerURL:       cfg.RerankerURL(),
		RerankPool:        cfg.RAG.RerankPool,
		HNSWThreshold:     cfg.RAG.HNSWThreshold,
		VecCacheSize:      cfg.RAG.VecCacheSize,
		SearchRules:       cfg.SearchRules,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "erreur store: %v\n", err)
		os.Exit(1)
	}

	engine = search.NewWithConfig(cl, store, cfg.RAG)

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/ask", handleAsk)
	mux.HandleFunc("/health", handleHealth)

	addr := fmt.Sprintf(":%d", cfg.Web.Port)
	fmt.Printf("rag-web démarré sur http://localhost%s\n", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status": "ok",
		"chunks": engine.Store().Count(),
		"model":  cfg.RAG.DefaultModel,
	})
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, indexHTML)
}

func handleAsk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST requis", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Q     string `json:"q"`
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || strings.TrimSpace(req.Q) == "" {
		http.Error(w, "body JSON invalide ou question vide", http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		req.Model = cfg.RAG.DefaultModel
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming non supporté", http.StatusInternalServerError)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	ch, err := engine.AskStreamWithModel(ctx, req.Q, req.Model)
	if err != nil {
		fmt.Fprintf(w, "data: %s\n\n", jsonErr(err))
		flusher.Flush()
		return
	}

	for tok := range ch {
		escaped, _ := json.Marshal(tok)
		fmt.Fprintf(w, "data: %s\n\n", string(escaped))
		flusher.Flush()
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func jsonErr(err error) string {
	b, _ := json.Marshal(map[string]string{"error": err.Error()})
	return string(b)
}

const indexHTML = `<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Hybrid — BTS CIEL</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #0d1117; color: #c9d1d9; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
  h1 { color: #58a6ff; margin-bottom: 1.5rem; font-size: 1.4rem; }
  #chat { width: 100%; max-width: 800px; flex: 1; overflow-y: auto; margin-bottom: 1rem; }
  .msg { margin: 0.8rem 0; padding: 0.8rem 1rem; border-radius: 6px; white-space: pre-wrap; line-height: 1.5; }
  .user { background: #161b22; border-left: 3px solid #58a6ff; }
  .bot  { background: #0d1117; border-left: 3px solid #3fb950; }
  .err  { border-left-color: #f85149; color: #f85149; }
  #form { display: flex; gap: 0.5rem; width: 100%; max-width: 800px; }
  #q { flex: 1; padding: 0.6rem 0.8rem; background: #161b22; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; font-family: monospace; font-size: 0.95rem; }
  #q:focus { outline: none; border-color: #58a6ff; }
  button { padding: 0.6rem 1.2rem; background: #238636; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-family: monospace; }
  button:hover { background: #2ea043; }
  select { padding: 0.6rem; background: #161b22; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; font-family: monospace; }
</style>
</head>
<body>
<h1>⚡ RAG Hybrid — BTS CIEL IR</h1>
<div id="chat"></div>
<div id="form">
  <select id="model">
    <option value="gemma4:latest">gemma4 (défaut)</option>
    <option value="deepseek-coder-v2:16b">deepseek-coder:16b</option>
    <option value="qwen2.5:14b-instruct-q4_K_M">qwen2.5:14b</option>
    <option value="mistral-nemo:latest">mistral-nemo</option>
    <option value="gemma2:9b">gemma2:9b</option>
  </select>
  <input id="q" type="text" placeholder="Pose ta question..." autocomplete="off">
  <button onclick="ask()">Envoyer</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('q');

input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

async function ask() {
  const q = input.value.trim();
  if (!q) return;
  const model = document.getElementById('model').value;
  input.value = '';
  addMsg(q, 'user');
  const bot = addMsg('', 'bot');

  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ q, model })
  });

  if (!res.ok) { bot.textContent = 'Erreur ' + res.status; bot.classList.add('err'); return; }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (!line.startsWith('data: ')) continue;
      const d = line.slice(6).trim();
      if (d === '[DONE]') break;
      try {
        const parsed = JSON.parse(d);
        if (typeof parsed === 'string') {
          bot.textContent += parsed;
        } else if (parsed && parsed.error) {
          bot.textContent = parsed.error;
          bot.classList.add('err');
        }
      } catch { bot.textContent += d; }
      chat.scrollTop = chat.scrollHeight;
    }
  }
}
</script>
</body>
</html>`
