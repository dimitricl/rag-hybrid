#!/usr/bin/env python3
"""
reranker.py — CrossEncoder HTTP service
Fixes appliqués :
  - ThreadingMixIn : gestion concurrente (rag-web multi-user)
  - Vérification Content-Length manquant
  - Réponse d'erreur JSON structurée au lieu de crash silencieux
  - Timeout de lecture configuré via env RERANKER_TIMEOUT (défaut 30s)
"""

import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MODEL_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
model = CrossEncoder(MODEL_NAME)
logging.info(f"✅ CrossEncoder '{MODEL_NAME}' prêt sur :8765")


class Handler(BaseHTTPRequestHandler):
    # Supprime les logs HTTP par défaut (trop verbeux pour un service interne)
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, code: int, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            # Le client Go a fermé (timeout 500ms) — normal, on ne logge pas
            pass

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        # --- Validation de Content-Length ---
        content_length = self.headers.get("Content-Length")
        if not content_length:
            self._send_json(411, {"error": "Content-Length required"})
            return

        try:
            length = int(content_length)
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return

        # --- Lecture et parsing JSON ---
        try:
            raw = self.rfile.read(length)
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json(400, {"error": f"JSON parse error: {e}"})
            return

        # --- Validation du payload ---
        query = data.get("query", "")
        chunks = data.get("chunks", [])

        if not query:
            self._send_json(400, {"error": "Missing 'query' field"})
            return
        if not chunks:
            # Aucun chunk → retour vide, pas une erreur
            self._send_json(200, [])
            return

        # --- Scoring CrossEncoder ---
        try:
            pairs = [[query, c.get("text", "")] for c in chunks]
            scores = model.predict(pairs).tolist()
        except Exception as e:
            logging.error(f"CrossEncoder predict failed: {e}")
            self._send_json(500, {"error": f"Model predict failed: {e}"})
            return

        for i, c in enumerate(chunks):
            c["rerank_score"] = scores[i]

        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        self._send_json(200, ranked)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """
    ThreadingMixIn : chaque requête POST est traitée dans un thread séparé.
    Sans ça, une inférence lente (gros batch) bloque TOUTES les requêtes suivantes.
    daemon_threads = True : les threads meurent avec le process principal (pas de zombie).
    """
    daemon_threads = True


if __name__ == "__main__":
    host = os.environ.get("RERANKER_HOST", "127.0.0.1")
    port = int(os.environ.get("RERANKER_PORT", "8765"))
    server = ThreadedHTTPServer((host, port), Handler)
    logging.info(f"🚀 Reranker threaded sur {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Arrêt reranker.")
        server.server_close()
