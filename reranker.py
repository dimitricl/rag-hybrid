#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
from sentence_transformers import CrossEncoder
import json

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ CrossEncoder prêt sur :8765")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass

    def do_GET(self):
        # Health check — utilisé par rag-chat et rag-web au démarrage
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        query = data['query']
        chunks = data['chunks']
        pairs = [[query, c['text']] for c in chunks]
        scores = model.predict(pairs).tolist()
        for i, c in enumerate(chunks):
            c['rerank_score'] = scores[i]
        ranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        try:
            self.wfile.write(json.dumps(ranked).encode())
        except BrokenPipeError:
            pass  # Le client Go a fermé la connexion — normal avec timeout

HTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
