# RAG Hybrid — BTS CIEL IR

Système RAG hybride pour révisions BTS CIEL IR.  
Combine recherche vectorielle + FTS5 BM25 + re-ranking cross-encoder, 100% local.

## Architecture

```
Question (rag-chat / rag-web)
        │
        ▼
[1] Embedding  nomic-embed-text  → Mac Mini Ollama (:11434)
        │
        ▼
[2] Recherche hybride SQLite
    ├── Vectorielle : cosine similarity, index HNSW (O log n), cache LRU RAM
    ├── FTS5 BM25   : 3 stratégies (AND tech → OR tech → OR tous)
    └── RRF         : fusion des scores (constante 60)
        │
        ▼
[3] Re-ranker cross-encoder → MacBook Air local (:8765)
    ms-marco-MiniLM-L-6-v2 — re-score top-6 candidats, timeout 500ms
    fallback RRF si down
        │
        ▼
[4] checkCoherence — bloque les questions multi-domaines sans co-occurrence
        │
        ▼
[5] LLM → Mac Mini Ollama  (deepseek-coder-v2:16b / gemma2:9b / mistral:7b)
        │
        ▼
    Réponse avec citations de sources (noms de fichiers exacts)
```

## Prérequis

| Machine | Rôle |
|---|---|
| Mac Mini M4 | Ollama : embedding + LLM |
| MacBook Air | Re-ranker Python + binaires Go |

- Go 1.21+
- Python 3.10+ (`sentence-transformers`, `uvicorn`, `fastapi`)
- `pdftotext` : `brew install poppler`

## Installation

```bash
git clone https://github.com/dimitricl/rag-hybrid.git
cd rag-hybrid

# Dépendances Go
go mod download

# Dépendances Python
pip3 install sentence-transformers uvicorn fastapi reportlab --break-system-packages

# Modèles Ollama (sur le Mac Mini)
ollama pull nomic-embed-text
ollama pull mistral:7b-instruct
ollama pull deepseek-coder-v2:16b
ollama pull gemma2:9b
```

## Configuration

`configs/config.yaml` :

```yaml
ollama:
  host: "192.168.x.x"   # IP du Mac Mini
  port: 11434

reranker:
  host: "127.0.0.1"
  port: 8765

web:
  port: 8080

rag:
  db_path: "~/.rag-hybrid"
  embed_model: "nomic-embed-text:latest"
  default_model: "mistral:7b-instruct"
```

Override sans toucher au fichier :

```bash
OLLAMA_HOST=192.168.1.50 ./rag-web
```

## Build

```bash
go build -o rag      ./cmd/rag/
go build -o rag-chat ./cmd/rag-chat/
go build -o rag-web  ./cmd/rag-web/
```

## Utilisation

### 1. Re-ranker

```bash
python3 reranker.py > /tmp/reranker.log 2>&1 &
curl http://127.0.0.1:8765/health   # {"status":"ok"}
```

### 2. Indexation

```bash
./rag index ~/Documents/2CIEL_IR
```

Formats supportés : `.pdf` `.docx` `.pptx` `.md` `.txt` `.c` `.cpp` `.h` `.ino` `.py`

### 3. Interface terminal

```bash
./rag-chat
```

| Commande | Action |
|---|---|
| `!model deepseek-coder-v2:16b` | Changer de modèle |
| `!files` | Lister les documents indexés |
| `!stats` | Statistiques de la base |
| `!help` | Aide |
| `exit` | Quitter |

### 4. Interface web

```bash
./rag-web
# http://localhost:8080
```

Endpoint santé : `GET /health` → `{"status":"ok","chunks":N,"model":"..."}`

## Tests

```bash
# Suite 12 tests → rapport PDF
python3 rag_test.py --model deepseek-coder-v2:16b --output rapport.pdf

# Comparaison modèles
python3 rag_test.py --model mistral:7b-instruct --output rapport_mistral.pdf
python3 rag_test.py --model gemma2:9b           --output rapport_gemma2.pdf
```

### Résultats baseline (16/03/2026)

| Modèle | PASS | PARTIAL | FAIL | Taux | Temps moyen |
|---|---|---|---|---|---|
| deepseek-coder-v2:16b | 10 | 2 | 0 | 83% | 15.3s |
| gemma2:9b | 10 | 2 | 0 | 83% | 14.6s |
| mistral:7b-instruct | 9 | 3 | 0 | 75% | 19.9s |

## Structure

```
rag-hybrid/
├── cmd/
│   ├── rag/        # Indexation  (rag index <dossier>)
│   ├── rag-chat/   # Terminal streaming
│   └── rag-web/    # Web SSE  (port 8080)
├── pkg/
│   ├── chunker/    # Découpage en chunks (1500 runes, overlap 150)
│   ├── client/     # Client HTTP Ollama
│   ├── config/     # config.yaml + env override
│   ├── indexer/    # Pipeline d'indexation batch
│   ├── search/     # Moteur + prompts adaptatifs + checkCoherence
│   └── storage/    # SQLite FTS5 + HNSW + cache LRU + re-ranker
├── configs/
│   └── config.yaml
├── reranker.py     # Service cross-encoder (FastAPI/uvicorn)
└── rag_test.py     # Suite de tests → PDF
```

## Détails techniques

### Chunking
- Taille max 1500 runes, overlap 150 runes depuis dernière phrase complète
- Minimum 150 runes (filtre les fragments)
- Détection de sections (titres numérotés, ALL CAPS)
- Blocs de code préservés intacts
- Normalisation NFC des noms de fichiers (fix macOS NFD)

### Recherche hybride
- **Vectorielle** : cosine similarity, HNSW si > 2000 chunks, sinon full scan exact, cache LRU RAM borné
- **FTS5** : AND tech → OR tech → OR tous (3 niveaux de fallback)
- **RRF** : constante 60, fetchSize k×5, rerankPool 6
- **Re-ranker** : ms-marco-MiniLM-L-6-v2, MPS (Apple Silicon), fallback RRF si timeout

### Anti-hallucination
- `checkCoherence` : bloque les questions croisant deux domaines sans co-occurrence dans les sources
- Prompt adaptatif par type (registre, code, calcul, concept, général)
- Citations : noms de fichiers exacts uniquement

### Cache embedding
- Session RAM + persistant disque (`~/.rag-hybrid/embed_cache.json`)
- Invalidation au-delà de 500 entrées
- −78% latence sur questions répétées après redémarrage

## Modèles recommandés

| Usage | Modèle | Pourquoi |
|---|---|---|
| Registres AVR / Code | `deepseek-coder-v2:16b` | Meilleur sur le technique bas niveau |
| Questions générales | `gemma2:9b` | Même score que deepseek, plus léger |
| Tests rapides | `mistral:7b-instruct` | Rapide, moins fiable sur registres |
| Embedding | `nomic-embed-text:latest` | Meilleur sur corpus FR+technique |
