# RAG Hybrid — BTS CIEL IR

Système de Retrieval-Augmented Generation (RAG) hybride pour révisions BTS CIEL IR.  
Combine recherche vectorielle (cosine similarity) + recherche plein-texte (FTS5 BM25) + re-ranking cross-encoder.

## Architecture

```
Question (rag-chat / rag-web)
        │
        ▼
[1] Embedding nomic-embed-text → Mac Mini Ollama (:11434)
        │
        ▼
[2] Recherche hybride SQLite
    ├── Vectorielle : cosine similarity sur vecteurs 768 dims (RAM)
    └── FTS5 BM25   : recherche plein-texte avec scoring BM25 natif
    └── RRF          : fusion des deux scores (Reciprocal Rank Fusion)
        │
        ▼
[3] Re-ranker cross-encoder → MacBook Air local (:8765)
    ms-marco-MiniLM-L-6-v2 — re-score les top-6 candidats
        │
        ▼
[4] checkCoherence : bloque les questions multi-domaines sans co-occurrence
        │
        ▼
[5] LLM → Mac Mini Ollama (deepseek-coder-v2:16b / gemma2:9b / mistral:7b)
        │
        ▼
    Réponse avec citations de sources
```

## Prérequis

- **Mac Mini M4** avec [Ollama](https://ollama.com) installé et accessible sur le réseau
- **MacBook Air** pour le re-ranker Python et les binaires Go
- Go 1.21+
- Python 3.10+ avec `sentence-transformers`
- `pdftotext` (poppler) : `brew install poppler`

## Installation

```bash
# Clone le repo
git clone https://github.com/dimitricl/rag-hybrid.git
cd rag-hybrid

# Dépendances Go
go mod download

# Dépendances Python
pip3 install sentence-transformers --break-system-packages
pip3 install reportlab --break-system-packages

# Modèles Ollama (sur le Mac Mini)
ollama pull nomic-embed-text
ollama pull mistral:7b-instruct
ollama pull deepseek-coder-v2:16b
ollama pull gemma2:9b
```

## Configuration

Édite `configs/config.yaml` :

```yaml
ollama:
  host: "192.168.x.x"   # IP de ton Mac Mini
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

## Build

```bash
go build -o rag ./cmd/rag/
go build -o rag-chat ./cmd/rag-chat/
go build -o rag-web ./cmd/rag-web/
```

## Utilisation

### 1. Démarrer le re-ranker

```bash
python3 reranker.py &
```

### 2. Indexer les documents

```bash
./rag index ~/Documents/2CIEL_IR
```

Formats supportés : `.pdf`, `.docx`, `.pptx`, `.md`, `.txt`, `.c`, `.cpp`, `.h`, `.ino`, `.py`

### 3. Interface terminal

```bash
./rag-chat
```

Commandes disponibles dans rag-chat :
```
!model deepseek-coder-v2:16b   # changer de modèle
!files                          # lister les documents indexés
!stats                          # statistiques de la base
!help                           # aide complète
exit                            # quitter
```

### 4. Interface web

```bash
./rag-web
# Ouvre http://localhost:8080
```

## Tests automatisés

```bash
# Lance la suite de 12 tests et génère un rapport PDF
python3 rag_test.py --model deepseek-coder-v2:16b --output rapport.pdf

# Comparaison entre modèles
python3 rag_test.py --model mistral:7b-instruct --output rapport_mistral.pdf
python3 rag_test.py --model gemma2:9b --output rapport_gemma2.pdf
```

### Résultats baseline (16/03/2026)

| Modèle | PASS | PARTIAL | FAIL | Taux | Temps moyen |
|--------|------|---------|------|------|-------------|
| deepseek-coder-v2:16b | 10 | 2 | 0 | **83%** | 15.3s |
| gemma2:9b | 10 | 2 | 0 | **83%** | 14.6s |
| mistral:7b-instruct | 9 | 3 | 0 | **75%** | 19.9s |

## Structure du projet

```
rag-hybrid/
├── cmd/
│   ├── rag/            # Binaire d'indexation (rag index <dossier>)
│   ├── rag-chat/       # Interface terminal streaming
│   └── rag-web/        # Interface web SSE
├── pkg/
│   ├── chunker/        # Découpage des documents en chunks
│   ├── client/         # Client HTTP Ollama
│   ├── config/         # Chargement config.yaml
│   ├── indexer/        # Pipeline d'indexation batch
│   ├── search/         # Moteur de recherche + prompts adaptatifs
│   └── storage/        # SQLite FTS5 + vecteurs binaires + re-ranker
├── configs/
│   └── config.yaml     # Configuration (IP, ports, modèles)
├── reranker.py         # Service re-ranker Python (cross-encoder)
├── rag_test.py         # Suite de tests → rapport PDF
└── .gitignore
```

## Détails techniques

### Chunking
- Taille max : 1000 runes, overlap depuis dernière phrase complète
- Minimum : 150 runes (filtre les fragments)
- Détection de sections (titres numérotés, ALL CAPS)
- Normalisation NFC des noms de fichiers (fix macOS NFD)
- Blocs de code préservés intacts

### Recherche hybride
- **Vectorielle** : cosine similarity, cache RAM complet, fallback disque
- **FTS5** : stratégie 3 niveaux (AND tech → OR tech → OR tous)
- **BM25** : score natif FTS5 intégré dans le RRF
- **RRF** : constante 60, fetchSize k×5, rerankPool 6
- **Re-ranker** : ms-marco-MiniLM-L-6-v2, timeout 500ms, fallback RRF si down

### Anti-hallucination
- `checkCoherence` : bloque les questions croisant deux domaines sans co-occurrence dans les sources (ex: ADC + infrarouge)
- Prompt adaptatif par type de question (registre, code, calcul, concept)
- Règle citation : noms de fichiers exacts uniquement, pas d'URLs inventées

### Cache embedding
- En mémoire (session) + persistant sur disque (`~/.rag-hybrid/embed_cache.json`)
- Invalidation automatique si > 500 entrées
- Gain : -78% latence sur questions répétées après redémarrage

## Modèles recommandés

| Usage | Modèle | Pourquoi |
|-------|--------|----------|
| Registres AVR / Code | `deepseek-coder-v2:16b` | Meilleur sur le technique bas niveau |
| Questions générales | `gemma2:9b` | Même score, plus léger, plus rapide |
| Tests rapides | `mistral:7b-instruct` | Rapide mais moins fiable sur les registres |
| Embedding | `nomic-embed-text:latest` | Meilleur sur ce corpus FR+technique |
