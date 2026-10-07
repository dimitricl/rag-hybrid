package storage

import (
	"bufio"
	"bytes"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/coder/hnsw"
	_ "modernc.org/sqlite"

	"rag-hybrid/pkg/config"
)

const defaultVecDim = 768 // nomic-embed-text ; la vraie dim est détectée au 1er insert

type Chunk struct {
	ID       string    `json:"id"`
	Text     string    `json:"text"`
	Filename string    `json:"filename"`
	Vector   []float32 `json:"-"`
	Score    float32   `json:"score"`
	RRFRaw   float32   `json:"rrf_raw"`
}

type Store struct {
	db       *sql.DB
	vecs     *os.File
	mu       sync.Mutex
	vecCache *lruCache           // Cache RAM LRU borné : clé = chunk.ID
	hnswIdx  *hnsw.Graph[string] // Index ANN : O(log n) au lieu de O(n)
	hnswPath string              // Chemin de persistance hnsw.bin
	vecDim   int                 // Dimension des vecteurs (détectée au 1er insert, persistée en table meta)
	// Paramètres RRF/reranker issus de config.yaml
	rrfConstant       float32
	ftsWeight         float32
	ftsTechWeight     float32
	rerankerTimeoutMs int
	rerankerURL       string
	rerankPool        int          // nb de candidats envoyés au reranker — depuis config.yaml
	hnswThreshold     int          // full scan si len(vecCache) < seuil, HNSW sinon
	hnswDirty         bool         // true si l'index HNSW a été modifié depuis le dernier saveHNSW
	maxPerFile        int          // max chunks par fichier dans les résultats finaux (diversité, 0=illimité)
	mmrLambda         float32      // 0..1 : 1 = pertinence pure, 0 = diversité pure (défaut 0.7, 0 = désactivé)
	httpClient        *http.Client // Client HTTP réutilisé pour le reranker (évite les leaks de connexions)
	searchRules       config.SearchRules
	stopWordsMap      map[string]bool
}

// StoreConfig regroupe les paramètres injectés depuis config.yaml
type StoreConfig struct {
	RRFConstant       float32
	FTSWeight         float32
	FTSTechWeight     float32
	RerankerTimeoutMs int
	RerankerURL       string
	RerankPool        int // Nb de candidats envoyés au cross-encoder (défaut 6)
	HNSWThreshold     int // Full scan si nb chunks < seuil, HNSW sinon
	VecCacheSize      int // Capacité max du cache LRU (0 = illimité)
	MaxPerFile        int // Max chunks par fichier dans résultats finaux (0=illimité, défaut 2)
	MMRLambda         float32
	SearchRules       config.SearchRules
}

func New(base string, cfg StoreConfig) (*Store, error) {
	p := os.ExpandEnv(base)
	if len(p) > 0 && p[0] == '~' {
		h, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("New: UserHomeDir: %w", err)
		}
		p = filepath.Join(h, p[1:])
	}
	if p == "" {
		return nil, fmt.Errorf("New: db_path est vide")
	}
	os.MkdirAll(p, 0755)

	db, err := sql.Open("sqlite", filepath.Join(p, "rag.db"))
	if err != nil {
		return nil, err
	}

	_, err = db.Exec(`
		PRAGMA journal_mode=WAL;
		PRAGMA synchronous=NORMAL;
		PRAGMA cache_size=-64000;
		PRAGMA mmap_size=268435456;

		CREATE TABLE IF NOT EXISTS chunks (
			id TEXT PRIMARY KEY,
			text TEXT,
			filename TEXT,
			offset INTEGER
		);

		CREATE TABLE IF NOT EXISTS meta (
			key TEXT PRIMARY KEY,
			value TEXT
		);

		CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
			text, 
			content='chunks', 
			content_rowid='rowid',
			tokenize='unicode61 remove_diacritics 1'
		);
	`)
	if err != nil {
		return nil, err
	}

	vf, err := os.OpenFile(filepath.Join(p, "vectors.bin"), os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}

	hnswPath := filepath.Join(p, "hnsw.bin")

	s := &Store{
		db:                db,
		vecs:              vf,
		vecCache:          newLRUCache(cfg.VecCacheSize),
		hnswPath:          hnswPath,
		vecDim:            0, // détecté ci-dessous (meta) ou au 1er InsertBatch
		rrfConstant:       cfg.RRFConstant,
		ftsWeight:         cfg.FTSWeight,
		ftsTechWeight:     cfg.FTSTechWeight,
		rerankerTimeoutMs: cfg.RerankerTimeoutMs,
		rerankerURL:       cfg.RerankerURL,
		rerankPool:        cfg.RerankPool,
		hnswThreshold:     cfg.HNSWThreshold,
		maxPerFile:        cfg.MaxPerFile,
		mmrLambda:         cfg.MMRLambda,
		searchRules:       cfg.SearchRules,
		stopWordsMap:      make(map[string]bool),
		httpClient:        &http.Client{Timeout: time.Duration(cfg.RerankerTimeoutMs) * time.Millisecond},
	}

	for _, w := range cfg.SearchRules.StopWords {
		s.stopWordsMap[w] = true
	}

	// --- Dimension des vecteurs (persistée en table meta) ---
	var dimStr string
	if err := db.QueryRow("SELECT value FROM meta WHERE key='vec_dim'").Scan(&dimStr); err == nil {
		var d int
		if _, err := fmt.Sscanf(dimStr, "%d", &d); err == nil && d > 0 {
			s.vecDim = d
		}
	}

	// --- Chargement des vecteurs en RAM ---
	info, err := vf.Stat()
	if err == nil && info.Size() > 0 && s.vecDim > 0 {
		vecBytes := int64(s.vecDim * 4)
		data := make([]byte, info.Size())
		if _, err := vf.ReadAt(data, 0); err != nil {
			return nil, fmt.Errorf("New: lecture vectors.bin: %w", err)
		}
		rows, rerr := db.Query("SELECT id, offset FROM chunks WHERE offset IS NOT NULL ORDER BY offset")
		if rerr == nil {
			loaded := 0
			for rows.Next() {
				var id string
				var offset int64
				if rows.Scan(&id, &offset) == nil && offset+vecBytes <= int64(len(data)) {
					vec := make([]float32, s.vecDim)
					for i := 0; i < s.vecDim; i++ {
						vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset+int64(i*4):]))
					}
					s.vecCache.set(id, vec)
					loaded++
				}
			}
			rows.Close()
			if loaded > 0 {
				fmt.Printf("✅ %d vecteurs chargés (dim=%d)\n", loaded, s.vecDim)
			}
		}
	}

	// Avertit si la base contient des chunks sans vecteurs (indexation pré-fix)
	var total, withVec int
	_ = db.QueryRow("SELECT COUNT(*) FROM chunks").Scan(&total)
	_ = db.QueryRow("SELECT COUNT(*) FROM chunks WHERE offset IS NOT NULL").Scan(&withVec)
	if total > 0 && withVec == 0 {
		fmt.Printf("⚠️  %d chunks sans vecteurs (base pré-fix) — recherche vectorielle inactive, relancez `rag index --clear <dossier>`\n", total)
	}

	// --- Chargement ou reconstruction de l'index HNSW ---
	s.hnswIdx = s.loadOrBuildHNSW(hnswPath)

	return s, nil
}

// loadOrBuildHNSW charge l'index depuis hnsw.bin si disponible,
// sinon le reconstruit depuis vecCache (après un rm hnsw.bin ou première indexation).
func (s *Store) loadOrBuildHNSW(path string) *hnsw.Graph[string] {
	// Tentative de chargement depuis le fichier persisté
	if f, err := os.Open(path); err == nil {
		defer f.Close()
		g := hnsw.NewGraph[string]()
		g.EfSearch = 100 // Augmenté pour plus de précision (défaut=20)
		// Import nécessite un io.ByteReader — os.File ne l'implémente pas
		// sans bufio, binary.Read retourne "does not implement io.ByteReader"
		if err := g.Import(bufio.NewReader(f)); err == nil {
			fmt.Printf("✅ Index HNSW chargé (%d vecteurs)\n", s.vecCache.len())
			return g
		}
		fmt.Printf("⚠️  hnsw.bin corrompu — reconstruction depuis vecCache\n")
	}

	// Reconstruction depuis le cache RAM
	g := hnsw.NewGraph[string]()
	g.EfSearch = 100 // Augmenté pour plus de précision (défaut=20)
	if s.vecCache.len() == 0 {
		return g
	}

	var nodes []hnsw.Node[string]
	s.vecCache.forEach(func(id string, vec []float32) {
		nodes = append(nodes, hnsw.MakeNode(id, vec))
	})
	g.Add(nodes...)
	fmt.Printf("✅ Index HNSW construit (%d vecteurs)\n", len(nodes))

	// Persistance immédiate
	s.saveHNSW(g, path)
	return g
}

// saveHNSW sérialise l'index HNSW sur disque (écriture atomique).
func (s *Store) saveHNSW(g *hnsw.Graph[string], path string) {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		fmt.Printf("⚠️  saveHNSW create: %v\n", err)
		return
	}
	if err := g.Export(f); err != nil {
		f.Close()
		os.Remove(tmp)
		fmt.Printf("⚠️  saveHNSW encode: %v\n", err)
		return
	}
	f.Close()
	if err := os.Rename(tmp, path); err != nil {
		fmt.Printf("⚠️  saveHNSW rename: %v\n", err)
	}
}

func (s *Store) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hnswDirty {
		s.saveHNSW(s.hnswIdx, s.hnswPath)
		s.hnswDirty = false
	}
	s.db.Close()
	s.vecs.Close()
}

// FlushHNSW force la persistance de l'index HNSW sur disque si modifié.
// À appeler explicitement après une indexation batch pour ne pas attendre Close().
func (s *Store) FlushHNSW() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hnswDirty {
		s.saveHNSW(s.hnswIdx, s.hnswPath)
		s.hnswDirty = false
		fmt.Printf("✅ Index HNSW persisté (%d vecteurs)\n", s.vecCache.len())
	}
}

func (s *Store) InsertBatch(ids, texts, filenames []string, vecs [][]float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(ids) == 0 {
		return nil
	}
	if len(vecs) > 0 && len(vecs[0]) > 0 {
		dim := len(vecs[0])
		if s.vecDim == 0 {
			s.vecDim = dim
			_, _ = s.db.Exec("INSERT OR REPLACE INTO meta(key, value) VALUES('vec_dim', ?)", fmt.Sprintf("%d", dim))
			fmt.Printf("📐 Dimension vecteurs détectée : %d\n", dim)
		} else if dim != s.vecDim {
			return fmt.Errorf("InsertBatch: dimension incohérente %d != %d (changez de modèle d'embedding → réindexez avec --clear)", dim, s.vecDim)
		}
	}
	vecBytes := s.vecDim * 4

	// 1. Offset de fin de fichier pour append
	endOff, err := s.vecs.Stat()
	var baseOff int64
	if err == nil {
		baseOff = endOff.Size()
	}

	// 2. Insertion SQL avec offset
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	for i := range ids {
		off := baseOff + int64(i*vecBytes)
		res, _ := tx.Exec("INSERT OR REPLACE INTO chunks (id, text, filename, offset) VALUES (?, ?, ?, ?)", ids[i], texts[i], filenames[i], off)
		rowid, _ := res.LastInsertId()
		tx.Exec("INSERT INTO fts_chunks(rowid, text) VALUES(?, ?)", rowid, texts[i])
	}
	if err := tx.Commit(); err != nil {
		return err
	}

	// 3. Append binaire des vecteurs (persistant au restart)
	buf := make([]byte, 0, len(ids)*vecBytes)
	tmp := make([]byte, 4)
	for _, v := range vecs {
		for _, f := range v {
			binary.LittleEndian.PutUint32(tmp, math.Float32bits(f))
			buf = append(buf, tmp...)
		}
	}
	if _, err := s.vecs.WriteAt(buf, baseOff); err != nil {
		return fmt.Errorf("InsertBatch: écriture vectors.bin: %w", err)
	}

	// 4. Cache RAM + HNSW
	var nodes []hnsw.Node[string]
	for i := range ids {
		s.vecCache.set(ids[i], vecs[i])
		nodes = append(nodes, hnsw.MakeNode(ids[i], vecs[i]))
	}
	s.hnswIdx.Add(nodes...)
	s.hnswDirty = true

	// ==========================================
	return nil
}

func (s *Store) SearchSmart(query string, queryVec []float32, k int) ([]Chunk, error) {
	fetchSize := k * 12 // pool large pour nourrir RRF + reranker avec plus de données
	if fetchSize < 40 {
		fetchSize = 40
	}

	vecHits, err := s.searchVector(queryVec, fetchSize)
	if err != nil {
		fmt.Printf("⚠️  SearchSmart: searchVector failed: %v\n", err)
		vecHits = nil
	}
	// Double passe FTS : query brute + query expansée (sans stopwords, termes techniques)
	// → plus de rappel sur les formulations étudiant ("c'est quoi", "pourquoi ça marche pas")
	keywordHits, err := s.fullTextSearch(query, fetchSize)
	if err != nil {
		fmt.Printf("⚠️  SearchSmart: fullTextSearch failed: %v\n", err)
		keywordHits = nil
	}
	if expanded := expandQuery(query, s.stopWordsMap); expanded != "" && expanded != query {
		if more, err := s.fullTextSearch(expanded, fetchSize/2); err == nil && len(more) > 0 {
			keywordHits = append(keywordHits, more...)
		}
	}
	if vecHits == nil && keywordHits == nil {
		return nil, fmt.Errorf("SearchSmart: both search paths failed for query %q", query)
	}

	chunkMap := make(map[string]*Chunk)
	rrfScores := make(map[string]float32)

	// Détection de query technique : si la query contient des termes
	// spécifiques (registres, acronymes, sigles), on booste fortement le FTS
	// car bge-m3 a un gap sémantique avec le contenu technique anglais/tabulaire
	ftsWeight := s.ftsWeight
	if s.isTechnicalQuery(query) {
		ftsWeight = s.ftsTechWeight // FTS dominante sur les questions registres/acronymes
	}

	vecWeight := float32(1.0)
	if s.isTechnicalQuery(query) {
		vecWeight = 0.0 // query technique : FTS5 uniquement, vectoriel pollue
	}
	if vecWeight > 0 {
		for rank, hit := range vecHits {
			chunkMap[hit.ID] = &vecHits[rank]
			rrfScores[hit.ID] += vecWeight / (s.rrfConstant + float32(rank+1))
		}
	}
	for rank, hit := range keywordHits {
		if _, exists := chunkMap[hit.ID]; !exists {
			chunkMap[hit.ID] = &keywordHits[rank]
		}
		rrfScores[hit.ID] += ftsWeight / (s.rrfConstant + float32(rank+1))
	}

	var maxScore float32 = 0.0001
	for _, score := range rrfScores {
		if score > maxScore {
			maxScore = score
		}
	}

	var candidates []Chunk
	for id, chunk := range chunkMap {
		chunk.RRFRaw = rrfScores[id]
		chunk.Score = rrfScores[id] / maxScore
		candidates = append(candidates, *chunk)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	rerankPool := s.rerankPool
	if rerankPool <= 0 {
		rerankPool = 12 // fallback si mal initialisé
	}
	if len(candidates) < rerankPool {
		rerankPool = len(candidates)
	}
	top := candidates[:rerankPool]

	if reranked, err := s.rerankChunks(query, top); err == nil {
		if len(reranked) > 0 {
			// Normalisation min-max des scores du re-ranker (Cross-Encoder)
			// On utilise une approche qui préserve la notion de pertinence absolue :
			// si le meilleur score est trop bas, on ne sature pas à 1.0.
			best := reranked[0].Score
			worst := reranked[len(reranked)-1].Score
			span := best - worst

			// Seuil empirique : si le meilleur score Cross-Encoder est < -2 (très faible)
			// on pénalise le score final pour éviter les faux positifs.
			absScale := float32(1.0)
			if best < -2.0 {
				absScale = 0.5
			}

			for i := range reranked {
				if span > 0.0001 {
					reranked[i].Score = ((reranked[i].Score - worst) / span) * absScale
				} else {
					reranked[i].Score = 0.5 * absScale
				}
			}
			// Diversité : MMR + plafond par fichier → évite que 8 chunks du même
			// gros PDF noient les petits fichiers pertinents quand on augmente k
			reranked = s.diversify(reranked, queryVec, k)
			return reranked, nil
		}
	}

	top = s.diversify(top, queryVec, k)
	return top, nil
}

// expandQuery retourne la query sans stopwords (2e passe FTS pour plus de rappel).
func expandQuery(q string, stop map[string]bool) string {
	var kept []string
	for _, w := range strings.Fields(q) {
		clean := strings.Trim(w, ".,;:!?\"'()[]")
		if len([]rune(clean)) <= 2 {
			continue
		}
		if stop[strings.ToLower(clean)] {
			continue
		}
		kept = append(kept, w)
	}
	if len(kept) == 0 {
		return ""
	}
	return strings.Join(kept, " ")
}

// diversify applique MMR (si vecteurs dispo) + plafond par fichier.
func (s *Store) diversify(chunks []Chunk, queryVec []float32, k int) []Chunk {
	if len(chunks) == 0 || k <= 0 {
		return nil
	}
	maxPer := s.maxPerFile
	if maxPer <= 0 {
		maxPer = 2 // défaut : évite la noyade par un seul gros fichier
	}
	lambda := s.mmrLambda
	if lambda <= 0 {
		lambda = 0.7
	}
	useMMR := lambda < 1.0 && len(queryVec) > 0

	var out []Chunk
	perFile := map[string]int{}
	// Pré-charge les vecteurs pour MMR
	vecs := make([][]float32, len(chunks))
	for i, c := range chunks {
		if v, ok := s.vecCache.get(c.ID); ok {
			vecs[i] = v
		}
	}
	remaining := make([]int, len(chunks))
	for i := range chunks {
		remaining[i] = i
	}
	selected := make([][]float32, 0, k)

	for len(out) < k && len(remaining) > 0 {
		bestIdx := -1
		var bestVal float32 = -1e30
		for pos, ci := range remaining {
			rel := chunks[ci].Score
			pen := float32(0)
			if useMMR && vecs[ci] != nil && len(selected) > 0 {
				maxSim := float32(-1)
				for _, sv := range selected {
					if sim := cosineSimilarity(vecs[ci], sv); sim > maxSim {
						maxSim = sim
					}
				}
				pen = maxSim
			}
			val := lambda*rel - (1-lambda)*pen
			// Pénalité douce si le fichier est déjà bien représenté
			if perFile[chunks[ci].Filename] >= maxPer {
				val -= 0.5
			}
			if val > bestVal {
				bestVal = val
				bestIdx = pos
			}
		}
		ci := remaining[bestIdx]
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
		// Plafond strict : on saute le chunk si le fichier a déjà maxPer*2,
		// mais on ne bloque pas indéfiniment si tout vient du même fichier
		if perFile[chunks[ci].Filename] >= maxPer*2 && len(out)+len(remaining) >= k {
			continue
		}
		out = append(out, chunks[ci])
		perFile[chunks[ci].Filename]++
		if vecs[ci] != nil {
			selected = append(selected, vecs[ci])
		}
	}
	return out
}

// Clear supprime tous les chunks, index FTS, vecteurs et HNSW (reindex propre).
func (s *Store) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.db.Exec("DELETE FROM chunks"); err != nil {
		return err
	}
	if _, err := s.db.Exec("DELETE FROM fts_chunks"); err != nil {
		return err
	}
	if _, err := s.db.Exec("DELETE FROM meta WHERE key='vec_dim'"); err != nil {
		return err
	}
	// Tronque vectors.bin
	if err := s.vecs.Truncate(0); err != nil {
		return err
	}
	s.vecCache = newLRUCache(s.vecCache.cap)
	s.hnswIdx = hnsw.NewGraph[string]()
	s.hnswIdx.EfSearch = 100
	s.vecDim = 0
	s.hnswDirty = true
	s.saveHNSW(s.hnswIdx, s.hnswPath)
	s.hnswDirty = false
	return nil
}

// VecDim retourne la dimension des vecteurs indexés (0 = base vide).
func (s *Store) VecDim() int { return s.vecDim }

func (s *Store) rerankChunks(query string, chunks []Chunk) ([]Chunk, error) {
	type rerankChunk struct {
		ID    string  `json:"id"`
		Text  string  `json:"text"`
		File  string  `json:"filename"`
		Score float32 `json:"rerank_score"`
		RRF   float32 `json:"rrf_raw"`
	}
	type req struct {
		Query  string        `json:"query"`
		Chunks []rerankChunk `json:"chunks"`
	}

	input := req{Query: query}
	for _, c := range chunks {
		input.Chunks = append(input.Chunks, rerankChunk{
			ID: c.ID, Text: c.Text, File: c.Filename, RRF: c.RRFRaw,
		})
	}

	body, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("reranker marshal: %w", err)
	}

	timeout := time.Duration(s.rerankerTimeoutMs) * time.Millisecond
	if s.httpClient.Timeout != timeout {
		// Mise à jour si le timeout a changé (cas rare)
		s.httpClient.Timeout = timeout
	}
	resp, err := s.httpClient.Post(s.rerankerURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("reranker unreachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("reranker returned HTTP %d", resp.StatusCode)
	}

	var result []rerankChunk
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("reranker decode: %w", err)
	}

	var out []Chunk
	for _, r := range result {
		out = append(out, Chunk{
			ID: r.ID, Text: r.Text, Filename: r.File,
			Score: r.Score, RRFRaw: r.RRF,
		})
	}
	return out, nil
}

// searchVector choisit automatiquement entre full scan (exact, O(n)) et HNSW (approx, O(log n))
// selon le nombre de vecteurs en cache par rapport au seuil configuré.
// - En dessous du seuil : full scan — résultats exacts, latence négligeable sur petites bases
// - Au-dessus du seuil : HNSW — O(log n), précision légèrement réduite mais indispensable en charge
// Fallback sur full scan si l'index HNSW est vide (ex: première indexation).
func (s *Store) searchVector(queryVec []float32, k int) ([]Chunk, error) {
	n := s.vecCache.len()
	threshold := s.hnswThreshold
	if threshold <= 0 {
		threshold = 2000 // valeur de sécurité si StoreConfig mal initialisé
	}

	if n < threshold || s.hnswIdx == nil || s.hnswIdx.Len() == 0 {
		return s.searchVectorFallback(queryVec, k)
	}

	// Recherche HNSW — O(log n)
	neighbors := s.hnswIdx.Search(queryVec, k)
	if len(neighbors) == 0 {
		return s.searchVectorFallback(queryVec, k)
	}

	// Récupère les métadonnées depuis SQLite pour chaque ID retourné par HNSW
	results := make([]Chunk, 0, len(neighbors))
	for _, node := range neighbors {
		id := node.Key
		var c Chunk
		err := s.db.QueryRow("SELECT id, text, filename FROM chunks WHERE id = ?", id).
			Scan(&c.ID, &c.Text, &c.Filename)
		if err != nil {
			continue
		}
		vec, ok := s.vecCache.get(id)
		if !ok {
			continue
		}
		c.Score = cosineSimilarity(queryVec, vec)
		results = append(results, c)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return results, nil
}

// searchVectorFallback : full scan O(n) — utilisé uniquement si HNSW indisponible.
func (s *Store) searchVectorFallback(queryVec []float32, k int) ([]Chunk, error) {
	dim := s.vecDim
	if dim <= 0 {
		dim = len(queryVec)
	}
	if dim <= 0 {
		dim = defaultVecDim
	}
	vecBytes := dim * 4
	rows, err := s.db.Query("SELECT id, text, filename, offset FROM chunks")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Chunk
	for rows.Next() {
		var c Chunk
		var offset sql.NullInt64
		if err := rows.Scan(&c.ID, &c.Text, &c.Filename, &offset); err != nil {
			continue
		}

		vec, ok := s.vecCache.get(c.ID)
		if !ok {
			if !offset.Valid {
				continue // chunk pré-fix sans vecteur : ignoré côté vectoriel (FTS prend le relais)
			}
			buf := make([]byte, vecBytes)
			if _, err := s.vecs.ReadAt(buf, offset.Int64); err != nil {
				continue
			}
			vec = make([]float32, dim)
			for i := 0; i < dim; i++ {
				vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
			}
			s.vecCache.set(c.ID, vec)
		}

		c.Score = cosineSimilarity(queryVec, vec)
		results = append(results, c)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > k {
		results = results[:k]
	}
	return results, nil
}

func (s *Store) fullTextSearch(query string, k int) ([]Chunk, error) {
	words := strings.Fields(query)

	// Recherche LIKE directe pour les tokens contenant de la ponctuation interne
	// (ex: "802.1X", "I²C", "RS-232") que FTS5 tokenise incorrectement.
	// On tente cette passe en premier — si elle retourne des résultats, on s'arrête.
	var punctTokens []string
	for _, w := range words {
		clean := strings.Trim(w, ".,;:!?\"'()[]")
		if len([]rune(clean)) < 3 {
			continue
		}
		// Heuristique : token avec chiffre+lettre+ponctuation interne → terme technique
		hasInternalPunct := strings.ContainsAny(clean, ".-/") &&
			len([]rune(clean)) >= 4
		if hasInternalPunct {
			punctTokens = append(punctTokens, clean)
		}
	}
	if len(punctTokens) > 0 {
		if results, err := s.likeSearch(punctTokens, k); err == nil && len(results) > 0 {
			return results, nil
		}
	}

	var techWords []string
	var validWords []string

	for _, w := range words {
		clean := strings.ReplaceAll(w, "\"", "\"\"")
		clean = strings.ReplaceAll(clean, "'", "")
		lower := strings.ToLower(clean)

		if len([]rune(clean)) <= 3 {
			continue
		}
		if s.stopWordsMap[lower] {
			continue
		}

		validWords = append(validWords, clean+"*")

		hasUpper := false
		hasDigit := false
		for _, r := range clean {
			if r >= 'A' && r <= 'Z' {
				hasUpper = true
			}
			if r >= '0' && r <= '9' {
				hasDigit = true
			}
		}

		if (hasUpper && len([]rune(clean)) >= 3) || hasDigit {
			techWords = append(techWords, clean+"*")
		}
	}

	// Ajoute les termes techniques reconnus (config) même en minuscules
	for _, w := range words {
		clean := strings.Trim(w, ".,;:!?\"'()[]")
		if len([]rune(clean)) < 3 {
			continue
		}
		lower := strings.ToLower(clean)
		for _, term := range s.searchRules.TechnicalTerms {
			if strings.TrimSpace(term) == lower {
				techWords = append(techWords, clean+"*")
				break
			}
		}
	}
	// Déduplique techWords
	seen := make(map[string]bool)
	var dedupTech []string
	for _, tw := range techWords {
		if !seen[tw] {
			seen[tw] = true
			dedupTech = append(dedupTech, tw)
		}
	}
	techWords = dedupTech

	if len(techWords) > 1 {
		if results, err := s.executeFTS(strings.Join(techWords, " AND "), k); err == nil && len(results) > 0 {
			return results, nil
		}
	}

	if len(techWords) > 0 {
		if results, err := s.executeFTS(strings.Join(techWords, " OR "), k); err == nil && len(results) > 0 {
			return results, nil
		}
	}

	if len(validWords) > 0 {
		if results, err := s.executeFTS(strings.Join(validWords, " OR "), k); err == nil && len(results) > 0 {
			return results, nil
		}
	}

	return nil, nil
}

func (s *Store) executeFTS(ftsQuery string, k int) ([]Chunk, error) {
	rows, err := s.db.Query(`
		SELECT chunks.id, chunks.text, chunks.filename, rank
		FROM fts_chunks
		JOIN chunks ON fts_chunks.rowid = chunks.rowid
		WHERE fts_chunks MATCH ?
		ORDER BY rank LIMIT ?`, ftsQuery, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var results []Chunk
	for rows.Next() {
		var c Chunk
		var rank float32
		if err := rows.Scan(&c.ID, &c.Text, &c.Filename, &rank); err == nil {
			if rank < 0 {
				c.Score = 1.0 / (1.0 + float32(math.Abs(float64(rank))))
			} else {
				c.Score = 0.1
			}
			results = append(results, c)
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return results, nil
}

// likeSearch : recherche SQL LIKE sur chunks.text pour les tokens
// que FTS5 tokenise incorrectement (ponctuation interne : 802.1X, RS-232, etc.)
// Retourne les k meilleurs résultats avec un score fixe de 0.8.
func (s *Store) likeSearch(tokens []string, k int) ([]Chunk, error) {
	if len(tokens) == 0 {
		return nil, nil
	}

	// Construit : WHERE text LIKE '%802.1X%' AND text LIKE '%autre%'
	baseQuery := "SELECT id, text, filename FROM chunks WHERE "
	args := make([]interface{}, 0, len(tokens)+1)
	clauses := make([]string, 0, len(tokens))
	for _, t := range tokens {
		clauses = append(clauses, "text LIKE ?")
		args = append(args, "%"+t+"%")
	}
	baseQuery += strings.Join(clauses, " AND ") + " LIMIT ?"
	args = append(args, k)

	rows, err := s.db.Query(baseQuery, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.Text, &c.Filename); err == nil {
			c.Score = 0.8 // score fixe élevé : match exact sur terme technique
			results = append(results, c)
		}
	}
	return results, nil
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, nA, nB float32
	for i := range a {
		dot += a[i] * b[i]
		nA += a[i] * a[i]
		nB += b[i] * b[i]
	}
	if nA == 0 || nB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(nA))) * float32(math.Sqrt(float64(nB))))
}

func (s *Store) GetDB() *sql.DB {
	return s.db
}

func (s *Store) Count() int {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM chunks").Scan(&count)
	if err != nil {
		return 0
	}
	return count
}

// isTechnicalQuery détecte si la query contient des termes techniques
// qui nécessitent un boost FTS (acronymes, noms de registres, termes électronique)
func (s *Store) isTechnicalQuery(q string) bool {
	lower := strings.ToLower(q)
	technicalTerms := s.searchRules.TechnicalTerms
	if len(technicalTerms) == 0 {
		// Fallback si la config est vide (ne devrait pas arriver avec defaults())
		technicalTerms = []string{
			"adcs", "admux", "tccr", "timsk", "portb", "ddrb", "ddrd",
			"prescaler", "registre", "register", "bit ", "0x", "pwm",
			"uart", "spi", "i2c", "twi", "isr", "int0", "int1",
			"adps", "aden", "adsc", "adie", "adif", "refs",
			"wgm", "com", "ocr", "icr", "tcnt",
			"vlan", "trunk", "switchport", "802.1x", "ddos", "slowloris", "radius",
		}
	}

	for _, term := range technicalTerms {
		if strings.Contains(lower, term) {
			return true
		}
	}
	return false
}
