package storage

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"bufio"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"fmt"
	"time"

	"github.com/coder/hnsw"
	_ "modernc.org/sqlite"
)

const vecDim = 768
const vecBytes = vecDim * 4

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
	vecCache map[string][]float32    // Cache RAM : clé = chunk.ID
	hnswIdx  *hnsw.Graph[string]     // Index ANN : O(log n) au lieu de O(n)
	hnswPath string                  // Chemin de persistance hnsw.bin
}

func New(base string) (*Store, error) {
	p := os.ExpandEnv(base)
	if p[0] == '~' {
		h, _ := os.UserHomeDir()
		p = filepath.Join(h, p[1:])
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
		db:       db,
		vecs:     vf,
		vecCache: make(map[string][]float32),
		hnswPath: hnswPath,
	}

	// --- Chargement des vecteurs en RAM ---
	info, err := vf.Stat()
	if err == nil && info.Size() > 0 {
		data := make([]byte, info.Size())
		vf.ReadAt(data, 0)
		rows, rerr := db.Query("SELECT id, offset FROM chunks ORDER BY offset")
		if rerr == nil {
			defer rows.Close()
			for rows.Next() {
				var id string
				var offset int64
				if rows.Scan(&id, &offset) == nil && offset+int64(vecBytes) <= int64(len(data)) {
					vec := make([]float32, vecDim)
					for i := 0; i < vecDim; i++ {
						vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset+int64(i*4):]))
					}
					s.vecCache[id] = vec
				}
			}
		}
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
		// Import nécessite un io.ByteReader — os.File ne l'implémente pas
		// sans bufio, binary.Read retourne "does not implement io.ByteReader"
		if err := g.Import(bufio.NewReader(f)); err == nil {
			fmt.Printf("✅ Index HNSW chargé (%d vecteurs)\n", len(s.vecCache))
			return g
		}
		fmt.Printf("⚠️  hnsw.bin corrompu — reconstruction depuis vecCache\n")
	}

	// Reconstruction depuis le cache RAM
	g := hnsw.NewGraph[string]()
	if len(s.vecCache) == 0 {
		return g
	}

	var nodes []hnsw.Node[string]
	for id, vec := range s.vecCache {
		nodes = append(nodes, hnsw.MakeNode(id, vec))
	}
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
	s.db.Close()
	s.vecs.Close()
}

func (s *Store) InsertBatch(ids, texts, filenames []string, vecs [][]float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	info, err := s.vecs.Stat()
	if err != nil {
		return fmt.Errorf("InsertBatch: stat vectors.bin: %w", err)
	}
	offset := info.Size()

	// Phase 1 : écriture des vecteurs binaires
	vecOffsets := make([]int64, len(ids))
	for i := range ids {
		vecOffsets[i] = offset
		buf := make([]byte, vecBytes)
		for j, f := range vecs[i] {
			binary.LittleEndian.PutUint32(buf[j*4:], math.Float32bits(f))
		}
		if _, err := s.vecs.WriteAt(buf, offset); err != nil {
			return fmt.Errorf("InsertBatch: write vec[%d]: %w", i, err)
		}
		offset += int64(vecBytes)
	}

	// Phase 2 : insertion SQLite
	for i := range ids {
		_, err = tx.Exec("INSERT INTO chunks (id, text, filename, offset) VALUES (?, ?, ?, ?)",
			ids[i], texts[i], filenames[i], vecOffsets[i])
		if err != nil {
			return fmt.Errorf("InsertBatch: insert chunk[%d]: %w", i, err)
		}
	}

	stmt, err := tx.Prepare("INSERT INTO fts_chunks(rowid, text) VALUES(?, ?)")
	if err != nil {
		return fmt.Errorf("InsertBatch: prepare FTS: %w", err)
	}
	defer stmt.Close()

	for i := range ids {
		var rowid int64
		err = tx.QueryRow("SELECT rowid FROM chunks WHERE id = ?", ids[i]).Scan(&rowid)
		if err != nil {
			return fmt.Errorf("InsertBatch: get rowid[%d]: %w", i, err)
		}
		if _, err = stmt.Exec(rowid, texts[i]); err != nil {
			return fmt.Errorf("InsertBatch: insert FTS[%d]: %w", i, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("InsertBatch: commit: %w", err)
	}

	// Phase 3 : mise à jour cache RAM + index HNSW (après commit SQL)
	var newNodes []hnsw.Node[string]
	for i := range ids {
		s.vecCache[ids[i]] = vecs[i]
		newNodes = append(newNodes, hnsw.MakeNode(ids[i], vecs[i]))
	}
	s.hnswIdx.Add(newNodes...)

	// Persistance de l'index HNSW après chaque batch
	s.saveHNSW(s.hnswIdx, s.hnswPath)

	return nil
}

func (s *Store) SearchSmart(query string, queryVec []float32, k int) ([]Chunk, error) {
	fetchSize := k * 5

	vecHits, _ := s.searchVector(queryVec, fetchSize)
	keywordHits, _ := s.fullTextSearch(query, fetchSize)

	chunkMap := make(map[string]*Chunk)
	rrfScores := make(map[string]float32)

	const rrfConstant = 60.0

	for rank, hit := range vecHits {
		chunkMap[hit.ID] = &vecHits[rank]
		rrfScores[hit.ID] += 1.0 / (rrfConstant + float32(rank+1))
	}
	for rank, hit := range keywordHits {
		if _, exists := chunkMap[hit.ID]; !exists {
			chunkMap[hit.ID] = &keywordHits[rank]
		}
		rrfScores[hit.ID] += 1.0 / (rrfConstant + float32(rank+1))
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

	rerankPool := 6
	if len(candidates) < rerankPool {
		rerankPool = len(candidates)
	}
	top := candidates[:rerankPool]

	if reranked, err := rerankChunks(query, top); err == nil {
		if len(reranked) > 0 {
			best := reranked[0].Score
			worst := reranked[len(reranked)-1].Score
			span := best - worst
			for i := range reranked {
				if span > 0 {
					reranked[i].Score = (reranked[i].Score - worst) / span
				} else {
					reranked[i].Score = 1.0
				}
			}
			if len(reranked) > k {
				reranked = reranked[:k]
			}
			return reranked, nil
		}
	}

	if len(top) > k {
		top = top[:k]
	}
	return top, nil
}

func rerankChunks(query string, chunks []Chunk) ([]Chunk, error) {
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

	httpClient := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := httpClient.Post("http://127.0.0.1:8765", "application/json", bytes.NewReader(body))
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

// searchVector utilise l'index HNSW pour une recherche ANN en O(log n).
// Remplace l'ancien full scan O(n) sur tous les chunks.
// Fallback sur le full scan si l'index est vide (base fraîchement créée).
func (s *Store) searchVector(queryVec []float32, k int) ([]Chunk, error) {
	// Fallback full scan si HNSW vide (ex: première indexation en cours)
	if s.hnswIdx == nil || s.hnswIdx.Len() == 0 {
		return s.searchVectorFallback(queryVec, k)
	}

	// Recherche ANN : retourne les k plus proches voisins approximatifs
	neighbors := s.hnswIdx.Search(queryVec, k)

	if len(neighbors) == 0 {
		return s.searchVectorFallback(queryVec, k)
	}

	// Récupération des métadonnées depuis SQLite par IDs
	if len(neighbors) == 0 {
		return nil, nil
	}

	// Construction de la clause IN pour la requête SQL
	placeholders := make([]string, len(neighbors))
	args := make([]interface{}, len(neighbors))
	for i, n := range neighbors {
		placeholders[i] = "?"
		args[i] = n.Key
	}

	query := fmt.Sprintf(
		"SELECT id, text, filename FROM chunks WHERE id IN (%s)",
		strings.Join(placeholders, ","),
	)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Map id → chunk pour associer les scores cosine
	chunkByID := make(map[string]Chunk)
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.Text, &c.Filename); err != nil {
			continue
		}
		chunkByID[c.ID] = c
	}

	// Calcul du score cosine réel pour chaque voisin HNSW
	var results []Chunk
	for _, n := range neighbors {
		c, ok := chunkByID[n.Key]
		if !ok {
			continue
		}
		if vec, ok := s.vecCache[c.ID]; ok {
			c.Score = cosineSimilarity(queryVec, vec)
		}
		results = append(results, c)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// searchVectorFallback : full scan O(n) — utilisé uniquement si HNSW indisponible.
func (s *Store) searchVectorFallback(queryVec []float32, k int) ([]Chunk, error) {
	rows, err := s.db.Query("SELECT id, text, filename, offset FROM chunks")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Chunk
	for rows.Next() {
		var c Chunk
		var offset int64
		if err := rows.Scan(&c.ID, &c.Text, &c.Filename, &offset); err != nil {
			continue
		}

		vec, ok := s.vecCache[c.ID]
		if !ok {
			buf := make([]byte, vecBytes)
			s.vecs.ReadAt(buf, offset)
			vec = make([]float32, vecDim)
			for i := 0; i < vecDim; i++ {
				vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
			}
			s.vecCache[c.ID] = vec
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

	frenchStopWords := map[string]bool{
		"comment": true, "fonctionne": true, "fonctionner": true,
		"quels": true, "quelle": true, "quelles": true, "quel": true,
		"quand": true, "pourquoi": true, "expliquer": true, "expliquez": true,
		"utiliser": true, "utilise": true, "donner": true, "faire": true,
		"avec": true, "dans": true, "pour": true, "vers": true,
		"depuis": true, "entre": true, "comme": true, "plus": true,
		"moins": true, "bien": true, "tout": true, "cette": true,
		"sont": true, "mais": true, "donc": true, "aussi": true,
		"même": true, "leur": true, "quoi": true, "être": true,
		"est": true, "les": true, "des": true, "une": true,
		"qui": true, "que": true, "sur": true, "par": true,
		"aux": true, "the": true, "and": true, "for": true,
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
		if frenchStopWords[lower] {
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
