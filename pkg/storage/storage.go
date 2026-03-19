package storage

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"fmt"
	"time"

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
	vecCache map[string][]float32 // Cache en mémoire : clé = chunk.ID (stable)
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

	s := &Store{
		db:       db,
		vecs:     vf,
		vecCache: make(map[string][]float32),
	}

	// Préchargement de tous les vecteurs en RAM avec clé = chunk.ID (stable)
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

	return s, nil
}

func (s *Store) Close() {
	s.db.Close()
	s.vecs.Close()
}

func (s *Store) InsertBatch(ids, texts, filenames []string, vecs [][]float32) error {
	// FIX : le mutex couvre TOUTE la transaction (SQLite + écriture binaire)
	// pour éviter la race condition sur l'offset vectors.bin
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	// Rollback ne fait rien si Commit a réussi — safe à appeler dans tous les cas
	defer tx.Rollback()

	// Calcul de l'offset SOUS le mutex — protégé contre les appels concurrents
	info, err := s.vecs.Stat()
	if err != nil {
		return fmt.Errorf("InsertBatch: stat vectors.bin: %w", err)
	}
	offset := info.Size()

	// Phase 1 : écriture des vecteurs dans vectors.bin
	// On écrit AVANT le commit SQL pour détecter les erreurs I/O avant de toucher la DB
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

	// Phase 2 : insertion SQLite (chunks + FTS) avec les offsets corrects
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

	// Phase 3 : commit SQL — si ça échoue, les vecteurs déjà écrits sont orphelins
	// (vectors.bin est append-only, ils seront ignorés sans entrée SQL correspondante)
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("InsertBatch: commit: %w", err)
	}

	// Phase 4 : mise à jour du cache RAM APRÈS le commit (cohérence garantie)
	for i := range ids {
		s.vecCache[ids[i]] = vecs[i]
	}

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

	// Timeout 500ms — si le reranker Python est down, on ne bloque pas le pipeline
	httpClient := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := httpClient.Post("http://127.0.0.1:8765", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("reranker unreachable: %w", err)
	}
	defer resp.Body.Close()

	// FIX : vérification du status HTTP avant de décoder
	// Sans ça, un 500 avec body JSON d'erreur corrompt silencieusement le pipeline
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

func (s *Store) searchVector(queryVec []float32, k int) ([]Chunk, error) {
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
			// Fallback disque (ne devrait pas arriver en prod)
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
		// Ajouts manquants
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
