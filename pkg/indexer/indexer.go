package indexer

import (
	"fmt"
	"sync"

	"rag-hybrid/pkg/chunker"
	"rag-hybrid/pkg/client"
	"rag-hybrid/pkg/storage"

	"github.com/google/uuid"
	"github.com/schollz/progressbar/v3"
)

type Indexer struct {
	client     *client.Client
	store      *storage.Store
	EmbedModel string // modèle d'embedding — depuis config.yaml
}

func New(c *client.Client, s *storage.Store, embedModel string) *Indexer {
	if embedModel == "" {
		embedModel = "nomic-embed-text:latest" // fallback
	}
	return &Indexer{client: c, store: s, EmbedModel: embedModel}
}

func (idx *Indexer) Index(dir string) error {
	fmt.Println("🔍 Scanning...")
	files, _ := chunker.ScanDir(dir)
	fmt.Printf("📂 %d files\n", len(files))

	var all []chunker.Chunk
	for _, f := range files {
		c, _ := chunker.ChunkFile(f, 1000, 200)
		all = append(all, c...)
	}
	fmt.Printf("✂️  %d chunks\n", len(all))

	// Taille de batch réduite à 32 pour coller à la limite Ollama /api/embed
	// mais les goroutines font 30 appels embed en parallèle
	// → chaque batch = 32 chunks × 30 goroutines = 960 chunks en vol simultané
	const embedBatch = 32
	const sqlBatch   = 500  // InsertBatch par tranche de 500 pour limiter la taille des transactions

	// Découpe en sous-batches pour l'embedding
	type embedResult struct {
		idx  int
		vecs [][]float32
		err  error
	}

	embedBatches := batchChunks(all, embedBatch)
	bar := progressbar.Default(int64(len(embedBatches)), "embedding")

	// Buffer de résultats pour reconstruire l'ordre
	results := make([][][]float32, len(embedBatches))
	var mu sync.Mutex
	_ = mu

	var wg sync.WaitGroup
	sem := make(chan struct{}, 30)
	errs := make([]error, len(embedBatches))

	for i, b := range embedBatches {
		wg.Add(1)
		sem <- struct{}{}
		go func(batchIdx int, chunks []chunker.Chunk) {
			defer wg.Done()
			defer func() { <-sem }()

			texts := make([]string, len(chunks))
			for j, c := range chunks {
				texts[j] = c.Text
			}
			vecs, err := idx.client.Embed(texts, idx.EmbedModel)
			if err != nil {
				errs[batchIdx] = err
				bar.Add(1)
				return
			}
			results[batchIdx] = vecs
			bar.Add(1)
		}(i, b)
	}
	wg.Wait()
	fmt.Println()

	// Compter les erreurs d'embedding
	var embedErrors int
	for _, e := range errs {
		if e != nil {
			embedErrors++
		}
	}
	if embedErrors > 0 {
		fmt.Printf("⚠️  %d batches d'embedding en erreur (ignorés)\n", embedErrors)
		// Affiche la première erreur pour diagnostiquer
		for _, e := range errs {
			if e != nil {
				fmt.Printf("   Première erreur : %v\n", e)
				break
			}
		}
	}

	// Reconstruction à plat dans l'ordre
	var (
		allIDs      []string
		allTexts    []string
		allFilenames []string
		allVecs     [][]float32
	)
	for i, b := range embedBatches {
		if results[i] == nil {
			continue
		}
		for j, v := range results[i] {
			if j >= len(b) {
				break
			}
			allIDs       = append(allIDs, uuid.New().String())
			allTexts     = append(allTexts, b[j].Text)
			allFilenames = append(allFilenames, b[j].Filename)
			allVecs      = append(allVecs, v)
		}
	}

	// Insertion en InsertBatch par tranches de sqlBatch
	fmt.Printf("💾 Insertion de %d chunks...\n", len(allIDs))
	insertBar := progressbar.Default(int64(len(allIDs)), "insertion")

	for start := 0; start < len(allIDs); start += sqlBatch {
		end := start + sqlBatch
		if end > len(allIDs) {
			end = len(allIDs)
		}
		if err := idx.store.InsertBatch(
			allIDs[start:end],
			allTexts[start:end],
			allFilenames[start:end],
			allVecs[start:end],
		); err != nil {
			fmt.Printf("⚠️  InsertBatch erreur (chunks %d-%d): %v\n", start, end, err)
		}
		insertBar.Add(end - start)
	}

	fmt.Println("\n✅ Done")
	return nil
}

func batchChunks(c []chunker.Chunk, size int) [][]chunker.Chunk {
	var batches [][]chunker.Chunk
	for i := 0; i < len(c); i += size {
		end := i + size
		if end > len(c) {
			end = len(c)
		}
		batches = append(batches, c[i:end])
	}
	return batches
}