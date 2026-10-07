package main

import (
	"fmt"
	"os"

	"rag-hybrid/pkg/client"
	"rag-hybrid/pkg/config"
	"rag-hybrid/pkg/indexer"
	"rag-hybrid/pkg/storage"

	"github.com/urfave/cli/v2"
)

func main() {
	app := &cli.App{
		Name:  "rag",
		Usage: "Indexation de documents pour RAG hybride",
		Commands: []*cli.Command{
			{
				Name:      "index",
				Usage:     "Indexer un dossier de documents",
				ArgsUsage: "<dossier>",
				Flags: []cli.Flag{
					&cli.BoolFlag{Name: "clear", Usage: "Vide la base avant d'indexer (reindex propre, requis après fix vecteurs)"},
				},
				Action: runIndex,
			},
			{
				Name:   "stats",
				Usage:  "Statistiques de la base (chunks, vecteurs, dimension)",
				Action: runStats,
			},
			{
				Name:   "clear",
				Usage:  "Vide la base (chunks + vecteurs + HNSW)",
				Action: runClear,
			},
		},
	}
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func openStore(cfg config.Config) (*storage.Store, error) {
	return storage.New(cfg.RAG.DBPath, storage.StoreConfig{
		RRFConstant:       cfg.RAG.RRFConstant,
		FTSWeight:         cfg.RAG.FTSWeight,
		FTSTechWeight:     cfg.RAG.FTSTechWeight,
		RerankerTimeoutMs: cfg.RAG.RerankerTimeoutMs,
		RerankerURL:       cfg.RerankerURL(),
		RerankPool:        cfg.RAG.RerankPool,
		HNSWThreshold:     cfg.RAG.HNSWThreshold,
		VecCacheSize:      cfg.RAG.VecCacheSize,
		MaxPerFile:        cfg.RAG.MaxPerFile,
		MMRLambda:         cfg.RAG.MMRLambda,
		SearchRules:       cfg.SearchRules,
	})
}

func runIndex(c *cli.Context) error {
	if c.NArg() < 1 {
		return fmt.Errorf("usage: rag index [--clear] <dossier>")
	}
	dir := c.Args().First()

	cfg := config.Load()

	cl := client.New(cfg.Ollama.Host, cfg.Ollama.Port)
	if err := cl.Health(); err != nil {
		return fmt.Errorf("Ollama inaccessible (%s): %w", cfg.OllamaURL(), err)
	}

	store, err := openStore(cfg)
	if err != nil {
		return fmt.Errorf("ouverture store: %w", err)
	}
	defer store.Close()

	if c.Bool("clear") {
		fmt.Println("🧹 Base vidée (reindex propre)")
		if err := store.Clear(); err != nil {
			return fmt.Errorf("clear: %w", err)
		}
	}

	idx := indexer.NewWithConfig(cl, store, cfg.RAG)
	return idx.Index(dir)
}

func runStats(c *cli.Context) error {
	cfg := config.Load()
	store, err := openStore(cfg)
	if err != nil {
		return err
	}
	defer store.Close()
	db := store.GetDB()
	var total, withVec int
	_ = db.QueryRow("SELECT COUNT(*) FROM chunks").Scan(&total)
	_ = db.QueryRow("SELECT COUNT(*) FROM chunks WHERE offset IS NOT NULL").Scan(&withVec)
	var files int
	_ = db.QueryRow("SELECT COUNT(DISTINCT filename) FROM chunks").Scan(&files)
	fmt.Printf("Chunks   : %d (%d fichiers)\n", total, files)
	fmt.Printf("Vecteurs : %d (dim=%d)\n", withVec, store.VecDim())
	fmt.Printf("Contexte : %d chunks / rerank_pool=%d / max_per_file=%d / mmr=%.2f\n",
		cfg.RAG.ContextChunks, cfg.RAG.RerankPool, cfg.RAG.MaxPerFile, cfg.RAG.MMRLambda)
	if total > 0 && withVec == 0 {
		fmt.Println("⚠️  Base pré-fix sans vecteurs → relancez : ./rag index --clear <dossier>")
	}
	return nil
}

func runClear(c *cli.Context) error {
	cfg := config.Load()
	store, err := openStore(cfg)
	if err != nil {
		return err
	}
	defer store.Close()
	return store.Clear()
}
