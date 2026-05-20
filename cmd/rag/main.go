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
				Action:    runIndex,
			},
		},
	}
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runIndex(c *cli.Context) error {
	if c.NArg() < 1 {
		return fmt.Errorf("usage: rag index <dossier>")
	}
	dir := c.Args().First()

	cfg := config.Load()

	cl := client.New(cfg.Ollama.Host, cfg.Ollama.Port)
	if err := cl.Health(); err != nil {
		return fmt.Errorf("Ollama inaccessible (%s): %w", cfg.OllamaURL(), err)
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
		return fmt.Errorf("ouverture store: %w", err)
	}
	defer store.Close()

	idx := indexer.NewWithConfig(cl, store, cfg.RAG)
	return idx.Index(dir)
}
