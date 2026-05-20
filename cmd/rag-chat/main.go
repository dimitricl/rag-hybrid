package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"rag-hybrid/pkg/client"
	"rag-hybrid/pkg/config"
	"rag-hybrid/pkg/search"
	"rag-hybrid/pkg/storage"
)

func main() {
	cfg := config.Load()

	cl := client.New(cfg.Ollama.Host, cfg.Ollama.Port)
	if err := cl.Health(); err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  Ollama inaccessible (%s): %v\n", cfg.OllamaURL(), err)
		os.Exit(1)
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
		fmt.Fprintf(os.Stderr, "erreur store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	engine := search.NewWithConfig(cl, store, cfg.RAG)
	model := cfg.RAG.DefaultModel

	fmt.Printf("rag-chat — modèle: %s | %d chunks indexés\n", model, store.Count())
	fmt.Println("Commandes: !model <nom>  !files  !stats  !help  exit")
	fmt.Println(strings.Repeat("─", 60))

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case line == "exit" || line == "quit":
			fmt.Println("Au revoir.")
			return

		case line == "!help":
			fmt.Println("!model <nom>   — changer de modèle LLM")
			fmt.Println("!files         — lister les fichiers indexés")
			fmt.Println("!stats         — statistiques de la base")
			fmt.Println("exit           — quitter")

		case strings.HasPrefix(line, "!model "):
			model = strings.TrimPrefix(line, "!model ")
			fmt.Printf("Modèle → %s\n", model)

		case line == "!stats":
			fmt.Printf("Chunks indexés : %d\n", store.Count())

		case line == "!files":
			db := store.GetDB()
			rows, err := db.Query(`SELECT DISTINCT filename FROM chunks ORDER BY filename`)
			if err != nil {
				fmt.Println("erreur:", err)
				continue
			}
			for rows.Next() {
				var f string
				rows.Scan(&f)
				fmt.Println(" •", f)
			}
			rows.Close()

		default:
			ch, err := engine.AskStreamWithModel(context.Background(), line, model)
			if err != nil {
				fmt.Println("erreur:", err)
				continue
			}
			for tok := range ch {
				fmt.Print(tok)
			}
			fmt.Println()
		}
	}
}
