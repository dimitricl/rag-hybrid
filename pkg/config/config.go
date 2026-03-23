package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
	"strings"
)

type OllamaConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

type RerankerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

type WebConfig struct {
	Port int `yaml:"port"`
}

type RAGConfig struct {
	DBPath               string   `yaml:"db_path"`
	EmbedModel           string   `yaml:"embed_model"`
	DefaultModel         string   `yaml:"default_model"`
	RerankPool           int      `yaml:"rerank_pool"`           // Nombre de chunks envoyés au reranker
	ContextChunks        int      `yaml:"context_chunks"`        // Nombre de chunks finaux envoyés au LLM (défaut 3)
	ChunkSize            int      `yaml:"chunk_size"`            // Taille des chunks en runes (défaut 1500)
	ChunkOverlap         int      `yaml:"chunk_overlap"`         // Overlap entre chunks en runes (défaut 150)
	MaxChunksPerFile     int      `yaml:"max_chunks_per_file"`   // Limite chunks par fichier (0=illimité, défaut 50)
	MinScore             float32  `yaml:"min_score"`             // Seuil de score pour inclusion dans le contexte
	RRFConstant          float32  `yaml:"rrf_constant"`          // Constante RRF (défaut 60.0)
	FTSWeight            float32  `yaml:"fts_weight"`            // Poids FTS pour queries normales (défaut 2.0)
	FTSTechWeight        float32  `yaml:"fts_tech_weight"`       // Poids FTS pour queries techniques (défaut 5.0)
	RerankerTimeoutMs    int      `yaml:"reranker_timeout_ms"`   // Timeout HTTP vers reranker en ms (défaut 500)
	HNSWThreshold        int      `yaml:"hnsw_threshold"`        // Nb de chunks au-dessus duquel HNSW remplace le full scan (défaut 2000)
	VecCacheSize         int      `yaml:"vec_cache_size"`        // Capacité max du cache LRU vecteurs (0 = illimité, défaut 10000)
	CoreFilePatterns     []string `yaml:"core_file_patterns"`    // Fichiers ignorant la limite max_chunks_per_file
	IgnoredDirs          []string `yaml:"ignored_dirs"`          // Répertoires ignorés au scan
	IgnoredFilePatterns  []string `yaml:"ignored_file_patterns"` // Sous-chaînes de noms de fichiers à ignorer
}

type Config struct {
	Ollama   OllamaConfig   `yaml:"ollama"`
	Reranker RerankerConfig `yaml:"reranker"`
	Web      WebConfig      `yaml:"web"`
	RAG      RAGConfig      `yaml:"rag"`
}

// Defaults appliqués si le fichier est absent ou incomplet
func defaults() Config {
	return Config{
		Ollama:   OllamaConfig{Host: "100.101.108.111", Port: 11434},
		Reranker: RerankerConfig{Host: "127.0.0.1", Port: 8765},
		Web:      WebConfig{Port: 8080},
		RAG: RAGConfig{
			DBPath:            "~/.rag-hybrid",
			EmbedModel:        "nomic-embed-text:latest",
			DefaultModel:      "mistral:7b-instruct",
			RerankPool:        6,
			ContextChunks:     3,
			ChunkSize:         1500,
			ChunkOverlap:      150,
			MaxChunksPerFile:  50,
			MinScore:          0.30,
			RRFConstant:       60.0,
			FTSWeight:         2.0,
			FTSTechWeight:     5.0,
			RerankerTimeoutMs: 500,
			HNSWThreshold:     2000,
			VecCacheSize:      10000,
			CoreFilePatterns:  []string{},
			IgnoredDirs:       []string{"node_modules", "__pycache__", ".git"},
			IgnoredFilePatterns: []string{},
		},
	}
}

// Load charge la config depuis configs/config.yaml relatif au binaire,
// ou depuis ~/.rag-hybrid/config.yaml en fallback.
// Si aucun fichier trouvé, retourne les valeurs par défaut sans erreur.
//
// Override par variable d'environnement (priorité sur le fichier) :
//   OLLAMA_HOST  → ollama.host
//   OLLAMA_PORT  → ollama.port  (parsé en int, ignoré si invalide)
func Load() Config {
	cfg := defaults()

	candidates := []string{
		"configs/config.yaml",
		filepath.Join(os.Getenv("HOME"), ".rag-hybrid", "config.yaml"),
	}

	for _, path := range candidates {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			fmt.Printf("⚠️  config.yaml invalide (%s): %v — valeurs par défaut utilisées\n", path, err)
			return defaults()
		}
		// Complète les champs vides avec les defaults
		if cfg.Ollama.Host == ""    { cfg.Ollama.Host = defaults().Ollama.Host }
		if cfg.Ollama.Port == 0     { cfg.Ollama.Port = defaults().Ollama.Port }
		if cfg.Reranker.Host == ""  { cfg.Reranker.Host = defaults().Reranker.Host }
		if cfg.Reranker.Port == 0   { cfg.Reranker.Port = defaults().Reranker.Port }
		if cfg.Web.Port == 0        { cfg.Web.Port = defaults().Web.Port }
		if cfg.RAG.DBPath == ""     { cfg.RAG.DBPath = defaults().RAG.DBPath }
		if cfg.RAG.EmbedModel == "" { cfg.RAG.EmbedModel = defaults().RAG.EmbedModel }
		if cfg.RAG.DefaultModel == "" { cfg.RAG.DefaultModel = defaults().RAG.DefaultModel }
		if cfg.RAG.RerankPool == 0  { cfg.RAG.RerankPool = defaults().RAG.RerankPool }
		if cfg.RAG.ContextChunks == 0 { cfg.RAG.ContextChunks = defaults().RAG.ContextChunks }
		if cfg.RAG.ChunkSize == 0   { cfg.RAG.ChunkSize = defaults().RAG.ChunkSize }
		if cfg.RAG.ChunkOverlap == 0 { cfg.RAG.ChunkOverlap = defaults().RAG.ChunkOverlap }
		if cfg.RAG.MaxChunksPerFile == 0 { cfg.RAG.MaxChunksPerFile = defaults().RAG.MaxChunksPerFile }
		// MinScore à 0 est une valeur intentionnellement valide (tout passe),
		// on applique le défaut seulement si négatif (valeur aberrante)
		if cfg.RAG.MinScore < 0     { cfg.RAG.MinScore = defaults().RAG.MinScore }
		if cfg.RAG.RRFConstant == 0      { cfg.RAG.RRFConstant = defaults().RAG.RRFConstant }
		if cfg.RAG.FTSWeight == 0        { cfg.RAG.FTSWeight = defaults().RAG.FTSWeight }
		if cfg.RAG.FTSTechWeight == 0    { cfg.RAG.FTSTechWeight = defaults().RAG.FTSTechWeight }
		if cfg.RAG.RerankerTimeoutMs == 0 { cfg.RAG.RerankerTimeoutMs = defaults().RAG.RerankerTimeoutMs }
		if cfg.RAG.HNSWThreshold == 0     { cfg.RAG.HNSWThreshold = defaults().RAG.HNSWThreshold }
		// VecCacheSize à 0 = illimité (valeur intentionnellement valide),
		// on applique le défaut seulement si négatif
		if cfg.RAG.VecCacheSize < 0      { cfg.RAG.VecCacheSize = defaults().RAG.VecCacheSize }
		break
	}

	// FIX : override par variable d'environnement
	// Permet de changer l'IP Ollama sans toucher au fichier config
	// Ex : OLLAMA_HOST=192.168.1.50 ./rag-web
	if h := os.Getenv("OLLAMA_HOST"); h != "" {
		h = strings.TrimPrefix(h, "https://")
		h = strings.TrimPrefix(h, "http://")
		if i := strings.LastIndex(h, ":"); i != -1 { h = h[:i] }
		cfg.Ollama.Host = h
	}
	if p := os.Getenv("OLLAMA_PORT"); p != "" {
		var port int
		if _, err := fmt.Sscanf(p, "%d", &port); err == nil && port > 0 {
			cfg.Ollama.Port = port
		}
	}

	return cfg
}

// OllamaURL retourne l'URL complète du serveur Ollama
func (c Config) OllamaURL() string {
	return fmt.Sprintf("http://%s:%d", c.Ollama.Host, c.Ollama.Port)
}

// RerankerURL retourne l'URL complète du reranker
func (c Config) RerankerURL() string {
	return fmt.Sprintf("http://%s:%d", c.Reranker.Host, c.Reranker.Port)
}
