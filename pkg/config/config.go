package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
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
	DBPath     string `yaml:"db_path"`
	EmbedModel string `yaml:"embed_model"`
	DefaultModel string `yaml:"default_model"`
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
			DBPath:       "~/.rag-hybrid",
			EmbedModel:   "nomic-embed-text:latest",
			DefaultModel: "mistral:7b-instruct",
		},
	}
}

// Load charge la config depuis configs/config.yaml relatif au binaire,
// ou depuis ~/.rag-hybrid/config.yaml en fallback.
// Si aucun fichier trouvé, retourne les valeurs par défaut sans erreur.
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
		if cfg.Ollama.Host == "" { cfg.Ollama.Host = defaults().Ollama.Host }
		if cfg.Ollama.Port == 0  { cfg.Ollama.Port = defaults().Ollama.Port }
		if cfg.Reranker.Host == "" { cfg.Reranker.Host = defaults().Reranker.Host }
		if cfg.Reranker.Port == 0  { cfg.Reranker.Port = defaults().Reranker.Port }
		if cfg.Web.Port == 0 { cfg.Web.Port = defaults().Web.Port }
		if cfg.RAG.DBPath == "" { cfg.RAG.DBPath = defaults().RAG.DBPath }
		if cfg.RAG.EmbedModel == "" { cfg.RAG.EmbedModel = defaults().RAG.EmbedModel }
		if cfg.RAG.DefaultModel == "" { cfg.RAG.DefaultModel = defaults().RAG.DefaultModel }
		return cfg
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
