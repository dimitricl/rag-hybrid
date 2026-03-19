package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	BaseURL string
	HTTP    *http.Client
}

func New(host string, port int) *Client {
	return &Client{
		BaseURL: fmt.Sprintf("http://%s:%d", host, port),
		HTTP:    &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *Client) Embed(texts []string, model string) ([][]float32, error) {
	req := map[string]interface{}{"model": model, "input": texts}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("embed marshal: %w", err)
	}
	resp, err := c.HTTP.Post(c.BaseURL+"/api/embed", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("embed decode: %w", err)
	}
	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("embed: réponse vide du modèle %s", model)
	}
	return result.Embeddings, nil
}

func (c *Client) Generate(prompt, model string) (string, error) {
	req := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"num_predict":    1024,
			"stop":           []string{"Utilisateur:", "User:", "QUESTION :"},
			"temperature":    0.1,
			"repeat_penalty": 1.5,
			"repeat_last_n":  256,
		},
	}
	data, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("generate marshal: %w", err)
	}
	resp, err := c.HTTP.Post(c.BaseURL+"/api/generate", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var result struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("generate decode: %w", err)
	}
	return result.Response, nil
}

// GenerateStream lance le streaming de tokens depuis Ollama.
//
// FIX goroutine leak : on accepte un context.Context.
// Si le client HTTP coupe la connexion (browser fermé, timeout), le ctx est annulé
// côté appelant (via r.Context() dans handleChatStream), ce qui provoque la sortie
// de la goroutine interne via le select sur <-ctx.Done().
//
// Sans ça : la goroutine continue de bloquer sur out <- token indéfiniment
// une fois le buffer de 100 saturé → leak mémoire progressif sous charge.
func (c *Client) GenerateStream(ctx context.Context, prompt, model string) (<-chan string, error) {
	req := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"num_predict": 1024,  // Limite max tokens — empêche les boucles infinies
			"stop":        []string{"Utilisateur:", "User:", "QUESTION :", "\n\nUtilisateur", "\n\nUser"},
			"temperature": 0.1,  // Faible température = moins d'hallucinations/répétitions
			"repeat_penalty": 1.5,
			"repeat_last_n":  256, // Pénalise les répétitions de tokens
		},
	}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("stream marshal: %w", err)
	}

	// On passe le context à la requête HTTP pour annuler la connexion Ollama
	// dès que le client web se déconnecte
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+"/api/generate", bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("stream new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return nil, err
	}

	out := make(chan string, 100)

	go func() {
		defer close(out)
		defer resp.Body.Close()
		defer func() {
			if r := recover(); r != nil {
				// Envoie l'erreur seulement si le context n'est pas annulé
				// (sinon le channel est peut-être déjà consommé/fermé)
				select {
				case out <- fmt.Sprintf("[ERREUR STREAMING: %v]", r):
				case <-ctx.Done():
				}
			}
		}()

		decoder := json.NewDecoder(resp.Body)
		for {
			// Vérifie l'annulation AVANT de bloquer sur Decode
			select {
			case <-ctx.Done():
				return
			default:
			}

			var streamResp struct {
				Response string `json:"response"`
				Done     bool   `json:"done"`
			}

			if err := decoder.Decode(&streamResp); err != nil {
				return
			}

			if streamResp.Response != "" {
				// Envoi avec fallback sur annulation — évite le blocage si le
				// canal est plein ET que le contexte est annulé simultanément
				select {
				case out <- streamResp.Response:
				case <-ctx.Done():
					return
				}
			}

			if streamResp.Done {
				return
			}
		}
	}()

	return out, nil
}

func (c *Client) Health() error {
	resp, err := c.HTTP.Get(c.BaseURL + "/api/tags")
	if err != nil {
		return fmt.Errorf("unreachable: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("unhealthy: HTTP %d", resp.StatusCode)
	}
	return nil
}
