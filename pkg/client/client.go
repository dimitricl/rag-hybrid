package client

import (
	"bytes"
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
	req := map[string]interface{}{"model": model, "prompt": prompt, "stream": false}
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

func (c *Client) GenerateStream(prompt, model string) (<-chan string, error) {
	req := map[string]interface{}{"model": model, "prompt": prompt, "stream": true}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("stream marshal: %w", err)
	}

	resp, err := c.HTTP.Post(c.BaseURL+"/api/generate", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}

	out := make(chan string, 100)

	go func() {
		defer close(out)
		defer resp.Body.Close()
		// Récupère les panics dans la goroutine de streaming
		defer func() {
			if r := recover(); r != nil {
				out <- fmt.Sprintf("[ERREUR STREAMING: %v]", r)
			}
		}()

		decoder := json.NewDecoder(resp.Body)
		for {
			var streamResp struct {
				Response string `json:"response"`
				Done     bool   `json:"done"`
			}

			if err := decoder.Decode(&streamResp); err != nil {
				return
			}

			if streamResp.Response != "" {
				out <- streamResp.Response
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
