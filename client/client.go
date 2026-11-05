package scalarflow

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const DefaultBaseURL = "http://localhost:8000"

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

type NetworkConfig struct {
	NumIn          int    `json:"num_in"`
	LayerSizes     []int  `json:"layer_sizes"`
	UnitActivation string `json:"unit_activation,omitempty"`
}

type Network struct {
	ID         string        `json:"id"`
	Config     NetworkConfig `json:"config"`
	Parameters []float64     `json:"parameters,omitempty"`
}

type FitResult struct {
	Epochs int       `json:"epochs"`
	Losses []float64 `json:"losses"`
}

type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("scalarflow: request failed with status %d", e.StatusCode)
	}
	return fmt.Sprintf("scalarflow: request failed with status %d: %s", e.StatusCode, e.Message)
}

type predictRequest struct {
	Inputs [][]float64 `json:"inputs"`
}

type predictResponse struct {
	Outputs [][]float64 `json:"outputs"`
}

type fitRequest struct {
	Inputs  [][]float64 `json:"inputs"`
	Outputs [][]float64 `json:"outputs"`
	Epochs  int         `json:"epochs"`
}

func New(baseURL string) *Client {
	return NewWithHTTPClient(baseURL, nil)
}

func NewWithHTTPClient(baseURL string, httpClient *http.Client) *Client {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = DefaultBaseURL
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 30 * time.Second}
	}
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		HTTPClient: httpClient,
	}
}

func (c *Client) CreateNetwork(ctx context.Context, cfg NetworkConfig) (*Network, error) {
	var network Network
	if err := c.doJSON(ctx, http.MethodPost, "/networks", cfg, &network); err != nil {
		return nil, err
	}
	return &network, nil
}

func (c *Client) GetNetwork(ctx context.Context, id string) (*Network, error) {
	var network Network
	if err := c.doJSON(ctx, http.MethodGet, "/networks/"+url.PathEscape(id), nil, &network); err != nil {
		return nil, err
	}
	return &network, nil
}

func (c *Client) DeleteNetwork(ctx context.Context, id string) error {
	return c.doJSON(ctx, http.MethodDelete, "/networks/"+url.PathEscape(id), nil, nil)
}

func (c *Client) Predict(ctx context.Context, id string, inputs [][]float64) ([][]float64, error) {
	var response predictResponse
	body := predictRequest{Inputs: inputs}
	if err := c.doJSON(ctx, http.MethodPost, "/networks/"+url.PathEscape(id)+"/predict", body, &response); err != nil {
		return nil, err
	}
	return response.Outputs, nil
}

func (c *Client) Fit(ctx context.Context, id string, inputs, outputs [][]float64, epochs int) (*FitResult, error) {
	var result FitResult
	body := fitRequest{Inputs: inputs, Outputs: outputs, Epochs: epochs}
	if err := c.doJSON(ctx, http.MethodPost, "/networks/"+url.PathEscape(id)+"/fit", body, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) doJSON(ctx context.Context, method, path string, body, out interface{}) error {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("scalarflow: encode request: %w", err)
		}
		payload = bytes.NewReader(encoded)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, payload)
	if err != nil {
		return fmt.Errorf("scalarflow: build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("scalarflow: send request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("scalarflow: read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &APIError{StatusCode: resp.StatusCode, Message: errorMessage(data)}
	}

	if out == nil || len(bytes.TrimSpace(data)) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("scalarflow: decode response: %w", err)
	}
	return nil
}

func errorMessage(data []byte) string {
	var payload struct {
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(data, &payload); err == nil {
		if payload.Error != "" {
			return payload.Error
		}
		if payload.Message != "" {
			return payload.Message
		}
	}
	return strings.TrimSpace(string(data))
}
