package scalarflow

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultPeerBaseURL points at the Namergence server, whose /ping endpoint is
// used to check that the two services can reach each other.
const DefaultPeerBaseURL = "http://127.0.0.1:8000"

// DefaultPongAddr is the address the ScalarFlow pong server listens on.
const DefaultPongAddr = "127.0.0.1:8001"

// PingResponse is the JSON body returned by the /ping and /pong endpoints.
type PingResponse struct {
	Message string `json:"message"`
	Service string `json:"service"`
}

// PingClient talks to a peer service's /ping endpoint.
type PingClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewPingClient creates a client for baseURL. An empty baseURL uses
// DefaultPeerBaseURL.
func NewPingClient(baseURL string) *PingClient {
	if baseURL == "" {
		baseURL = DefaultPeerBaseURL
	}
	return &PingClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Ping calls GET /ping on the peer and returns its response.
func (c *PingClient) Ping(ctx context.Context) (*PingResponse, error) {
	baseURL := c.BaseURL
	if baseURL == "" {
		baseURL = DefaultPeerBaseURL
	}
	url := strings.TrimRight(baseURL, "/") + "/ping"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("scalarflow: create ping request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("scalarflow: send ping request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("scalarflow: peer returned %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}

	var pong PingResponse
	if err := json.NewDecoder(resp.Body).Decode(&pong); err != nil {
		return nil, fmt.Errorf("scalarflow: decode ping response: %w", err)
	}
	return &pong, nil
}

// Pong answers a ping with the ScalarFlow pong payload.
func Pong(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(PingResponse{Message: "pong", Service: "scalarflow"})
}

// PongHandler returns an HTTP handler that serves the /pong endpoint.
func PongHandler() http.Handler {
	return http.HandlerFunc(Pong)
}

// NewPongServer returns an HTTP server that serves /pong on addr. An empty
// addr uses DefaultPongAddr.
func NewPongServer(addr string) *http.Server {
	if addr == "" {
		addr = DefaultPongAddr
	}

	mux := http.NewServeMux()
	mux.Handle("/pong", PongHandler())

	return &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
}
