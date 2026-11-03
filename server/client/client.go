// Package client is a Go client for the ScalarFlow HTTP API.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultTimeout matches the server's bridge timeout.
const DefaultTimeout = 2 * time.Minute

// Model is the serialized ScalarFlow network returned by the server.
type Model struct {
	NumIn      int     `json:"num_in"`
	LayerSizes []int   `json:"layer_sizes"`
	Activation string  `json:"activation"`
	Layers     []Layer `json:"layers"`
}

// Layer is a single dense layer in a Model.
type Layer struct {
	Neurons []Neuron `json:"neurons"`
}

// Neuron holds one unit's weights, bias and activation.
type Neuron struct {
	Weights    []float64 `json:"weights"`
	Bias       float64   `json:"bias"`
	Activation string    `json:"activation"`
}

// Stats describes the parameter counts of a Model.
type Stats struct {
	Layers     int `json:"layers"`
	Neurons    int `json:"neurons"`
	Weights    int `json:"weights"`
	Biases     int `json:"biases"`
	Parameters int `json:"parameters"`
}

// TrainRequest is the body sent to POST /v1/train.
//
// Either Model is set to continue training an existing network, or NumIn and
// LayerSizes are set to create a new one.
type TrainRequest struct {
	NumIn      int         `json:"num_in,omitempty"`
	LayerSizes []int       `json:"layer_sizes,omitempty"`
	Activation string      `json:"activation,omitempty"`
	Inputs     [][]float64 `json:"inputs"`
	Outputs    []float64   `json:"outputs"`
	Epochs     int         `json:"epochs,omitempty"`
	Model      *Model      `json:"model,omitempty"`
}

// TrainResponse is the result of a successful train call.
type TrainResponse struct {
	Model       *Model    `json:"model"`
	Predictions []float64 `json:"predictions"`
	Loss        *float64  `json:"loss"`
}

// PredictRequest is the body sent to POST /v1/predict.
type PredictRequest struct {
	Model  *Model      `json:"model"`
	Inputs [][]float64 `json:"inputs"`
}

// PredictResponse is the result of a successful predict call.
type PredictResponse struct {
	Predictions []float64 `json:"predictions"`
}

// SummaryRequest is the body sent to POST /v1/summary.
type SummaryRequest struct {
	Model *Model `json:"model"`
}

// SummaryResponse is the result of a successful summary call.
type SummaryResponse struct {
	Stats *Stats `json:"stats"`
	Model *Model `json:"model"`
}

// HealthResponse is the result of a successful health check.
type HealthResponse struct {
	Status string `json:"status"`
}

// APIError is returned when the server responds with a non-2xx status.
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("scalarflow: %s (status %d)", e.Message, e.StatusCode)
}

// Client talks to a ScalarFlow HTTP server.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// New returns a Client using the default HTTP client.
func New(baseURL string) *Client {
	return NewWithHTTPClient(baseURL, nil)
}

// NewWithHTTPClient returns a Client using the supplied HTTP client. A nil
// client falls back to http.DefaultClient, and a client without a timeout gets
// DefaultTimeout.
func NewWithHTTPClient(baseURL string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: DefaultTimeout}
	} else if httpClient.Timeout == 0 {
		clone := *httpClient
		clone.Timeout = DefaultTimeout
		httpClient = &clone
	}
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: httpClient,
	}
}

// Health checks that the server is reachable.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var response HealthResponse
	if err := c.do(ctx, http.MethodGet, "/health", nil, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

// Train creates or continues training a network.
func (c *Client) Train(ctx context.Context, request TrainRequest) (*TrainResponse, error) {
	var response TrainResponse
	if err := c.do(ctx, http.MethodPost, "/v1/train", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

// Predict runs inference for a supplied model.
func (c *Client) Predict(ctx context.Context, request PredictRequest) (*PredictResponse, error) {
	var response PredictResponse
	if err := c.do(ctx, http.MethodPost, "/v1/predict", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

// Summary returns parameter counts for a supplied model.
func (c *Client) Summary(ctx context.Context, request SummaryRequest) (*SummaryResponse, error) {
	var response SummaryResponse
	if err := c.do(ctx, http.MethodPost, "/v1/summary", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

func (c *Client) do(ctx context.Context, method, path string, payload, destination any) error {
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		body = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	request.Header.Set("Accept", "application/json")

	response, err := c.httpClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	data, err := io.ReadAll(response.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return newAPIError(response.StatusCode, data)
	}

	if destination == nil {
		return nil
	}
	if err := json.Unmarshal(data, destination); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func newAPIError(status int, data []byte) *APIError {
	var payload struct {
		Error string `json:"error"`
	}
	message := strings.TrimSpace(string(data))
	if err := json.Unmarshal(data, &payload); err == nil && payload.Error != "" {
		message = payload.Error
	}
	if message == "" {
		message = http.StatusText(status)
	}
	return &APIError{StatusCode: status, Message: message}
}
