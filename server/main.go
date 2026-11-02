package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type model struct {
	NumIn       int     `json:"num_in"`
	LayerSizes  []int   `json:"layer_sizes"`
	Activation  string  `json:"activation"`
	Layers      []layer `json:"layers"`
}

type layer struct {
	Neurons []neuron `json:"neurons"`
}

type neuron struct {
	Weights    []float64 `json:"weights"`
	Bias       float64   `json:"bias"`
	Activation string    `json:"activation"`
}

type stats struct {
	Layers     int `json:"layers"`
	Neurons    int `json:"neurons"`
	Weights    int `json:"weights"`
	Biases     int `json:"biases"`
	Parameters int `json:"parameters"`
}

type trainRequest struct {
	NumIn      int       `json:"num_in"`
	LayerSizes []int     `json:"layer_sizes"`
	Activation string    `json:"activation"`
	Inputs     [][]float64 `json:"inputs"`
	Outputs    targets   `json:"outputs"`
	Epochs     int       `json:"epochs"`
	Model      *model    `json:"model"`
}

type predictRequest struct {
	Model  *model      `json:"model"`
	Inputs [][]float64 `json:"inputs"`
}

type summaryRequest struct {
	Model *model `json:"model"`
}

type targets []float64

func (t *targets) UnmarshalJSON(data []byte) error {
	var flat []float64
	if err := json.Unmarshal(data, &flat); err == nil {
		*t = flat
		return nil
	}

	var rows [][]float64
	if err := json.Unmarshal(data, &rows); err != nil {
		return errors.New("outputs must be an array of numbers or single-value arrays")
	}
	flat = make([]float64, 0, len(rows))
	for _, row := range rows {
		if len(row) != 1 {
			return errors.New("outputs rows must each contain exactly one number")
		}
		flat = append(flat, row[0])
	}
	*t = flat
	return nil
}

type bridgeRequest struct {
	Action     string        `json:"action"`
	NumIn      int           `json:"num_in,omitempty"`
	LayerSizes []int         `json:"layer_sizes,omitempty"`
	Activation string        `json:"activation,omitempty"`
	Inputs     [][]float64   `json:"inputs,omitempty"`
	Outputs    []float64     `json:"outputs,omitempty"`
	Epochs     int           `json:"epochs,omitempty"`
	Model      *model        `json:"model,omitempty"`
}

type bridgeResponse struct {
	OK          bool     `json:"ok"`
	Error       string   `json:"error"`
	Kind        string   `json:"kind"`
	Model       *model   `json:"model"`
	Predictions []float64 `json:"predictions"`
	Loss        *float64 `json:"loss"`
	Stats       *stats   `json:"stats"`
}

type server struct {
	python     string
	bridgePath string
	timeout    time.Duration
}

func main() {
	addr := envOr("SCALARFLOW_ADDR", ":8080")
	python := envOr("SCALARFLOW_PYTHON", "python3")
	bridge, err := resolveBridge()
	if err != nil {
		log.Fatal(err)
	}

	srv := &server{
		python:     python,
		bridgePath: bridge,
		timeout:    2 * time.Minute,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/v1/train", srv.handleTrain)
	mux.HandleFunc("/v1/predict", srv.handlePredict)
	mux.HandleFunc("/v1/summary", srv.handleSummary)

	httpServer := &http.Server{
		Addr:              addr,
		Handler:           withJSON(mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("scalarflow HTTP server listening on %s (bridge: %s)", addr, bridge)
	log.Fatal(httpServer.ListenAndServe())
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func resolveBridge() (string, error) {
	if path := strings.TrimSpace(os.Getenv("SCALARFLOW_BRIDGE")); path != "" {
		return path, nil
	}
	candidates := []string{"bridge.py", filepath.Join("server", "bridge.py")}
	if executable, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(executable), "bridge.py"))
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", errors.New("could not find bridge.py; set SCALARFLOW_BRIDGE")
}

func withJSON(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		next.ServeHTTP(w, r)
	})
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *server) handleTrain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request trainRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if request.Model == nil {
		if err := validateShape(request.NumIn, request.LayerSizes); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}
	if err := validateInputs(request.Inputs); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(request.Outputs) != len(request.Inputs) {
		writeError(w, http.StatusBadRequest, "outputs must contain one target per input row")
		return
	}

	response, err := s.callBridge(bridgeRequest{
		Action:     "train",
		NumIn:      request.NumIn,
		LayerSizes: request.LayerSizes,
		Activation: request.Activation,
		Inputs:     request.Inputs,
		Outputs:    request.Outputs,
		Epochs:     request.Epochs,
		Model:      request.Model,
	})
	if err != nil {
		writeBridgeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request predictRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if request.Model == nil {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	if err := validateInputs(request.Inputs); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	response, err := s.callBridge(bridgeRequest{
		Action: "predict",
		Inputs: request.Inputs,
		Model:  request.Model,
	})
	if err != nil {
		writeBridgeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *server) handleSummary(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request summaryRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if request.Model == nil {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	response, err := s.callBridge(bridgeRequest{Action: "summary", Model: request.Model})
	if err != nil {
		writeBridgeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *server) callBridge(request bridgeRequest) (*bridgeResponse, error) {
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, s.python, s.bridgePath)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, errors.New("python bridge timed out")
		}
		message := strings.TrimSpace(stderr.String())
		if message == "" {
			message = err.Error()
		}
		return nil, fmt.Errorf("python bridge failed: %s", message)
	}

	var response bridgeResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		return nil, fmt.Errorf("invalid response from python bridge: %w", err)
	}
	if !response.OK {
		status := http.StatusInternalServerError
		if response.Kind == "validation" {
			status = http.StatusBadRequest
		}
		return nil, &bridgeError{status: status, message: response.Error}
	}
	return &response, nil
}

type bridgeError struct {
	status  int
	message string
}

func (e *bridgeError) Error() string { return e.message }

func decodeJSON(w http.ResponseWriter, r *http.Request, destination any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 8<<20)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("invalid JSON body: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("request body must contain a single JSON object")
	}
	return nil
}

func validateShape(numIn int, layerSizes []int) error {
	if numIn < 1 {
		return errors.New("num_in must be a positive integer")
	}
	if len(layerSizes) == 0 {
		return errors.New("layer_sizes must not be empty")
	}
	for _, size := range layerSizes {
		if size < 1 {
			return errors.New("every layer size must be a positive integer")
		}
	}
	return nil
}

func validateInputs(inputs [][]float64) error {
	if len(inputs) == 0 {
		return errors.New("inputs must be a non-empty array of rows")
	}
	width := len(inputs[0])
	if width == 0 {
		return errors.New("input rows must not be empty")
	}
	for _, row := range inputs {
		if len(row) != width {
			return errors.New("all input rows must have the same length")
		}
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("failed to write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func writeBridgeError(w http.ResponseWriter, err error) {
	var bridgeErr *bridgeError
	if errors.As(err, &bridgeErr) {
		writeError(w, bridgeErr.status, bridgeErr.message)
		return
	}
	writeError(w, http.StatusBadGateway, err.Error())
}
