package scalarflow

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewDefaultsBaseURL(t *testing.T) {
	client := New("")
	if client.BaseURL != DefaultBaseURL {
		t.Fatalf("BaseURL = %q, want %q", client.BaseURL, DefaultBaseURL)
	}
}

func TestNewTrimsTrailingSlash(t *testing.T) {
	client := New("http://example.com/")
	if client.BaseURL != "http://example.com" {
		t.Fatalf("BaseURL = %q, want %q", client.BaseURL, "http://example.com")
	}
}

func TestCreateNetwork(t *testing.T) {
	var gotConfig NetworkConfig
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/networks" {
			t.Errorf("path = %s, want /networks", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotConfig); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"net-1","config":{"num_in":2,"layer_sizes":[2,1],"unit_activation":"relu"}}`)
	}))
	defer server.Close()

	client := New(server.URL)
	network, err := client.CreateNetwork(context.Background(), NetworkConfig{
		NumIn:          2,
		LayerSizes:     []int{2, 1},
		UnitActivation: "relu",
	})
	if err != nil {
		t.Fatalf("CreateNetwork: %v", err)
	}
	if network.ID != "net-1" {
		t.Errorf("ID = %q, want net-1", network.ID)
	}
	if gotConfig.NumIn != 2 || len(gotConfig.LayerSizes) != 2 || gotConfig.UnitActivation != "relu" {
		t.Errorf("unexpected request config: %+v", gotConfig)
	}
}

func TestGetNetworkEscapesID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("method = %s, want GET", r.Method)
		}
		if got, want := r.URL.EscapedPath(), "/networks/a%2Fb%20c"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"a/b c","config":{"num_in":1,"layer_sizes":[1]}}`)
	}))
	defer server.Close()

	client := New(server.URL)
	network, err := client.GetNetwork(context.Background(), "a/b c")
	if err != nil {
		t.Fatalf("GetNetwork: %v", err)
	}
	if network.ID != "a/b c" {
		t.Errorf("ID = %q, want %q", network.ID, "a/b c")
	}
}

func TestDeleteNetwork(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			t.Errorf("method = %s, want DELETE", r.Method)
		}
		if r.URL.Path != "/networks/net-1" {
			t.Errorf("path = %s, want /networks/net-1", r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	if err := New(server.URL).DeleteNetwork(context.Background(), "net-1"); err != nil {
		t.Fatalf("DeleteNetwork: %v", err)
	}
}

func TestPredict(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/networks/net-1/predict" {
			t.Errorf("path = %s, want /networks/net-1/predict", r.URL.Path)
		}
		var body predictRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if len(body.Inputs) != 2 || len(body.Inputs[0]) != 2 {
			t.Errorf("unexpected inputs: %+v", body.Inputs)
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"outputs":[[0.25],[0.75]]}`)
	}))
	defer server.Close()

	outputs, err := New(server.URL).Predict(context.Background(), "net-1", [][]float64{{0, 1}, {1, 0}})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if len(outputs) != 2 || outputs[0][0] != 0.25 || outputs[1][0] != 0.75 {
		t.Errorf("unexpected outputs: %+v", outputs)
	}
}

func TestFit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/networks/net-1/fit" {
			t.Errorf("path = %s, want /networks/net-1/fit", r.URL.Path)
		}
		var body fitRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if body.Epochs != 5 || len(body.Inputs) != 1 || len(body.Outputs) != 1 {
			t.Errorf("unexpected fit request: %+v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"epochs":5,"losses":[1.0,0.5,0.25,0.1,0.05]}`)
	}))
	defer server.Close()

	result, err := New(server.URL).Fit(context.Background(), "net-1", [][]float64{{1}}, [][]float64{{0}}, 5)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if result.Epochs != 5 || len(result.Losses) != 5 {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestAPIErrorFromJSONBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":"bad request"}`)
	}))
	defer server.Close()

	_, err := New(server.URL).GetNetwork(context.Background(), "missing")
	if err == nil {
		t.Fatal("expected an error")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *APIError", err)
	}
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, http.StatusBadRequest)
	}
	if apiErr.Message != "bad request" {
		t.Errorf("Message = %q, want %q", apiErr.Message, "bad request")
	}
	if got, want := apiErr.Error(), "scalarflow: request failed with status 400: bad request"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestAPIErrorWithoutMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	_, err := New(server.URL).GetNetwork(context.Background(), "missing")

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type = %T, want *APIError", err)
	}
	if got, want := apiErr.Error(), "scalarflow: request failed with status 404"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestDoJSONEncodeError(t *testing.T) {
	client := New("http://example.com")
	err := client.doJSON(context.Background(), http.MethodPost, "/networks", make(chan int), nil)
	if err == nil {
		t.Fatal("expected an encode error")
	}
}
