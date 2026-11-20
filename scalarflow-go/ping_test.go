package scalarflow

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("method = %s, want GET", r.Method)
		}
		if r.URL.Path != "/ping" {
			t.Errorf("path = %s, want /ping", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(PingResponse{Message: "pong", Service: "namergence"})
	}))
	defer server.Close()

	client := NewPingClient(server.URL + "/")
	pong, err := client.Ping(context.Background())
	if err != nil {
		t.Fatalf("Ping() error = %v", err)
	}
	if pong.Message != "pong" {
		t.Errorf("message = %q, want pong", pong.Message)
	}
	if pong.Service != "namergence" {
		t.Errorf("service = %q, want namergence", pong.Service)
	}
}

func TestPingServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "offline", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	client := NewPingClient(server.URL)
	if _, err := client.Ping(context.Background()); err == nil {
		t.Fatal("Ping() error = nil, want server error")
	}
}

func TestPongHandler(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/pong", nil)

	PongHandler().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}

	var pong PingResponse
	if err := json.NewDecoder(recorder.Body).Decode(&pong); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if pong.Message != "pong" {
		t.Errorf("message = %q, want pong", pong.Message)
	}
	if pong.Service != "scalarflow" {
		t.Errorf("service = %q, want scalarflow", pong.Service)
	}
}

func TestPongHandlerRejectsPost(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/pong", nil)

	PongHandler().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusMethodNotAllowed)
	}
}
