package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

const (
	listenAddr  = "0.0.0.0:8080"
	mimoBinary  = "mimo"
	modelName   = "mimo-code"
)

// OpenAI-compatible request/response types

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
}

type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

func (m *Message) ContentString() string {
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				sb.WriteString(p.Text)
			}
		}
		return sb.String()
	}
	return string(m.Content)
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int      `json:"index"`
	Message      Message  `json:"message"`
	FinishReason string   `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type StreamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        StreamDelta  `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

type Model struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	OwnedBy  string `json:"owned_by"`
}

type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/health", handleHealth)

	log.Printf("MiMo proxy listening on %s", listenAddr)
	log.Fatal(http.ListenAndServe(listenAddr, mux))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ModelList{
		Object: "list",
		Data: []Model{
			{ID: modelName, Object: "model", OwnedBy: "mimo"},
		},
	})
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, http.StatusMethodNotAllowed, "only POST allowed", "invalid_request_error")
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if len(req.Messages) == 0 {
		sendError(w, http.StatusBadRequest, "messages required", "invalid_request_error")
		return
	}

	// Build prompt from messages
	prompt := buildPrompt(req.Messages)

	if req.Stream {
		handleStreamResponse(w, prompt)
	} else {
		handleSyncResponse(w, prompt)
	}
}

func buildPrompt(messages []Message) string {
	var sb strings.Builder
	for _, m := range messages {
		content := m.ContentString()
		switch m.Role {
		case "system":
			sb.WriteString("[System]: ")
			sb.WriteString(content)
			sb.WriteString("\n\n")
		case "user":
			sb.WriteString("[User]: ")
			sb.WriteString(content)
			sb.WriteString("\n\n")
		case "assistant":
			sb.WriteString("[Assistant]: ")
			sb.WriteString(content)
			sb.WriteString("\n\n")
		}
	}
	return sb.String()
}

func handleSyncResponse(w http.ResponseWriter, prompt string) {
	output, err := runMimo(prompt)
	if err != nil {
		sendError(w, http.StatusInternalServerError, "mimo error: "+err.Error(), "server_error")
		return
	}

	contentJSON, _ := json.Marshal(output)
	resp := ChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []Choice{
			{
				Index:        0,
				Message:      Message{Role: "assistant", Content: contentJSON},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     estimateTokens(prompt),
			CompletionTokens: estimateTokens(output),
			TotalTokens:      estimateTokens(prompt) + estimateTokens(output),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleStreamResponse(w http.ResponseWriter, prompt string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		sendError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	// Send role chunk first
	sendStreamChunk(w, flusher, StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   modelName,
		Choices: []StreamChoice{
			{Index: 0, Delta: StreamDelta{Role: "assistant"}, FinishReason: nil},
		},
	})

	// Run mimo and stream output
	output, err := runMimo(prompt)
	if err != nil {
		sendError(w, http.StatusInternalServerError, "mimo error: "+err.Error(), "server_error")
		return
	}

	// Stream the full response as a single chunk
	sendStreamChunk(w, flusher, StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   modelName,
		Choices: []StreamChoice{
			{Index: 0, Delta: StreamDelta{Content: output}, FinishReason: nil},
		},
	})

	// Send finish
	finishReason := "stop"
	sendStreamChunk(w, flusher, StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   modelName,
		Choices: []StreamChoice{
			{Index: 0, Delta: StreamDelta{}, FinishReason: &finishReason},
		},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func sendStreamChunk(w http.ResponseWriter, flusher http.Flusher, chunk StreamChunk) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

type MimoEvent struct {
	Type string          `json:"type"`
	Part json.RawMessage `json:"part,omitempty"`
}

type MimoTextPart struct {
	Text string `json:"text"`
}

type MimoStepFinish struct {
	Reason string `json:"reason"`
	Tokens struct {
		Total  int `json:"total"`
		Input  int `json:"input"`
		Output int `json:"output"`
	} `json:"tokens"`
}

func runMimo(prompt string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, mimoBinary, "run", "--format", "json", "--dangerously-skip-permissions", prompt)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return "", fmt.Errorf("mimo timeout after 300s")
		}
		if stderr.Len() > 0 {
			return "", fmt.Errorf("%s: %s", err.Error(), strings.TrimSpace(stderr.String()))
		}
		return "", err
	}

	var textParts []string
	var tokens MimoStepFinish

	scanner := bufio.NewScanner(bytes.NewReader(stdout.Bytes()))
	scanner.Buffer(make([]byte, 0, 64*1024), 64*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var event MimoEvent
		if err := json.Unmarshal(line, &event); err != nil {
			continue
		}

		switch event.Type {
		case "text":
			var part MimoTextPart
			if err := json.Unmarshal(event.Part, &part); err == nil {
				textParts = append(textParts, part.Text)
			}
		case "step_finish":
			json.Unmarshal(event.Part, &tokens)
		}
	}

	output := strings.Join(textParts, "")
	if output == "" {
		return "", fmt.Errorf("empty response from mimo")
	}

	return output, nil
}

func runMimoStream(prompt string, onChunk func(string)) error {
	cmd := exec.Command(mimoBinary, "run", prompt)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		return err
	}

	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		onChunk(scanner.Text() + "\n")
	}

	if err := cmd.Wait(); err != nil {
		if stderr.Len() > 0 {
			return fmt.Errorf("%s: %s", err.Error(), strings.TrimSpace(stderr.String()))
		}
		return err
	}

	return nil
}

func estimateTokens(text string) int {
	return len(text) / 4
}

func sendError(w http.ResponseWriter, code int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	resp := ErrorResponse{}
	resp.Error.Message = message
	resp.Error.Type = errType
	resp.Error.Code = errType
	json.NewEncoder(w).Encode(resp)
}
