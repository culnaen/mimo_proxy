package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const (
	defaultListenAddr = "0.0.0.0:8080"
	defaultMimoBinary = "mimo"
	defaultModelName  = "mimo-code"
)

var (
	listenAddr  = envOrDefault("MIMO_LISTEN_ADDR", defaultListenAddr)
	mimoBinary  = envOrDefault("MIMO_BINARY", defaultMimoBinary)
	modelName   = envOrDefault("MIMO_MODEL", defaultModelName)
	modelsConfig *ModelsConfig
)

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func genID(prefix string) string {
	b := make([]byte, 8)
	rand.Read(b)
	return fmt.Sprintf("%s_%s%d", prefix, hex.EncodeToString(b), time.Now().UnixNano()%1000000)
}

// OpenAI types

type ChatRequest struct {
	Model           string          `json:"model"`
	Messages        []Message       `json:"messages"`
	Stream          bool            `json:"stream,omitempty"`
	StreamOptions   *StreamOptions  `json:"stream_options,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	SessionID       string          `json:"session_id,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type ChatResponse struct {
	ID        string   `json:"id"`
	Object    string   `json:"object"`
	Created   int64    `json:"created"`
	Model     string   `json:"model"`
	Choices   []Choice `json:"choices"`
	Usage     *Usage   `json:"usage,omitempty"`
	SessionID string   `json:"session_id,omitempty"`
}

type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type ResponseMessage struct {
	Role             string     `json:"role"`
	Content          *string    `json:"content"`
	ReasoningContent *string    `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type StreamChunk struct {
	ID        string         `json:"id"`
	Object    string         `json:"object"`
	Created   int64          `json:"created"`
	Model     string         `json:"model"`
	Choices   []StreamChoice `json:"choices"`
	Usage     *Usage         `json:"usage,omitempty"`
	SessionID string         `json:"session_id,omitempty"`
}

type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type StreamDelta struct {
	Role             string     `json:"role,omitempty"`
	Content          *string    `json:"content,omitempty"`
	ReasoningContent *string    `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type ModelList struct {
	Object string       `json:"object"`
	Data   []ModelEntry `json:"data"`
}

type ModelEntry struct {
	ID            string   `json:"id"`
	Object        string   `json:"object"`
	OwnedBy       string   `json:"owned_by"`
	Capabilities  []string `json:"capabilities,omitempty"`
	ContextWindow int      `json:"context_window,omitempty"`
	MaxTokens     int      `json:"max_tokens,omitempty"`
}

// MiMo CLI event types

type MimoEvent struct {
	Type      string          `json:"type"`
	Timestamp int64           `json:"timestamp"`
	SessionID string          `json:"sessionID"`
	Part      json.RawMessage `json:"part,omitempty"`
}

type MimoTextPart struct {
	Text string `json:"text"`
}

type MimoReasoningPart struct {
	Text string `json:"text"`
}

type MimoToolUsePart struct {
	Tool   string `json:"tool"`
	CallID string `json:"callID"`
	State  struct {
		Input json.RawMessage `json:"input"`
	} `json:"state"`
}

type MimoStepFinishPart struct {
	Reason string `json:"reason"`
	Tokens struct {
		Total    int `json:"total"`
		Input    int `json:"input"`
		Output   int `json:"output"`
		Reasoning int `json:"reasoning"`
	} `json:"tokens"`
}

type MimoResult struct {
	Text             string
	Reasoning        string
	ToolCalls        []MimoToolCall
	PromptTokens     int
	CompletionTokens int
	ReasoningTokens  int
	TotalTokens      int
	FinishReason     string
	SessionID        string
}

type MimoToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// models.json config

type ModelsConfig struct {
	Providers map[string]ProviderConfig `json:"providers"`
}

type ProviderConfig struct {
	Models []ModelDef `json:"models,omitempty"`
}

type ModelDef struct {
	ID            string `json:"id"`
	Name          string `json:"name,omitempty"`
	Reasoning     bool   `json:"reasoning,omitempty"`
	ContextWindow int    `json:"contextWindow,omitempty"`
	MaxTokens     int    `json:"maxTokens,omitempty"`
}

// Anthropic types

type AnthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []AnthropicMessage `json:"messages"`
	System      interface{}        `json:"system,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Stream      bool               `json:"stream,omitempty"`
	Thinking    *AnthropicThinking `json:"thinking,omitempty"`
	Temperature *float64           `json:"temperature,omitempty"`
}

type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type AnthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type AnthropicResponse struct {
	ID         string             `json:"id"`
	Type       string             `json:"type"`
	Role       string             `json:"role"`
	Content    []AnthropicContent `json:"content"`
	Model      string             `json:"model"`
	StopReason string             `json:"stop_reason"`
	Usage      *AnthropicUsage    `json:"usage"`
}

type AnthropicContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type AnthropicStreamEvent struct {
	Type  string          `json:"type"`
	Delta interface{}     `json:"delta,omitempty"`
	Usage *AnthropicUsage `json:"usage,omitempty"`
}

func main() {
	loadModelsConfig()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/v1/messages", handleAnthropicMessages)
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/health", handleHealth)

	srv := &http.Server{Addr: listenAddr, Handler: mux}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		<-ctx.Done()
		log.Println("Shutting down gracefully...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Printf("shutdown: %v", err)
		}
	}()

	log.Printf("MiMo proxy listening on %s", listenAddr)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func loadModelsConfig() {
	configPath := envOrDefault("MIMO_MODELS_CONFIG", "")
	if configPath == "" {
		home, _ := os.UserHomeDir()
		configPath = filepath.Join(home, ".mimocode", "models.json")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("No models.json at %s, using defaults", configPath)
			return
		}
		log.Printf("Warning: failed to read models.json: %v", err)
		return
	}

	var cfg ModelsConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		log.Printf("Warning: failed to parse models.json: %v", err)
		return
	}

	modelsConfig = &cfg
	log.Printf("Loaded models.json with %d providers", len(cfg.Providers))
}

// Handlers

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	var models []ModelEntry
	if modelsConfig != nil {
		for _, prov := range modelsConfig.Providers {
			for _, m := range prov.Models {
				entry := ModelEntry{
					ID:            m.ID,
					Object:        "model",
					OwnedBy:       "mimo",
					Capabilities:  []string{"text", "tool_use"},
					ContextWindow: m.ContextWindow,
					MaxTokens:     m.MaxTokens,
				}
				if m.Reasoning {
					entry.Capabilities = append(entry.Capabilities, "reasoning")
				}
				if entry.ContextWindow == 0 {
					entry.ContextWindow = 128000
				}
				if entry.MaxTokens == 0 {
					entry.MaxTokens = 32000
				}
				models = append(models, entry)
			}
		}
	}

	if len(models) == 0 {
		models = []ModelEntry{{
			ID:            modelName,
			Object:        "model",
			OwnedBy:       "mimo",
			Capabilities:  []string{"text", "reasoning", "tool_use"},
			ContextWindow: 128000,
			MaxTokens:     32000,
		}}
	}

	json.NewEncoder(w).Encode(ModelList{Object: "list", Data: models})
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

	prompt := buildPrompt(req.Messages)
	variant := mapReasoningEffort(req.ReasoningEffort)

	if req.Stream {
		streamOpenAI(w, prompt, variant, req.StreamOptions, req.SessionID)
	} else {
		syncOpenAI(w, prompt, variant, req.SessionID)
	}
}

func handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, http.StatusMethodNotAllowed, "only POST allowed", "invalid_request")
		return
	}

	var req AnthropicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request")
		return
	}

	if len(req.Messages) == 0 {
		sendError(w, http.StatusBadRequest, "messages required", "invalid_request")
		return
	}

	prompt := buildAnthropicPrompt(req)
	variant := ""
	if req.Thinking != nil && req.Thinking.Type == "enabled" {
		variant = "high"
	}

	if req.Stream {
		streamAnthropic(w, prompt, variant, req)
	} else {
		syncAnthropic(w, prompt, variant, req)
	}
}

// Prompt builders

func extractContent(content json.RawMessage) string {
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				sb.WriteString(p.Text)
			}
		}
		return sb.String()
	}
	return string(content)
}

func buildPrompt(messages []Message) string {
	var sb strings.Builder
	for _, m := range messages {
		content := extractContent(m.Content)
		switch m.Role {
		case "system", "developer":
			sb.WriteString("[System]: ")
			sb.WriteString(content)
			sb.WriteString("\n\n")
		case "user":
			sb.WriteString("[User]: ")
			sb.WriteString(content)
			sb.WriteString("\n\n")
		case "assistant":
			for _, tc := range m.ToolCalls {
				sb.WriteString(fmt.Sprintf("[ToolCall: %s(%s)]: \n", tc.Function.Name, tc.ID))
			}
			if content != "" {
				sb.WriteString("[Assistant]: ")
				sb.WriteString(content)
				sb.WriteString("\n\n")
			}
		case "tool":
			sb.WriteString(fmt.Sprintf("[ToolResult: %s]: ", m.ToolCallID))
			sb.WriteString(content)
			sb.WriteString("\n\n")
		}
	}
	return sb.String()
}

func buildAnthropicPrompt(req AnthropicRequest) string {
	var sb strings.Builder

	if req.System != nil {
		switch s := req.System.(type) {
		case string:
			sb.WriteString("[System]: ")
			sb.WriteString(s)
			sb.WriteString("\n\n")
		case []interface{}:
			for _, block := range s {
				if m, ok := block.(map[string]interface{}); ok {
					if text, ok := m["text"].(string); ok {
						sb.WriteString("[System]: ")
						sb.WriteString(text)
						sb.WriteString("\n\n")
					}
				}
			}
		}
	}

	for _, m := range req.Messages {
		content := extractContent(m.Content)
		switch m.Role {
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

func mapReasoningEffort(effort string) string {
	switch effort {
	case "low":
		return "minimal"
	case "medium", "high":
		return "high"
	default:
		return ""
	}
}

// Shared mimo runner

func buildMimoArgs(prompt, variant, sessionID string) []string {
	args := []string{"run", "--format", "json", "--thinking", "--dangerously-skip-permissions"}
	if variant != "" {
		args = append(args, "--variant", variant)
	}
	if sessionID != "" {
		args = append(args, "--session", sessionID)
	}
	args = append(args, prompt)
	return args
}

func runMimoCollect(prompt, variant, sessionID string) (*MimoResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, mimoBinary, buildMimoArgs(prompt, variant, sessionID)...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("mimo timeout after 300s")
		}
		if stderr.Len() > 0 {
			return nil, fmt.Errorf("%s: %s", err.Error(), strings.TrimSpace(stderr.String()))
		}
		return nil, err
	}

	result := &MimoResult{}
	var textParts, reasoningParts []string

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
		case "reasoning":
			var part MimoReasoningPart
			if err := json.Unmarshal(event.Part, &part); err == nil {
				reasoningParts = append(reasoningParts, part.Text)
			}
		case "tool_use":
			var part MimoToolUsePart
			if err := json.Unmarshal(event.Part, &part); err == nil {
				argsBytes, _ := json.Marshal(part.State.Input)
				result.ToolCalls = append(result.ToolCalls, MimoToolCall{
					ID:        part.CallID,
					Name:      part.Tool,
					Arguments: string(argsBytes),
				})
			}
		case "step_finish":
			var part MimoStepFinishPart
			if err := json.Unmarshal(event.Part, &part); err == nil {
				result.PromptTokens = part.Tokens.Input
				result.CompletionTokens = part.Tokens.Output
				result.ReasoningTokens = part.Tokens.Reasoning
				result.TotalTokens = part.Tokens.Total
				result.FinishReason = mapFinishReason(part.Reason)
			}
		}

		if event.SessionID != "" && result.SessionID == "" {
			result.SessionID = event.SessionID
		}
	}

	result.Text = strings.Join(textParts, "")
	result.Reasoning = strings.Join(reasoningParts, "")

	if result.Text == "" && len(result.ToolCalls) == 0 {
		return nil, fmt.Errorf("empty response from mimo")
	}

	return result, nil
}

func mapFinishReason(reason string) string {
	switch reason {
	case "tool-calls":
		return "tool_calls"
	case "length":
		return "length"
	default:
		return "stop"
	}
}

// OpenAI sync

func syncOpenAI(w http.ResponseWriter, prompt, variant, sessionID string) {
	result, err := runMimoCollect(prompt, variant, sessionID)
	if err != nil {
		sendError(w, http.StatusInternalServerError, "mimo error: "+err.Error(), "server_error")
		return
	}

	var content *string
	if result.Text != "" {
		content = &result.Text
	}

	var reasoning *string
	if result.Reasoning != "" {
		reasoning = &result.Reasoning
	}

	toolCalls := make([]ToolCall, len(result.ToolCalls))
	for i, tc := range result.ToolCalls {
		toolCalls[i] = ToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{Name: tc.Name, Arguments: tc.Arguments},
		}
	}

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ChatResponse{
		ID:        genID("chatcmpl"),
		Object:    "chat.completion",
		Created:   time.Now().Unix(),
		Model:     modelName,
		SessionID: result.SessionID,
		Choices: []Choice{{
			Index: 0,
			Message: ResponseMessage{
				Role:             "assistant",
				Content:          content,
				ReasoningContent: reasoning,
				ToolCalls:        toolCalls,
			},
			FinishReason: finishReason,
		}},
		Usage: &Usage{
			PromptTokens:     result.PromptTokens,
			CompletionTokens: result.CompletionTokens + result.ReasoningTokens,
			TotalTokens:      result.TotalTokens,
		},
	})
}

// OpenAI streaming

func streamOpenAI(w http.ResponseWriter, prompt, variant string, opts *StreamOptions, sessionID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		sendError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	id := genID("chatcmpl")
	created := time.Now().Unix()

	sendSSE(w, flusher, StreamChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: modelName,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Role: "assistant"}}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, mimoBinary, buildMimoArgs(prompt, variant, sessionID)...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		sendError(w, http.StatusInternalServerError, "failed to start mimo: "+err.Error(), "server_error")
		return
	}
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		sendError(w, http.StatusInternalServerError, "failed to start mimo: "+err.Error(), "server_error")
		return
	}

	var (
		promptTokens  int
		compTokens    int
		reasonTokens  int
		totalTokens   int
		finishReason  string
		respSessionID string
	)

	scanner := bufio.NewScanner(stdout)
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

		if event.SessionID != "" {
			if respSessionID == "" {
				respSessionID = event.SessionID
			}
		}

		switch event.Type {
		case "reasoning":
			var part MimoReasoningPart
			if err := json.Unmarshal(event.Part, &part); err == nil && part.Text != "" {
				delta := part.Text
				sendSSE(w, flusher, StreamChunk{
					ID: id, Object: "chat.completion.chunk", Created: created, Model: modelName,
					Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{ReasoningContent: &delta}}},
				})
			}
		case "text":
			var part MimoTextPart
			if err := json.Unmarshal(event.Part, &part); err == nil && part.Text != "" {
				delta := part.Text
				sendSSE(w, flusher, StreamChunk{
					ID: id, Object: "chat.completion.chunk", Created: created, Model: modelName,
					Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Content: &delta}}},
				})
			}
		case "step_finish":
			var part MimoStepFinishPart
			if err := json.Unmarshal(event.Part, &part); err == nil {
				totalTokens = part.Tokens.Total
				promptTokens = part.Tokens.Input
				compTokens = part.Tokens.Output
				reasonTokens = part.Tokens.Reasoning
				finishReason = mapFinishReason(part.Reason)
			}
		}
	}

	if err := cmd.Wait(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			log.Printf("mimo timeout after 300s")
		} else if stderr.Len() > 0 {
			log.Printf("mimo error: %s", strings.TrimSpace(stderr.String()))
		}
		if finishReason == "" {
			finishReason = "stop"
		}
	}

	sendSSE(w, flusher, StreamChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: modelName, SessionID: respSessionID,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{}, FinishReason: &finishReason}},
	})

	if opts != nil && opts.IncludeUsage {
		sendSSE(w, flusher, StreamChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: modelName, SessionID: respSessionID,
			Choices: []StreamChoice{},
			Usage: &Usage{PromptTokens: promptTokens, CompletionTokens: compTokens + reasonTokens, TotalTokens: totalTokens},
		})
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// Anthropic sync

func syncAnthropic(w http.ResponseWriter, prompt, variant string, req AnthropicRequest) {
	result, err := runMimoCollect(prompt, variant, "")
	if err != nil {
		anthropicError(w, http.StatusInternalServerError, err.Error(), "api_error")
		return
	}

	var content []AnthropicContent
	if result.Reasoning != "" {
		content = append(content, AnthropicContent{Type: "thinking", Thinking: result.Reasoning})
	}
	if result.Text != "" {
		content = append(content, AnthropicContent{Type: "text", Text: result.Text})
	}

	stopReason := "end_turn"
	if len(result.ToolCalls) > 0 {
		stopReason = "tool_use"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(AnthropicResponse{
		ID:         genID("msg"),
		Type:       "message",
		Role:       "assistant",
		Content:    content,
		Model:      req.Model,
		StopReason: stopReason,
		Usage: &AnthropicUsage{
			InputTokens:  result.PromptTokens,
			OutputTokens: result.CompletionTokens + result.ReasoningTokens,
		},
	})
}

// Anthropic streaming

func streamAnthropic(w http.ResponseWriter, prompt, variant string, req AnthropicRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		anthropicError(w, http.StatusInternalServerError, "streaming not supported", "api_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	msgID := genID("msg")

	sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
		Type: "message_start",
		Delta: map[string]interface{}{
			"type": "message", "id": msgID, "role": "assistant", "model": req.Model,
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, mimoBinary, buildMimoArgs(prompt, variant, "")...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		anthropicError(w, http.StatusInternalServerError, "failed to start mimo: "+err.Error(), "api_error")
		return
	}
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		anthropicError(w, http.StatusInternalServerError, "failed to start mimo: "+err.Error(), "api_error")
		return
	}

	var (
		promptTokens int
		compTokens   int
		reasonTokens int
		blockIndex   int
		inThinking   bool
		inText       bool
	)

	scanner := bufio.NewScanner(stdout)
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
		case "reasoning":
			var part MimoReasoningPart
			if err := json.Unmarshal(event.Part, &part); err == nil && part.Text != "" {
				if !inThinking {
					inThinking = true
					idx := blockIndex
					sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
						Type:  "content_block_start",
						Delta: map[string]interface{}{"index": idx, "content_block": map[string]interface{}{"type": "thinking"}},
					})
				}
				idx := blockIndex
				sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
					Type:  "content_block_delta",
					Delta: map[string]interface{}{"index": idx, "type": "thinking_delta", "thinking": part.Text},
				})
			}
		case "text":
			var part MimoTextPart
			if err := json.Unmarshal(event.Part, &part); err == nil && part.Text != "" {
				if inThinking {
					inThinking = false
					idx := blockIndex
					sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
						Type: "content_block_stop", Delta: map[string]interface{}{"index": idx},
					})
					blockIndex++
				}
				if !inText {
					inText = true
					idx := blockIndex
					sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
						Type:  "content_block_start",
						Delta: map[string]interface{}{"index": idx, "content_block": map[string]interface{}{"type": "text"}},
					})
				}
				idx := blockIndex
				sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
					Type:  "content_block_delta",
					Delta: map[string]interface{}{"index": idx, "type": "text_delta", "text": part.Text},
				})
			}
		case "step_finish":
			var part MimoStepFinishPart
			if err := json.Unmarshal(event.Part, &part); err == nil {
				promptTokens = part.Tokens.Input
				compTokens = part.Tokens.Output
				reasonTokens = part.Tokens.Reasoning
			}
		}
	}

	if inText {
		sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
			Type: "content_block_stop", Delta: map[string]interface{}{"index": blockIndex},
		})
	}
	if inThinking {
		sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
			Type: "content_block_stop", Delta: map[string]interface{}{"index": blockIndex},
		})
	}

	if err := cmd.Wait(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			log.Printf("mimo timeout after 300s")
		} else if stderr.Len() > 0 {
			log.Printf("mimo error: %s", strings.TrimSpace(stderr.String()))
		}
	}

	sendAnthropicEvent(w, flusher, AnthropicStreamEvent{
		Type:  "message_delta",
		Delta: map[string]interface{}{"stop_reason": "end_turn"},
		Usage: &AnthropicUsage{InputTokens: promptTokens, OutputTokens: compTokens + reasonTokens},
	})
	sendAnthropicEvent(w, flusher, AnthropicStreamEvent{Type: "message_stop"})
}

// SSE helpers

func sendSSE(w http.ResponseWriter, flusher http.Flusher, chunk StreamChunk) {
	data, err := json.Marshal(chunk)
	if err != nil {
		log.Printf("sendSSE marshal: %v", err)
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func sendAnthropicEvent(w http.ResponseWriter, flusher http.Flusher, event AnthropicStreamEvent) {
	data, err := json.Marshal(event)
	if err != nil {
		log.Printf("sendAnthropicEvent marshal: %v", err)
		return
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.Type, data)
	flusher.Flush()
}

// Error helpers

func sendError(w http.ResponseWriter, code int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{"message": message, "type": errType, "code": errType},
	})
}

func anthropicError(w http.ResponseWriter, code int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"type": "error",
		"error": map[string]string{"type": errType, "message": message},
	})
}
