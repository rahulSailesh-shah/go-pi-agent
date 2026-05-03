package agent

import (
	"context"
	"errors"
	"io"
	"maps"
	"sync"
	"time"

	gopiai "github.com/rahulSailesh-shah/go-pi-ai"
)

// AgentState contains all configuration and conversation data for an agent instance.
type AgentState struct {
	SystemPrompt     string
	Provider         gopiai.Provider
	ModelName        string
	Tools            []AgentTool
	Messages         []Message
	IsStreaming      bool
	StreamMessage    Message
	PendingToolCalls map[string]bool
	Error            *string
}

type agentConfig struct {
	initialState *AgentState
	sessionID    string
}

// AgentOption configures a new Agent instance.
type AgentOption func(*agentConfig)

func WithInitialState(state *AgentState) AgentOption {
	return func(c *agentConfig) {
		c.initialState = state
	}
}

func WithSessionID(id string) AgentOption {
	return func(c *agentConfig) {
		c.sessionID = id
	}
}

// Agent is the main interface for interacting with the agent loop.
// It provides state management, event subscription, and lifecycle control.
// All public methods are thread-safe.
type Agent struct {
	state         AgentState
	listeners     map[int]func(AgentEvent)
	listenerID    int
	listenersMu   sync.RWMutex
	abortCancel   context.CancelFunc
	sessionID     string
	runningPrompt chan struct{}
	mu            sync.RWMutex
}

// NewAgent creates a new Agent instance with the given options.
func NewAgent(opts ...AgentOption) *Agent {
	cfg := &agentConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	state := AgentState{
		Tools:            []AgentTool{},
		Messages:         []Message{},
		PendingToolCalls: make(map[string]bool),
	}

	if cfg.initialState != nil {
		s := cfg.initialState
		if s.SystemPrompt != "" {
			state.SystemPrompt = s.SystemPrompt
		}
		if s.Provider != nil {
			state.Provider = s.Provider
		}
		if s.ModelName != "" {
			state.ModelName = s.ModelName
		}
		if s.Tools != nil {
			state.Tools = s.Tools
		}
		if s.Messages != nil {
			state.Messages = s.Messages
		}
		if s.PendingToolCalls != nil {
			state.PendingToolCalls = s.PendingToolCalls
		}
	}

	return &Agent{
		state:     state,
		listeners: make(map[int]func(AgentEvent)),
		sessionID: cfg.sessionID,
	}
}

// State returns a copy of the current agent state.
func (a *Agent) State() AgentState {
	a.mu.RLock()
	defer a.mu.RUnlock()

	state := a.state
	state.Messages = make([]Message, len(a.state.Messages))
	copy(state.Messages, a.state.Messages)
	state.Tools = make([]AgentTool, len(a.state.Tools))
	copy(state.Tools, a.state.Tools)
	state.PendingToolCalls = make(map[string]bool, len(a.state.PendingToolCalls))
	maps.Copy(state.PendingToolCalls, a.state.PendingToolCalls)
	if a.state.Error != nil {
		errCopy := *a.state.Error
		state.Error = &errCopy
	}

	return state
}

func (a *Agent) SetSystemPrompt(v string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.SystemPrompt = v
}

func (a *Agent) SetProvider(m gopiai.Provider) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Provider = m
}

func (a *Agent) SetModelName(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.ModelName = name
}

func (a *Agent) SetTools(t []AgentTool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Tools = t
}

func (a *Agent) SessionID() string {
	return a.sessionID
}

func (a *Agent) SetSessionID(value string) {
	a.sessionID = value
}

func (a *Agent) ReplaceMessages(ms []Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = make([]Message, len(ms))
	copy(a.state.Messages, ms)
}

func (a *Agent) AppendMessage(m Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = append(a.state.Messages, m)
}

func (a *Agent) ClearMessages() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = []Message{}
}

// Subscribe adds an event listener and returns an unsubscribe function.
func (a *Agent) Subscribe(fn func(AgentEvent)) func() {
	a.listenersMu.Lock()
	defer a.listenersMu.Unlock()

	id := a.listenerID
	a.listenerID++
	a.listeners[id] = fn

	return func() {
		a.listenersMu.Lock()
		defer a.listenersMu.Unlock()
		delete(a.listeners, id)
	}
}

func (a *Agent) emit(e AgentEvent) {
	a.listenersMu.RLock()
	defer a.listenersMu.RUnlock()
	for _, listener := range a.listeners {
		listener(e)
	}
}

// Abort cancels the current operation.
func (a *Agent) Abort() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.abortCancel != nil {
		a.abortCancel()
	}
}

// WaitForIdle returns a channel that closes when the agent is idle.
func (a *Agent) WaitForIdle() <-chan struct{} {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.runningPrompt == nil {
		ch := make(chan struct{})
		close(ch)
		return ch
	}

	return a.runningPrompt
}

// Reset clears messages, queues, and error state. Preserves system prompt, model, and tools.
func (a *Agent) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.Messages = []Message{}
	a.state.IsStreaming = false
	a.state.StreamMessage = nil
	a.state.PendingToolCalls = make(map[string]bool)
	a.state.Error = nil
}

// Prompt sends a prompt to the agent and starts processing.
// Input can be a string, Message, or []Message.
// Optional images can be attached when using a string input.
func (a *Agent) Prompt(ctx context.Context, input any, images ...gopiai.ImageContent) error {
	a.mu.RLock()
	if a.state.IsStreaming {
		a.mu.RUnlock()
		return errors.New("agent is already processing a prompt")
	}
	a.mu.RUnlock()

	a.mu.RLock()
	model := a.state.Provider
	a.mu.RUnlock()

	if model == nil {
		return errors.New("no model configured")
	}

	var msgs []Message

	switch v := input.(type) {
	case string:
		content := []gopiai.Content{gopiai.TextContent{Text: v}}
		for _, img := range images {
			content = append(content, img)
		}
		msgs = []Message{
			gopiai.UserMessage{
				Timestamp: time.Now(),
				Contents:  content,
			},
		}
	case Message:
		msgs = []Message{v}
	case []Message:
		msgs = v
	default:
		return errors.New("invalid input type for Prompt")
	}

	return a.run(ctx, msgs)
}

// Continue continues from the current context without adding a new message.
func (a *Agent) Continue(ctx context.Context) error {
	a.mu.RLock()
	if a.state.IsStreaming {
		a.mu.RUnlock()
		return errors.New("agent is already processing")
	}

	messages := a.state.Messages
	a.mu.RUnlock()

	if len(messages) == 0 {
		return errors.New("no messages to continue from")
	}

	lastMsg := messages[len(messages)-1]
	if lastMsg.Role() == "assistant" {
		return errors.New("cannot continue from message role: assistant")
	}

	return a.run(ctx, nil)
}

func (a *Agent) run(ctx context.Context, prompts []Message) error {
	a.mu.RLock()
	model := a.state.Provider
	a.mu.RUnlock()

	if model == nil {
		return errors.New("no model configured")
	}

	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)

	a.mu.Lock()
	done := make(chan struct{})
	a.runningPrompt = done
	a.abortCancel = cancel
	a.state.IsStreaming = true
	a.state.StreamMessage = nil
	a.state.Error = nil
	a.mu.Unlock()

	a.mu.RLock()
	agentContext := AgentContext{
		SystemPrompt: a.state.SystemPrompt,
		Messages:     make([]Message, len(a.state.Messages)),
		Tools:        a.state.Tools,
	}
	copy(agentContext.Messages, a.state.Messages)
	a.mu.RUnlock()

	config := AgentLoopConfig{
		Provider:  model,
		ModelName: a.state.ModelName,
		SessionID: a.sessionID,
	}

	var stream *Stream
	if prompts != nil {
		stream = AgentLoop(ctx, prompts, agentContext, config)
	} else {
		var err error
		stream, err = AgentLoopContinue(ctx, agentContext, config)
		if err != nil {
			a.cleanup(done)
			return err
		}
	}

	go func() {
		defer a.cleanup(done)
		defer stream.Close()

		for {
			event, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				errMsg := err.Error()
				a.mu.Lock()
				a.state.Error = &errMsg
				a.mu.Unlock()

				// Capture whatever was already streamed into state
				a.mu.RLock()
				partialMsg := a.state.StreamMessage
				a.mu.RUnlock()

				var finalMsg Message
				if partialMsg != nil {
					finalMsg = partialMsg
				} else {
					finalMsg = gopiai.AssistantMessage{
						Contents:   []gopiai.Content{gopiai.TextContent{Text: ""}},
						Timestamp:  time.Now(),
						StopReason: gopiai.StopReasonError,
					}
				}

				a.AppendMessage(finalMsg)
				a.emit(AgentError{Error: err})
				a.emit(AgentEnd{Messages: []Message{finalMsg}})
				return
			}

			switch e := event.(type) {
			case MessageStart:
				a.mu.Lock()
				a.state.StreamMessage = e.Message
				a.mu.Unlock()

			case MessageUpdate:
				if e.Message != nil {
					a.mu.Lock()
					a.state.StreamMessage = e.Message
					a.mu.Unlock()
				}

			case MessageEnd:
				a.mu.Lock()
				a.state.StreamMessage = nil
				a.mu.Unlock()
				a.AppendMessage(e.Message)

			case ToolExecutionStart:
				a.mu.Lock()
				a.state.PendingToolCalls[e.ToolCallID] = true
				a.mu.Unlock()

			case ToolExecutionEnd:
				a.mu.Lock()
				delete(a.state.PendingToolCalls, e.ToolCallID)
				a.mu.Unlock()
				a.AppendMessage(e.Result)

			case AgentEnd:
				a.mu.Lock()
				a.state.IsStreaming = false
				a.state.StreamMessage = nil
				a.mu.Unlock()
			}

			a.emit(event)
		}
	}()

	return nil
}

func (a *Agent) cleanup(done chan struct{}) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.IsStreaming = false
	a.state.StreamMessage = nil
	a.state.PendingToolCalls = make(map[string]bool)
	a.abortCancel = nil

	if a.runningPrompt == done {
		a.runningPrompt = nil
	}
	close(done)
}
