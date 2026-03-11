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
	Model            gopiai.Provider
	ModelName        string
	Tools            []AgentTool
	Messages         []Message
	IsStreaming      bool
	StreamMessage    Message
	PendingToolCalls map[string]bool
	Error            *string
}

// AgentOptions configures a new Agent instance.
type AgentOptions struct {
	InitialState *AgentState

	// SteeringMode controls how steering messages are processed.
	// "one-at-a-time" (default) or "all".
	SteeringMode string

	// FollowUpMode controls how follow-up messages are processed.
	// "one-at-a-time" (default) or "all".
	FollowUpMode string

	// SessionID is an optional identifier for the conversation session.
	SessionID string
}

// Agent is the main interface for interacting with the agent loop.
// It provides state management, event subscription, and lifecycle control.
// All public methods are thread-safe.
type Agent struct {
	state           AgentState
	listeners       map[int]func(AgentEvent)
	listenerID      int
	listenersMu     sync.RWMutex
	abortCancel     context.CancelFunc
	abortCancelMu   sync.Mutex
	steeringQueue   []Message
	steeringQueueMu sync.Mutex
	followUpQueue   []Message
	followUpQueueMu sync.Mutex
	steeringMode    string
	followUpMode    string
	sessionID       string
	runningPrompt   chan struct{}
	runningPromptMu sync.Mutex
	mu              sync.RWMutex
}

// NewAgent creates a new Agent instance with the given options.
func NewAgent(opts *AgentOptions) *Agent {
	state := AgentState{
		Tools:            []AgentTool{},
		Messages:         []Message{},
		PendingToolCalls: make(map[string]bool),
	}

	if opts != nil && opts.InitialState != nil {
		s := opts.InitialState
		if s.SystemPrompt != "" {
			state.SystemPrompt = s.SystemPrompt
		}
		if s.Model != nil {
			state.Model = s.Model
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

	steeringMode := "one-at-a-time"
	followUpMode := "one-at-a-time"
	sessionID := ""

	if opts != nil {
		if opts.SteeringMode != "" {
			steeringMode = opts.SteeringMode
		}
		if opts.FollowUpMode != "" {
			followUpMode = opts.FollowUpMode
		}
		sessionID = opts.SessionID
	}

	return &Agent{
		state:         state,
		listeners:     make(map[int]func(AgentEvent)),
		steeringQueue: []Message{},
		followUpQueue: []Message{},
		steeringMode:  steeringMode,
		followUpMode:  followUpMode,
		sessionID:     sessionID,
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

func (a *Agent) SetModel(m gopiai.Provider) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Model = m
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

func (a *Agent) SetSteeringMode(mode string) {
	a.steeringMode = mode
}

func (a *Agent) GetSteeringMode() string {
	return a.steeringMode
}

func (a *Agent) SetFollowUpMode(mode string) {
	a.followUpMode = mode
}

func (a *Agent) GetFollowUpMode() string {
	return a.followUpMode
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

// Steer queues a steering message to interrupt the agent mid-run.
func (a *Agent) Steer(m Message) {
	a.steeringQueueMu.Lock()
	defer a.steeringQueueMu.Unlock()
	a.steeringQueue = append(a.steeringQueue, m)
}

// FollowUp queues a follow-up message to be processed after the current task.
func (a *Agent) FollowUp(m Message) {
	a.followUpQueueMu.Lock()
	defer a.followUpQueueMu.Unlock()
	a.followUpQueue = append(a.followUpQueue, m)
}

func (a *Agent) ClearSteeringQueue() {
	a.steeringQueueMu.Lock()
	defer a.steeringQueueMu.Unlock()
	a.steeringQueue = []Message{}
}

func (a *Agent) ClearFollowUpQueue() {
	a.followUpQueueMu.Lock()
	defer a.followUpQueueMu.Unlock()
	a.followUpQueue = []Message{}
}

func (a *Agent) ClearAllQueues() {
	a.ClearSteeringQueue()
	a.ClearFollowUpQueue()
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
	a.abortCancelMu.Lock()
	defer a.abortCancelMu.Unlock()
	if a.abortCancel != nil {
		a.abortCancel()
	}
}

// WaitForIdle returns a channel that closes when the agent is idle.
func (a *Agent) WaitForIdle() <-chan struct{} {
	a.runningPromptMu.Lock()
	defer a.runningPromptMu.Unlock()

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

	a.ClearAllQueues()
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
	model := a.state.Model
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

func (a *Agent) run(ctx context.Context, messages []Message) error {
	a.mu.RLock()
	model := a.state.Model
	a.mu.RUnlock()

	if model == nil {
		return errors.New("no model configured")
	}

	a.runningPromptMu.Lock()
	done := make(chan struct{})
	a.runningPrompt = done
	a.runningPromptMu.Unlock()

	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	a.abortCancelMu.Lock()
	a.abortCancel = cancel
	a.abortCancelMu.Unlock()

	a.mu.Lock()
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
		Model:     model,
		ModelName: a.state.ModelName,
		SessionID: a.sessionID,
		GetSteeringMessages: func() ([]Message, error) {
			a.steeringQueueMu.Lock()
			defer a.steeringQueueMu.Unlock()

			if a.steeringMode == "one-at-a-time" {
				if len(a.steeringQueue) > 0 {
					first := a.steeringQueue[0]
					a.steeringQueue = a.steeringQueue[1:]
					return []Message{first}, nil
				}
				return nil, nil
			}
			steering := make([]Message, len(a.steeringQueue))
			copy(steering, a.steeringQueue)
			a.steeringQueue = []Message{}
			return steering, nil
		},
		GetFollowUpMessages: func() ([]Message, error) {
			a.followUpQueueMu.Lock()
			defer a.followUpQueueMu.Unlock()

			if a.followUpMode == "one-at-a-time" {
				if len(a.followUpQueue) > 0 {
					first := a.followUpQueue[0]
					a.followUpQueue = a.followUpQueue[1:]
					return []Message{first}, nil
				}
				return nil, nil
			}
			followUp := make([]Message, len(a.followUpQueue))
			copy(followUp, a.followUpQueue)
			a.followUpQueue = []Message{}
			return followUp, nil
		},
	}

	var stream *Stream
	if messages != nil {
		stream = AgentLoop(ctx, messages, agentContext, config)
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

				errorMessage := gopiai.AssistantMessage{
					Contents:   []gopiai.Content{gopiai.TextContent{Text: ""}},
					Timestamp:  time.Now(),
					StopReason: gopiai.StopReasonError,
				}

				a.AppendMessage(errorMessage)
				a.emit(AgentEnd{Messages: []Message{errorMessage}})
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
	a.state.IsStreaming = false
	a.state.StreamMessage = nil
	a.state.PendingToolCalls = make(map[string]bool)
	a.mu.Unlock()

	a.abortCancelMu.Lock()
	a.abortCancel = nil
	a.abortCancelMu.Unlock()

	a.runningPromptMu.Lock()
	if a.runningPrompt == done {
		a.runningPrompt = nil
	}
	close(done)
	a.runningPromptMu.Unlock()
}
