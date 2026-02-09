package agent

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/rahulSailesh-shah/go-pi-ai/provider"
	"github.com/rahulSailesh-shah/go-pi-ai/types"
)

// AgentState contains all configuration and conversation data for an agent instance.
// It holds the system prompt, model provider, available tools, conversation history,
// and streaming state information.
type AgentState struct {
	// SystemPrompt is the initial instruction given to the LLM that defines the agent's behavior.
	SystemPrompt string

	// Model is the LLM provider used for generating responses.
	Model provider.Provider

	// Tools is the list of tools available to the agent for execution.
	Tools []AgentTool

	// Messages contains the conversation history.
	Messages []AgentMessage

	// IsStreaming indicates whether the agent is currently processing a request.
	IsStreaming bool

	// StreamMessage holds the partial message being streamed (nil when not streaming).
	StreamMessage AgentMessage

	// PendingToolCalls tracks tool calls that are currently being executed.
	PendingToolCalls map[string]bool

	// Error contains the last error message if an error occurred (nil otherwise).
	Error *string
}

// AgentOptions configures a new Agent instance.
type AgentOptions struct {
	// InitialState provides the initial configuration for the agent.
	InitialState *AgentState

	// ConvertToLlm is an optional function to filter/transform messages before sending to the LLM.
	// If not provided, a default implementation that passes user, assistant, and tool messages is used.
	ConvertToLlm func([]AgentMessage) ([]types.Message, error)

	// TransformContext is an optional function to transform the message context before each LLM call.
	TransformContext func([]AgentMessage, context.Context) ([]AgentMessage, error)

	// SteeringMode controls how steering messages are processed.
	// "one-at-a-time" (default): Process one steering message per turn.
	// "all": Process all queued steering messages at once.
	SteeringMode string

	// FollowUpMode controls how follow-up messages are processed.
	// "one-at-a-time" (default): Process one follow-up message per turn.
	// "all": Process all queued follow-up messages at once.
	FollowUpMode string

	// SessionId is an optional identifier for the conversation session.
	SessionId string
}

// Agent is the main interface for interacting with the agent loop.
// It provides a high-level API for managing conversations, handling events,
// and controlling the agent lifecycle.
//
// The Agent is thread-safe and supports concurrent access for event subscription,
// steering, and follow-up message queuing.
type Agent struct {
	state            AgentState
	listeners        map[int]func(AgentEvent)
	listeneeID       int
	listenersMu      sync.RWMutex
	abortCancel      context.CancelFunc
	abortCancelMu    sync.Mutex
	convertToLlm     func([]AgentMessage) ([]types.Message, error)
	transformContext func([]AgentMessage, context.Context) ([]AgentMessage, error)
	steeringQueue    []AgentMessage
	steeringQueueMu  sync.Mutex
	followUpQueue    []AgentMessage
	followUpQueueMu  sync.Mutex
	steeringMode     string
	followUpMode     string
	sessionId        string
	runningPrompt    chan struct{}
	runningPromptMu  sync.Mutex
	mu               sync.RWMutex
}

// defaultConvertToLlm filters messages to only LLM-compatible roles (user, assistant, tool).
func defaultConvertToLlm(messages []AgentMessage) ([]types.Message, error) {
	result := make([]types.Message, 0, len(messages))
	for _, m := range messages {
		role := m.Role()
		if role == "user" || role == "assistant" || role == "tool" {
			result = append(result, m)
		}
	}
	return result, nil
}

// NewAgent creates a new Agent instance with the given options.
// If opts is nil, the agent is created with default settings.
//
// Example:
//
//	agent := NewAgent(&AgentOptions{
//	    InitialState: &AgentState{
//	        SystemPrompt: "You are a helpful assistant.",
//	        Model:        model,
//	        Tools:        tools,
//	    },
//	})
func NewAgent(opts *AgentOptions) *Agent {
	state := AgentState{
		SystemPrompt:     "",
		Tools:            []AgentTool{},
		Messages:         []AgentMessage{},
		IsStreaming:      false,
		PendingToolCalls: make(map[string]bool),
		Error:            nil,
	}

	if opts != nil && opts.InitialState != nil {
		if opts.InitialState.SystemPrompt != "" {
			state.SystemPrompt = opts.InitialState.SystemPrompt
		}
		if opts.InitialState.Model != nil {
			state.Model = opts.InitialState.Model
		}
		if opts.InitialState.Tools != nil {
			state.Tools = opts.InitialState.Tools
		}
		if opts.InitialState.Messages != nil {
			state.Messages = opts.InitialState.Messages
		}
		if opts.InitialState.PendingToolCalls != nil {
			state.PendingToolCalls = opts.InitialState.PendingToolCalls
		} else {
			state.PendingToolCalls = make(map[string]bool)
		}
	}

	convertToLlm := defaultConvertToLlm
	if opts != nil && opts.ConvertToLlm != nil {
		convertToLlm = opts.ConvertToLlm
	}

	steeringMode := "one-at-a-time"
	if opts != nil && opts.SteeringMode != "" {
		steeringMode = opts.SteeringMode
	}

	followUpMode := "one-at-a-time"
	if opts != nil && opts.FollowUpMode != "" {
		followUpMode = opts.FollowUpMode
	}

	sessionId := ""
	if opts != nil {
		sessionId = opts.SessionId
	}

	var transformContext func([]AgentMessage, context.Context) ([]AgentMessage, error)
	if opts != nil {
		transformContext = opts.TransformContext
	}

	return &Agent{
		state:            state,
		listeners:        make(map[int]func(AgentEvent)),
		listeneeID:       0,
		convertToLlm:     convertToLlm,
		transformContext: transformContext,
		steeringQueue:    []AgentMessage{},
		followUpQueue:    []AgentMessage{},
		steeringMode:     steeringMode,
		followUpMode:     followUpMode,
		sessionId:        sessionId,
		runningPrompt:    nil,
	}
}

// State returns a deep copy of the current agent state.
// This is safe to call concurrently.
func (a *Agent) State() AgentState {
	a.mu.RLock()
	defer a.mu.RUnlock()

	// Create a deep copy
	state := a.state
	state.Messages = make([]AgentMessage, len(a.state.Messages))
	copy(state.Messages, a.state.Messages)
	state.Tools = make([]AgentTool, len(a.state.Tools))
	copy(state.Tools, a.state.Tools)
	state.PendingToolCalls = make(map[string]bool)
	for k, v := range a.state.PendingToolCalls {
		state.PendingToolCalls[k] = v
	}
	if a.state.Error != nil {
		errCopy := *a.state.Error
		state.Error = &errCopy
	}

	return state
}

// SetSystemPrompt sets the system prompt for the agent.
func (a *Agent) SetSystemPrompt(v string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.SystemPrompt = v
}

// SetModel sets the model provider for the agent.
func (a *Agent) SetModel(m provider.Provider) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Model = m
}

// SetTools sets the available tools for the agent.
func (a *Agent) SetTools(t []AgentTool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Tools = t
}

// SetSteeringMode sets the steering mode ("all" or "one-at-a-time").
func (a *Agent) SetSteeringMode(mode string) {
	a.steeringMode = mode
}

// GetSteeringMode returns the current steering mode.
func (a *Agent) GetSteeringMode() string {
	return a.steeringMode
}

// SetFollowUpMode sets the follow-up mode ("all" or "one-at-a-time").
func (a *Agent) SetFollowUpMode(mode string) {
	a.followUpMode = mode
}

// GetFollowUpMode returns the current follow-up mode.
func (a *Agent) GetFollowUpMode() string {
	return a.followUpMode
}

// SessionId returns the current session ID.
func (a *Agent) SessionId() string {
	return a.sessionId
}

// SetSessionId sets the session ID.
func (a *Agent) SetSessionId(value string) {
	a.sessionId = value
}

// ReplaceMessages replaces all messages in the conversation history.
func (a *Agent) ReplaceMessages(ms []AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = make([]AgentMessage, len(ms))
	copy(a.state.Messages, ms)
}

// AppendMessage appends a message to the conversation history.
func (a *Agent) AppendMessage(m AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = append(a.state.Messages, m)
}

// ClearMessages clears all messages from the conversation history.
func (a *Agent) ClearMessages() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = []AgentMessage{}
}

// Steer queues a steering message to interrupt the agent mid-run.
// Steering messages are processed during tool execution, allowing you to
// redirect the agent's behavior before it completes its current task.
func (a *Agent) Steer(m AgentMessage) {
	a.steeringQueueMu.Lock()
	defer a.steeringQueueMu.Unlock()
	a.steeringQueue = append(a.steeringQueue, m)
}

// FollowUp queues a follow-up message to be processed after the agent finishes
// its current task. Unlike steering, follow-up messages wait for the current
// execution to complete before being processed.
func (a *Agent) FollowUp(m AgentMessage) {
	a.followUpQueueMu.Lock()
	defer a.followUpQueueMu.Unlock()
	a.followUpQueue = append(a.followUpQueue, m)
}

// ClearSteeringQueue clears all pending steering messages.
func (a *Agent) ClearSteeringQueue() {
	a.steeringQueueMu.Lock()
	defer a.steeringQueueMu.Unlock()
	a.steeringQueue = []AgentMessage{}
}

// ClearFollowUpQueue clears all pending follow-up messages.
func (a *Agent) ClearFollowUpQueue() {
	a.followUpQueueMu.Lock()
	defer a.followUpQueueMu.Unlock()
	a.followUpQueue = []AgentMessage{}
}

// ClearAllQueues clears both steering and follow-up message queues.
func (a *Agent) ClearAllQueues() {
	a.ClearSteeringQueue()
	a.ClearFollowUpQueue()
}

// Subscribe adds an event listener and returns an unsubscribe function.
// The listener will be called for every event emitted by the agent.
//
// Example:
//
//	unsubscribe := agent.Subscribe(func(e AgentEvent) {
//	    switch ev := e.(type) {
//	    case MessageUpdate:
//	        // Handle streaming updates
//	    case AgentEnd:
//	        // Handle completion
//	    }
//	})
//	defer unsubscribe()
func (a *Agent) Subscribe(fn func(AgentEvent)) func() {
	a.listenersMu.Lock()
	defer a.listenersMu.Unlock()

	id := a.listeneeID
	a.listeneeID++
	a.listeners[id] = fn

	return func() {
		a.listenersMu.Lock()
		defer a.listenersMu.Unlock()
		delete(a.listeners, id)
	}
}

// emit sends an event to all listeners.
func (a *Agent) emit(e AgentEvent) {
	a.listenersMu.RLock()
	defer a.listenersMu.RUnlock()
	for _, listener := range a.listeners {
		listener(e)
	}
}

// Abort cancels the current operation.
// This will stop any ongoing LLM request and tool execution.
func (a *Agent) Abort() {
	a.abortCancelMu.Lock()
	defer a.abortCancelMu.Unlock()
	if a.abortCancel != nil {
		a.abortCancel()
	}
}

// WaitForIdle returns a channel that closes when the agent is idle.
// Use this to wait for the agent to finish processing before reading results.
//
// Example:
//
//	agent.Prompt("Hello!")
//	<-agent.WaitForIdle()
//	state := agent.State()
func (a *Agent) WaitForIdle() <-chan struct{} {
	a.runningPromptMu.Lock()
	defer a.runningPromptMu.Unlock()

	if a.runningPrompt == nil {
		// Already idle, return closed channel
		ch := make(chan struct{})
		close(ch)
		return ch
	}

	return a.runningPrompt
}

// Reset resets the agent state, clearing messages, queues, and error state.
// The system prompt, model, and tools are preserved.
func (a *Agent) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.Messages = []AgentMessage{}
	a.state.IsStreaming = false
	a.state.StreamMessage = nil
	a.state.PendingToolCalls = make(map[string]bool)
	a.state.Error = nil

	a.ClearAllQueues()
}

// Prompt sends a prompt to the agent and starts processing.
// The input can be a string, AgentMessage, or []AgentMessage.
// Optional images can be attached when using a string input.
//
// Returns an error if the agent is already processing or no model is configured.
//
// Example:
//
//	// Simple string prompt
//	err := agent.Prompt("What is the weather?")
//
//	// With images
//	err := agent.Prompt("Describe this image", imageContent)
//
//	// With custom message
//	err := agent.Prompt(types.UserMessage{...})
func (a *Agent) Prompt(input interface{}, images ...types.ImageContent) error {
	a.mu.RLock()
	if a.state.IsStreaming {
		a.mu.RUnlock()
		return errors.New("agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion")
	}
	a.mu.RUnlock()

	a.mu.RLock()
	model := a.state.Model
	a.mu.RUnlock()

	if model == nil {
		return errors.New("no model configured")
	}

	var msgs []AgentMessage

	switch v := input.(type) {
	case string:
		content := []types.Content{types.TextContent{Text: v}}
		for _, img := range images {
			content = append(content, img)
		}
		msgs = []AgentMessage{
			types.UserMessage{
				Timestamp: time.Now(),
				Contents:  content,
			},
		}
	case AgentMessage:
		msgs = []AgentMessage{v}
	case []AgentMessage:
		msgs = v
	default:
		return errors.New("invalid input type for Prompt")
	}

	return a.runLoop(msgs)
}

// Continue continues from the current context without adding a new message.
// This is useful for retrying after an error or continuing after tool results.
//
// Returns an error if the agent is already processing, there are no messages,
// or the last message is from the assistant.
func (a *Agent) Continue() error {
	a.mu.RLock()
	if a.state.IsStreaming {
		a.mu.RUnlock()
		return errors.New("agent is already processing. Wait for completion before continuing")
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

	return a.runLoop(nil)
}

// runLoop runs the agent loop with optional new messages.
func (a *Agent) runLoop(messages []AgentMessage) error {
	a.mu.RLock()
	model := a.state.Model
	a.mu.RUnlock()

	if model == nil {
		return errors.New("no model configured")
	}

	// Set up running prompt channel
	a.runningPromptMu.Lock()
	done := make(chan struct{})
	a.runningPrompt = done
	a.runningPromptMu.Unlock()

	// Create context for cancellation
	ctx, cancel := context.WithCancel(context.Background())
	a.abortCancelMu.Lock()
	a.abortCancel = cancel
	a.abortCancelMu.Unlock()

	// Update state to streaming
	a.mu.Lock()
	a.state.IsStreaming = true
	a.state.StreamMessage = nil
	a.state.Error = nil
	a.mu.Unlock()

	// Build context from state
	a.mu.RLock()
	agentContext := AgentContext{
		SystemPrompt: a.state.SystemPrompt,
		Messages:     make([]AgentMessage, len(a.state.Messages)),
		Tools:        a.state.Tools,
	}
	copy(agentContext.Messages, a.state.Messages)
	a.mu.RUnlock()

	// Build config with closures for steering/follow-up
	config := AgentLoopConfig{
		Model:     model,
		SessionId: a.sessionId,
		GetSteeringMessages: func() ([]AgentMessage, error) {
			a.steeringQueueMu.Lock()
			defer a.steeringQueueMu.Unlock()

			if a.steeringMode == "one-at-a-time" {
				if len(a.steeringQueue) > 0 {
					first := a.steeringQueue[0]
					a.steeringQueue = a.steeringQueue[1:]
					return []AgentMessage{first}, nil
				}
				return []AgentMessage{}, nil
			} else {
				steering := make([]AgentMessage, len(a.steeringQueue))
				copy(steering, a.steeringQueue)
				a.steeringQueue = []AgentMessage{}
				return steering, nil
			}
		},
		GetFollowUpMessages: func() ([]AgentMessage, error) {
			a.followUpQueueMu.Lock()
			defer a.followUpQueueMu.Unlock()

			if a.followUpMode == "one-at-a-time" {
				if len(a.followUpQueue) > 0 {
					first := a.followUpQueue[0]
					a.followUpQueue = a.followUpQueue[1:]
					return []AgentMessage{first}, nil
				}
				return []AgentMessage{}, nil
			} else {
				followUp := make([]AgentMessage, len(a.followUpQueue))
				copy(followUp, a.followUpQueue)
				a.followUpQueue = []AgentMessage{}
				return followUp, nil
			}
		},
	}

	var stream AgentEventStream
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

	// Process events
	go func() {
		defer a.cleanup(done)

		var partial AgentMessage
		eventsDone := false

		for !eventsDone {
			select {
			case event, ok := <-stream.Events:
				if !ok {
					eventsDone = true
					break
				}

				// Update state based on event type
				switch e := event.(type) {
				case MessageStart:
					partial = e.Message
					a.mu.Lock()
					a.state.StreamMessage = e.Message
					a.mu.Unlock()

				case MessageUpdate:
					// MessageUpdate may have nil Message in current agentloop.go implementation
					// Update only if Message is set
					if e.Message != nil {
						partial = e.Message
						a.mu.Lock()
						a.state.StreamMessage = e.Message
						a.mu.Unlock()
					}

				case MessageEnd:
					partial = nil
					a.mu.Lock()
					a.state.StreamMessage = nil
					a.mu.Unlock()
					a.AppendMessage(e.Message)

				case ToolExecutionStart:
					a.mu.Lock()

					a.state.PendingToolCalls[e.ToolCallId] = true
					a.mu.Unlock()

				case ToolExecutionEnd:
					a.mu.Lock()
					delete(a.state.PendingToolCalls, e.ToolCallId)
					a.mu.Unlock()

				case TurnEnd:
					// Check for errors in assistant message
					if e.Message.Role() == "assistant" {
						if am, ok := e.Message.(types.AssistantMessage); ok && am.ErrorMessage != nil {
							a.mu.Lock()
							a.state.Error = am.ErrorMessage
							a.mu.Unlock()
						}
					}

				case AgentEnd:
					a.mu.Lock()
					a.state.IsStreaming = false
					a.state.StreamMessage = nil
					a.mu.Unlock()
				}

				// Emit to listeners
				a.emit(event)

			case <-ctx.Done():
				// Context was cancelled
				errMsg := "Request was aborted"
				a.mu.Lock()
				a.state.Error = &errMsg
				a.mu.Unlock()

				errorMessage := types.AssistantMessage{
					Contents:     []types.Content{types.TextContent{Text: ""}},
					Timestamp:    time.Now(),
					ErrorMessage: &errMsg,
					StopReason:   types.StopReasonAborted,
				}

				a.AppendMessage(errorMessage)
				a.emit(AgentEnd{Messages: []AgentMessage{errorMessage}})

				// Safety: drain channels in background to prevent goroutine leaks
				go func() {
					for range stream.Events {
					}
					select {
					case <-stream.Result:
					default:
					}
					select {
					case <-stream.Err:
					default:
					}
				}()

				return
			}
		}

		// Events channel closed, now wait for Result or Err
		select {
		case finalMessages := <-stream.Result:
			// Successfully completed - finalMessages already processed via events
			_ = finalMessages // Use if needed for validation

		case err := <-stream.Err:
			if err != nil {
				// Create error message
				errMsg := err.Error()
				a.mu.Lock()
				a.state.Error = &errMsg
				a.mu.Unlock()

				errorMessage := types.AssistantMessage{
					Contents:     []types.Content{types.TextContent{Text: ""}},
					Timestamp:    time.Now(),
					ErrorMessage: &errMsg,
					StopReason:   types.StopReasonError,
				}

				a.AppendMessage(errorMessage)
				a.emit(AgentEnd{Messages: []AgentMessage{errorMessage}})
			}

		case <-ctx.Done():
			// Context was cancelled
			errMsg := "Request was aborted"
			a.mu.Lock()
			a.state.Error = &errMsg
			a.mu.Unlock()

			errorMessage := types.AssistantMessage{
				Contents:     []types.Content{types.TextContent{Text: ""}},
				Timestamp:    time.Now(),
				ErrorMessage: &errMsg,
				StopReason:   types.StopReasonAborted,
			}

			a.AppendMessage(errorMessage)
			a.emit(AgentEnd{Messages: []AgentMessage{errorMessage}})
		}

		// Handle any remaining partial message
		if partial != nil && partial.Role() == "assistant" {
			contents := partial.Content()
			hasContent := false
			for _, c := range contents {
				switch t := c.(type) {
				case types.TextContent:
					if t.Text != "" {
						hasContent = true
						break
					}
				case types.ToolCall:
					if t.Name != "" {
						hasContent = true
						break
					}
				}
			}
			if hasContent {
				a.AppendMessage(partial)
			}
		}
	}()

	return nil
}

// cleanup resets streaming state and closes the running prompt channel.
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
