package agent

import (
	"context"
	"io"
	"sync"

	gopiai "github.com/rahulSailesh-shah/go-pi-ai"
)

// Re-export gopiai types so consumers only need to import the agent package.
type (
	Message          = gopiai.Message
	Content          = gopiai.Content
	TextContent      = gopiai.TextContent
	ImageContent     = gopiai.ImageContent
	ToolCall         = gopiai.ToolCall
	UserMessage      = gopiai.UserMessage
	AssistantMessage = gopiai.AssistantMessage
	ToolMessage      = gopiai.ToolMessage
	Tool             = gopiai.Tool
	StopReason       = gopiai.StopReason
	Provider         = gopiai.Provider
	Request          = gopiai.Request
)

// Re-export gopiai streaming event types.
type (
	Event              = gopiai.Event
	EventStart         = gopiai.EventStart
	EventTextStart     = gopiai.EventTextStart
	EventTextDelta     = gopiai.EventTextDelta
	EventTextEnd       = gopiai.EventTextEnd
	EventToolcallStart = gopiai.EventToolcallStart
	EventToolcallDelta = gopiai.EventToolcallDelta
	EventToolcallEnd   = gopiai.EventToolcallEnd
	EventDone          = gopiai.EventDone
	EventError         = gopiai.EventError
)

// Re-export gopiai stop reason constants.
var (
	StopReasonStop    = gopiai.StopReasonStop
	StopReasonLength  = gopiai.StopReasonLength
	StopReasonToolUse = gopiai.StopReasonToolUse
	StopReasonAborted = gopiai.StopReasonAborted
	StopReasonError   = gopiai.StopReasonError
	StopReasonUnknown = gopiai.StopReasonUnknown
)

// AgentTool defines a tool that can be executed by the agent.
// It combines the tool schema (Name, Description, Parameters) with
// an Execute function that performs the actual work.
type AgentTool struct {
	gopiai.Tool

	// Label is an optional human-readable label for UI display.
	Label string

	// Execute is called when the LLM requests this tool.
	// It receives the tool call ID (for tracking) and the parsed arguments.
	// Returns the tool result message or an error.
	Execute func(toolCallID string, params map[string]any) (ToolMessage, error)
}

// AgentContext holds the context for an agent loop execution.
type AgentContext struct {
	SystemPrompt string
	Messages     []Message
	Tools        []AgentTool
}

// AgentLoopConfig configures the behavior of the agent loop.
type AgentLoopConfig struct {
	// Model is the LLM provider used for generating responses.
	Model gopiai.Provider

	// ModelName is the model identifier (e.g. "gpt-4o").
	ModelName string

	// SessionID is an optional identifier for the conversation session.
	SessionID string

	// GetSteeringMessages is called during execution to check for steering messages.
	// Steering messages can interrupt tool execution to redirect the agent.
	// Return nil if no steering messages are available.
	GetSteeringMessages func() ([]Message, error)

	// GetFollowUpMessages is called after the agent completes to check for follow-up messages.
	// Follow-up messages are processed in a new turn after the current execution completes.
	// Return nil if no follow-up messages are available.
	GetFollowUpMessages func() ([]Message, error)
}

// Stream provides an iterator-based API for consuming agent events.
// Call Recv() in a loop until it returns io.EOF (complete) or an error.
// Always call Close() when done (use defer).
type Stream struct {
	events chan AgentEvent
	cancel context.CancelFunc
	ctx    context.Context
	once   sync.Once
	err    error
}

// NewStream creates a new Stream. The provided context is used for cancellation.
// Close() also cancels the stream's context.
// Returns the stream and a send-only channel for the producer.
func NewStream(ctx context.Context) (*Stream, chan<- AgentEvent) {
	ctx, cancel := context.WithCancel(ctx)
	ch := make(chan AgentEvent)
	s := &Stream{
		events: ch,
		ctx:    ctx,
		cancel: cancel,
	}
	return s, ch
}

// Recv returns the next event from the stream.
// Returns io.EOF when the stream is complete.
func (s *Stream) Recv() (AgentEvent, error) {
	event, ok := <-s.events
	if !ok {
		if s.err != nil {
			return nil, s.err
		}
		return nil, io.EOF
	}
	if e, ok := event.(AgentError); ok {
		s.err = e.Error
		return nil, e.Error
	}
	return event, nil
}

// Close signals that the consumer is done reading and releases resources.
func (s *Stream) Close() error {
	s.once.Do(func() {
		s.cancel()
		for range s.events {
		}
	})
	return nil
}

// Context returns the stream's context. Producers should select on
// ctx.Done() to detect cancellation from Close() or the parent context.
func (s *Stream) Context() context.Context {
	return s.ctx
}

// AgentEvent is the interface implemented by all agent events.
type AgentEvent interface {
	agentEvent()
	Type() string
}

// AgentStart is emitted when the agent begins processing.
type AgentStart struct{}

func (e AgentStart) agentEvent()  {}
func (e AgentStart) Type() string { return "agent_start" }

// AgentEnd is emitted when the agent completes processing.
type AgentEnd struct {
	Messages []Message
}

func (e AgentEnd) agentEvent()  {}
func (e AgentEnd) Type() string { return "agent_end" }

// AgentError is emitted when the agent encounters an error.
// When received via Recv(), it is returned as the error value (not as an event).
type AgentError struct {
	Error error
}

func (e AgentError) agentEvent()  {}
func (e AgentError) Type() string { return "agent_error" }

// TurnStart is emitted at the beginning of each turn.
type TurnStart struct{}

func (e TurnStart) agentEvent()  {}
func (e TurnStart) Type() string { return "turn_start" }

// TurnEnd is emitted at the end of each turn.
type TurnEnd struct {
	Message     Message
	ToolResults []ToolMessage
}

func (e TurnEnd) agentEvent()  {}
func (e TurnEnd) Type() string { return "turn_end" }

// MessageStart is emitted when a new message begins.
type MessageStart struct {
	Message Message
}

func (e MessageStart) agentEvent()  {}
func (e MessageStart) Type() string { return "message_start" }

// MessageUpdate is emitted during streaming to provide partial content.
type MessageUpdate struct {
	Message Message
	Event   gopiai.Event
}

func (e MessageUpdate) agentEvent()  {}
func (e MessageUpdate) Type() string { return "message_update" }

// MessageEnd is emitted when a message is complete.
type MessageEnd struct {
	Message Message
}

func (e MessageEnd) agentEvent()  {}
func (e MessageEnd) Type() string { return "message_end" }

// ToolExecutionStart is emitted when a tool begins execution.
type ToolExecutionStart struct {
	ToolCallID string
	ToolName   string
	Args       map[string]any
}

func (e ToolExecutionStart) agentEvent()  {}
func (e ToolExecutionStart) Type() string { return "tool_execution_start" }

// ToolExecutionEnd is emitted when a tool completes execution.
type ToolExecutionEnd struct {
	ToolCallID string
	ToolName   string
	Result     any
	IsError    bool
}

func (e ToolExecutionEnd) agentEvent()  {}
func (e ToolExecutionEnd) Type() string { return "tool_execution_end" }
