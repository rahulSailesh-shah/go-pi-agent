package agent

import (
	"github.com/rahulSailesh-shah/go-pi-ai/provider"
	"github.com/rahulSailesh-shah/go-pi-ai/types"
)

// AgentMessage is an alias for types.Message, representing any message in the conversation.
// Messages can be from the user, assistant, or tool.
type AgentMessage = types.Message

// AgentToolResult is an alias for types.ToolMessage, representing the result of a tool execution.
type AgentToolResult = types.ToolMessage

// AgentTool defines a tool that can be executed by the agent.
// It combines the tool schema (Name, Description, Parameters) with
// an Execute function that performs the actual work.
//
// Example:
//
//	tool := AgentTool{
//	    Tool: types.Tool{
//	        Name:        "getWeather",
//	        Description: "Get the weather for a location",
//	        Parameters: map[string]any{
//	            "type": "object",
//	            "properties": map[string]any{
//	                "location": map[string]string{"type": "string"},
//	            },
//	            "required": []string{"location"},
//	        },
//	    },
//	    Label: "Weather",
//	    Execute: func(toolCallId string, params map[string]any) (AgentToolResult, error) {
//	        location := params["location"].(string)
//	        // Fetch weather...
//	        return types.ToolMessage{
//	            ToolCallId: toolCallId,
//	            ToolName:   "getWeather",
//	            Contents:   []types.Content{types.TextContent{Text: "Sunny, 72F"}},
//	        }, nil
//	    },
//	}
type AgentTool struct {
	types.Tool

	// Label is a human-readable label for the tool (optional).
	// This can be used for UI display purposes.
	Label string

	// Execute is called when the LLM requests this tool.
	// It receives the tool call ID (for tracking) and the parsed arguments.
	// Returns the tool result or an error.
	Execute func(toolCallId string, params map[string]any) (AgentToolResult, error)
}

// AgentContext holds the context for an agent loop execution.
// It contains the system prompt, conversation history, and available tools.
type AgentContext struct {
	// SystemPrompt is the initial instruction given to the LLM.
	SystemPrompt string

	// Messages is the conversation history.
	Messages []AgentMessage

	// Tools is the list of tools available for execution.
	Tools []AgentTool
}

// AgentLoopConfig configures the behavior of the agent loop.
type AgentLoopConfig struct {
	// Model is the LLM provider to use for generating responses.
	Model provider.Provider

	// SessionId is an optional identifier for the conversation session.
	SessionId string

	// GetSteeringMessages is called during execution to check for steering messages.
	// Steering messages can interrupt tool execution to redirect the agent.
	// Return nil or empty slice if no steering messages are available.
	GetSteeringMessages func() ([]AgentMessage, error)

	// GetFollowUpMessages is called after the agent completes to check for follow-up messages.
	// Follow-up messages are processed in a new turn after the current execution completes.
	// Return nil or empty slice if no follow-up messages are available.
	GetFollowUpMessages func() ([]AgentMessage, error)
}

// AgentEventStream provides channels for receiving agent events, results, and errors.
// Events are streamed in real-time as the agent processes.
type AgentEventStream struct {
	// Events channel receives all agent events during execution.
	// The channel is closed when the agent completes or encounters an error.
	Events chan AgentEvent

	// Result channel receives the final list of new messages when the agent completes successfully.
	// This is a buffered channel with capacity 1.
	Result chan []AgentMessage

	// Err channel receives any error that occurs during execution.
	// This is a buffered channel with capacity 1.
	Err chan error
}

// AgentEvent is the interface implemented by all agent events.
// Events are emitted during agent execution to track progress and state changes.
type AgentEvent interface {
	isAssistantMessageEvent()

	// Type returns the event type as a string (e.g., "agent_start", "message_update").
	Type() string
}

// AgentStart is emitted when the agent begins processing.
type AgentStart struct{}

func (e AgentStart) isAssistantMessageEvent() {}

// Type returns "agent_start".
func (e AgentStart) Type() string {
	return "agent_start"
}

// AgentEnd is emitted when the agent completes processing.
// It contains all new messages generated during this execution.
type AgentEnd struct {
	// Messages contains all new messages generated during this agent run.
	Messages []AgentMessage
}

func (e AgentEnd) isAssistantMessageEvent() {}

// Type returns "agent_end".
func (e AgentEnd) Type() string {
	return "agent_end"
}

// TurnStart is emitted at the beginning of each turn.
// A turn consists of an LLM response and any subsequent tool executions.
type TurnStart struct{}

func (e TurnStart) isAssistantMessageEvent() {}

// Type returns "turn_start".
func (e TurnStart) Type() string {
	return "turn_start"
}

// TurnEnd is emitted at the end of each turn.
// It contains the assistant message and any tool results from this turn.
type TurnEnd struct {
	// Message is the assistant's response for this turn.
	Message AgentMessage

	// ToolResults contains the results of any tools executed during this turn.
	ToolResults []AgentToolResult
}

func (e TurnEnd) isAssistantMessageEvent() {}

// Type returns "turn_end".
func (e TurnEnd) Type() string {
	return "turn_end"
}

// MessageStart is emitted when a new message begins.
// This is emitted for user messages, assistant messages, and tool results.
type MessageStart struct {
	// Message is the message that is starting.
	Message AgentMessage
}

func (e MessageStart) isAssistantMessageEvent() {}

// Type returns "message_start".
func (e MessageStart) Type() string {
	return "message_start"
}

// MessageUpdate is emitted during streaming to provide partial message content.
// This event contains both the partial message and the underlying LLM event.
type MessageUpdate struct {
	// Message is the partial message with content received so far.
	Message AgentMessage

	// AssistantMessageEvent is the underlying event from the LLM provider.
	// This can be used to access detailed streaming information like text deltas.
	AssistantMessageEvent types.AssistantMessageEvent
}

func (e MessageUpdate) isAssistantMessageEvent() {}

// Type returns "message_update".
func (e MessageUpdate) Type() string {
	return "message_update"
}

// MessageEnd is emitted when a message is complete.
type MessageEnd struct {
	// Message is the complete message.
	Message AgentMessage
}

func (e MessageEnd) isAssistantMessageEvent() {}

// Type returns "message_end".
func (e MessageEnd) Type() string {
	return "message_end"
}

// ToolExecutionStart is emitted when a tool begins execution.
type ToolExecutionStart struct {
	// ToolCallId is the unique identifier for this tool call.
	ToolCallId string

	// ToolName is the name of the tool being executed.
	ToolName string

	// Args contains the arguments passed to the tool.
	Args map[string]any
}

func (e ToolExecutionStart) isAssistantMessageEvent() {}

// Type returns "tool_execution_start".
func (e ToolExecutionStart) Type() string {
	return "tool_execution_start"
}

// ToolExecutionUpdate is emitted during tool execution to provide progress updates.
// This is optional and depends on the tool implementation.
type ToolExecutionUpdate struct {
	// ToolCallId is the unique identifier for this tool call.
	ToolCallId string

	// ToolName is the name of the tool being executed.
	ToolName string

	// Args contains the arguments passed to the tool.
	Args map[string]any

	// PartialResult contains any partial result from the tool.
	PartialResult any
}

func (e ToolExecutionUpdate) isAssistantMessageEvent() {}

// Type returns "tool_execution_update".
func (e ToolExecutionUpdate) Type() string {
	return "tool_execution_update"
}

// ToolExecutionEnd is emitted when a tool completes execution.
type ToolExecutionEnd struct {
	// ToolCallId is the unique identifier for this tool call.
	ToolCallId string

	// ToolName is the name of the tool that was executed.
	ToolName string

	// Result contains the tool execution result.
	Result any

	// IsError indicates whether the tool execution resulted in an error.
	IsError bool
}

func (e ToolExecutionEnd) isAssistantMessageEvent() {}

// Type returns "tool_execution_end".
func (e ToolExecutionEnd) Type() string {
	return "tool_execution_end"
}
