package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	gopiai "github.com/rahulSailesh-shah/go-pi-ai"
)

// sendEvent sends an event to the channel, returning false if the context is cancelled.
func sendEvent(ctx context.Context, events chan<- AgentEvent, event AgentEvent) bool {
	select {
	case events <- event:
		return true
	case <-ctx.Done():
		return false
	}
}

// AgentLoop starts an agent loop with new prompt messages.
// This is the low-level API. For most use cases, prefer the Agent struct.
// The returned Stream must be consumed with Recv() and closed with Close().
func AgentLoop(ctx context.Context, prompts []Message, agentContext AgentContext,
	config AgentLoopConfig) *Stream {

	stream, events := NewStream(ctx)

	go func() {
		defer close(events)

		sctx := stream.Context()

		newMessages := make([]Message, len(prompts))
		copy(newMessages, prompts)

		currentContext := AgentContext{
			SystemPrompt: agentContext.SystemPrompt,
			Messages:     append(agentContext.Messages, prompts...),
			Tools:        agentContext.Tools,
		}

		if !sendEvent(sctx, events, AgentStart{}) {
			return
		}
		if !sendEvent(sctx, events, TurnStart{}) {
			return
		}

		for _, prompt := range prompts {
			if !sendEvent(sctx, events, MessageStart{Message: prompt}) {
				return
			}
			if !sendEvent(sctx, events, MessageEnd{Message: prompt}) {
				return
			}
		}

		if err := runLoop(sctx, &currentContext, &newMessages, config, events); err != nil {
			sendEvent(sctx, events, AgentError{Error: err})
		}
	}()

	return stream
}

// AgentLoopContinue continues an agent loop from the current context without adding new messages.
// Useful for retries when the context already has user messages or tool results as the last message.
func AgentLoopContinue(
	ctx context.Context,
	agentContext AgentContext,
	config AgentLoopConfig,
) (*Stream, error) {

	if len(agentContext.Messages) == 0 {
		return nil, errors.New("cannot continue: no messages in context")
	}

	lastMsg := agentContext.Messages[len(agentContext.Messages)-1]
	if lastMsg.Role() == "assistant" {
		return nil, errors.New("cannot continue from message role: assistant")
	}

	stream, events := NewStream(ctx)

	go func() {
		defer close(events)

		sctx := stream.Context()

		newMessages := []Message{}
		currentContext := AgentContext{
			SystemPrompt: agentContext.SystemPrompt,
			Messages:     agentContext.Messages,
			Tools:        agentContext.Tools,
		}

		if !sendEvent(sctx, events, AgentStart{}) {
			return
		}
		if !sendEvent(sctx, events, TurnStart{}) {
			return
		}

		if err := runLoop(sctx, &currentContext, &newMessages, config, events); err != nil {
			sendEvent(sctx, events, AgentError{Error: err})
		}
	}()

	return stream, nil
}

// runLoop is the core agent execution loop that handles turns, tool calls, and steering.
func runLoop(
	ctx context.Context,
	currentContext *AgentContext,
	newMessages *[]Message,
	config AgentLoopConfig,
	events chan<- AgentEvent,
) error {
	firstTurn := true
	var pendingMessages []Message

	if config.GetSteeringMessages != nil {
		var err error
		pendingMessages, err = config.GetSteeringMessages()
		if err != nil {
			return err
		}
	}

	for {
		hasMoreToolCalls := true
		var steeringAfterTools []Message

		for hasMoreToolCalls || len(pendingMessages) > 0 {
			if !firstTurn {
				if !sendEvent(ctx, events, TurnStart{}) {
					return ctx.Err()
				}
			} else {
				firstTurn = false
			}

			if len(pendingMessages) > 0 {
				for _, message := range pendingMessages {
					if !sendEvent(ctx, events, MessageStart{Message: message}) {
						return ctx.Err()
					}
					if !sendEvent(ctx, events, MessageEnd{Message: message}) {
						return ctx.Err()
					}
					currentContext.Messages = append(currentContext.Messages, message)
					*newMessages = append(*newMessages, message)
				}
				pendingMessages = nil
			}

			message, err := streamAssistantResponse(ctx, currentContext, config, events)
			if err != nil {
				return err
			}

			*newMessages = append(*newMessages, message)
			currentContext.Messages = append(currentContext.Messages, message)

			var toolCalls []gopiai.ToolCall
			for _, c := range message.GetContents() {
				if tc, ok := c.(gopiai.ToolCall); ok {
					toolCalls = append(toolCalls, tc)
				}
			}

			hasMoreToolCalls = len(toolCalls) > 0
			var toolResults []ToolMessage

			if hasMoreToolCalls {
				executionResults, err := executeToolCalls(ctx, currentContext.Tools, toolCalls, events,
					config.GetSteeringMessages)
				if err != nil {
					return err
				}

				toolResults = append(toolResults, executionResults.ToolResults...)
				steeringAfterTools = executionResults.SteeringMessages

				for _, result := range toolResults {
					currentContext.Messages = append(currentContext.Messages, result)
					*newMessages = append(*newMessages, result)
				}
			}

			if !sendEvent(ctx, events, TurnEnd{Message: message, ToolResults: toolResults}) {
				return ctx.Err()
			}

			if len(steeringAfterTools) > 0 {
				pendingMessages = steeringAfterTools
				steeringAfterTools = nil
			} else if config.GetSteeringMessages != nil {
				var err error
				pendingMessages, err = config.GetSteeringMessages()
				if err != nil {
					return err
				}
			}
		}

		if config.GetFollowUpMessages != nil {
			followUpMessages, err := config.GetFollowUpMessages()
			if err != nil {
				return err
			}
			if len(followUpMessages) > 0 {
				pendingMessages = followUpMessages
				continue
			}
		}

		break
	}

	if !sendEvent(ctx, events, AgentEnd{Messages: *newMessages}) {
		return ctx.Err()
	}

	return nil
}

// streamAssistantResponse streams the LLM response and emits events.
func streamAssistantResponse(
	ctx context.Context,
	currentContext *AgentContext,
	config AgentLoopConfig,
	events chan<- AgentEvent,
) (Message, error) {
	var tools []gopiai.Tool
	for _, t := range currentContext.Tools {
		tools = append(tools, t.Tool)
	}

	req := gopiai.Request{
		Model:        config.ModelName,
		SystemPrompt: currentContext.SystemPrompt,
		Messages:     currentContext.Messages,
		Tools:        tools,
	}

	llmStream, err := config.Model.Stream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start stream: %w", err)
	}
	defer llmStream.Close()

	messageStarted := false
	var finalMessage gopiai.AssistantMessage

	for {
		event, err := llmStream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		var partial Message
		switch e := event.(type) {
		case gopiai.EventStart:
			continue
		case gopiai.EventTextStart:
			partial = e.Partial
		case gopiai.EventTextDelta:
			partial = e.Partial
		case gopiai.EventTextEnd:
			partial = e.Partial
		case gopiai.EventToolcallStart:
			partial = e.Partial
		case gopiai.EventToolcallDelta:
			partial = e.Partial
		case gopiai.EventToolcallEnd:
			partial = e.Partial
		case gopiai.EventDone:
			finalMessage = e.Message
			partial = e.Message
		default:
			continue
		}

		if !messageStarted {
			if !sendEvent(ctx, events, MessageStart{Message: partial}) {
				return nil, ctx.Err()
			}
			messageStarted = true
		}

		if !sendEvent(ctx, events, MessageUpdate{Event: event, Message: partial}) {
			return nil, ctx.Err()
		}
	}

	if finalMessage.GetContents() == nil && !messageStarted {
		if !sendEvent(ctx, events, MessageStart{Message: finalMessage}) {
			return nil, ctx.Err()
		}
	}

	if !sendEvent(ctx, events, MessageEnd{Message: finalMessage}) {
		return nil, ctx.Err()
	}

	return finalMessage, nil
}

type executionResult struct {
	ToolResults      []ToolMessage
	SteeringMessages []Message
}

func executeToolCalls(
	ctx context.Context,
	tools []AgentTool,
	toolCalls []gopiai.ToolCall,
	events chan<- AgentEvent,
	getSteeringMessages func() ([]Message, error),
) (executionResult, error) {

	var results []ToolMessage
	var steeringMessages []Message

	for i, toolCall := range toolCalls {
		select {
		case <-ctx.Done():
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, ctx.Err()
		default:
		}

		var tool *AgentTool
		for _, t := range tools {
			if t.Name == toolCall.Name {
				val := t
				tool = &val
				break
			}
		}

		if !sendEvent(ctx, events, ToolExecutionStart{
			ToolCallID: toolCall.ID,
			ToolName:   toolCall.Name,
			Args:       toolCall.Arguments,
		}) {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, ctx.Err()
		}

		var result ToolMessage
		var isError bool

		if tool == nil {
			result = createErrorToolResult(toolCall.ID, toolCall.Name, fmt.Sprintf("Tool %s not found", toolCall.Name))
			isError = true
		} else {
			res, err := tool.Execute(toolCall.ID, toolCall.Arguments)
			if err != nil {
				result = createErrorToolResult(toolCall.ID, toolCall.Name, err.Error())
				isError = true
			} else {
				result = res
			}
		}

		if !sendEvent(ctx, events, ToolExecutionEnd{
			ToolCallID: toolCall.ID,
			ToolName:   toolCall.Name,
			Result:     result,
			IsError:    isError,
		}) {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, ctx.Err()
		}

		results = append(results, result)

		if !sendEvent(ctx, events, MessageStart{Message: result}) {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, ctx.Err()
		}
		if !sendEvent(ctx, events, MessageEnd{Message: result}) {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, ctx.Err()
		}

		if getSteeringMessages != nil {
			steering, err := getSteeringMessages()
			if err == nil && len(steering) > 0 {
				steeringMessages = steering
				remaining := toolCalls[i+1:]
				for _, skipped := range remaining {
					if err := skipToolCall(ctx, skipped, events); err != nil {
						return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, err
					}
					results = append(results, createErrorToolResult(skipped.ID, skipped.Name, "Skipped due to queued user message."))
				}
				break
			}
		}
	}

	return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, nil
}

func skipToolCall(ctx context.Context, toolCall gopiai.ToolCall, events chan<- AgentEvent) error {
	result := createErrorToolResult(toolCall.ID, toolCall.Name, "Skipped due to queued user message.")

	if !sendEvent(ctx, events, ToolExecutionStart{
		ToolCallID: toolCall.ID,
		ToolName:   toolCall.Name,
		Args:       toolCall.Arguments,
	}) {
		return ctx.Err()
	}

	if !sendEvent(ctx, events, ToolExecutionEnd{
		ToolCallID: toolCall.ID,
		ToolName:   toolCall.Name,
		Result:     result,
		IsError:    true,
	}) {
		return ctx.Err()
	}

	if !sendEvent(ctx, events, MessageStart{Message: result}) {
		return ctx.Err()
	}
	if !sendEvent(ctx, events, MessageEnd{Message: result}) {
		return ctx.Err()
	}

	return nil
}

func createErrorToolResult(id, name, text string) ToolMessage {
	return gopiai.ToolMessage{
		ToolCallID: id,
		ToolName:   name,
		Timestamp:  time.Now(),
		Contents: []gopiai.Content{
			gopiai.TextContent{Text: text},
		},
		IsError: true,
	}
}
