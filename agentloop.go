package agent

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/rahulSailesh-shah/go-pi-ai/types"
)

// sendEvent safely sends an event to the channel, respecting context cancellation.
// Returns ctx.Err() if context is cancelled before send completes.
func sendEvent(ctx context.Context, ch chan<- AgentEvent, event AgentEvent) error {
	select {
	case ch <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// AgentLoop starts an agent loop with a new prompt message.
// This is the low-level API for running an agent. For most use cases,
// prefer using the Agent struct's Prompt method instead.
//
// The prompts are added to the context and events are emitted for them.
// The returned AgentEventStream provides channels for events, results, and errors.
//
// Example:
//
//	stream := AgentLoop(ctx, prompts, agentContext, config)
//	for event := range stream.Events {
//	    // Handle events
//	}
//	result := <-stream.Result
func AgentLoop(ctx context.Context,
	prompts []AgentMessage,
	agentContext AgentContext,
	config AgentLoopConfig) AgentEventStream {

	stream := newAgentEventStream()

	go func() {
		defer close(stream.Events)
		defer close(stream.Result)
		defer close(stream.Err)

		newMessages := make([]AgentMessage, len(prompts))
		copy(newMessages, prompts)

		currentContext := AgentContext{
			SystemPrompt: agentContext.SystemPrompt,
			Messages:     append(agentContext.Messages, prompts...),
			Tools:        agentContext.Tools,
		}

		if err := sendEvent(ctx, stream.Events, AgentStart{}); err != nil {
			stream.Err <- err
			return
		}
		if err := sendEvent(ctx, stream.Events, TurnStart{}); err != nil {
			stream.Err <- err
			return
		}

		for _, prompt := range prompts {
			if err := sendEvent(ctx, stream.Events, MessageStart{Message: prompt}); err != nil {
				stream.Err <- err
				return
			}
			if err := sendEvent(ctx, stream.Events, MessageEnd{Message: prompt}); err != nil {
				stream.Err <- err
				return
			}
		}

		err := runLoop(ctx, &currentContext, &newMessages, config, stream)
		if err != nil {
			stream.Err <- err
		}
	}()

	return stream
}

// AgentLoopContinue continues an agent loop from the current context without adding a new message.
// This is useful for retries when the context already has user message or tool results as the last message.
//
// Returns an error if there are no messages in the context or if the last message is from the assistant.
func AgentLoopContinue(
	ctx context.Context,
	agentContext AgentContext,
	config AgentLoopConfig,
) (AgentEventStream, error) {

	if len(agentContext.Messages) == 0 {
		return AgentEventStream{}, errors.New("cannot continue: no messages in context")
	}

	lastMsg := agentContext.Messages[len(agentContext.Messages)-1]
	if lastMsg.Role() == "assistant" {
		return AgentEventStream{}, errors.New("cannot continue from message role: assistant")
	}

	stream := newAgentEventStream()

	go func() {
		defer close(stream.Events)
		defer close(stream.Result)
		defer close(stream.Err)

		newMessages := []AgentMessage{}
		// Copy context to avoid mutating the original passed context
		currentContext := AgentContext{
			SystemPrompt: agentContext.SystemPrompt,
			Messages:     agentContext.Messages,
			Tools:        agentContext.Tools,
		}

		if err := sendEvent(ctx, stream.Events, AgentStart{}); err != nil {
			stream.Err <- err
			return
		}
		if err := sendEvent(ctx, stream.Events, TurnStart{}); err != nil {
			stream.Err <- err
			return
		}

		err := runLoop(ctx, &currentContext, &newMessages, config, stream)
		if err != nil {
			stream.Err <- err
		}
	}()

	return stream, nil
}

// newAgentEventStream creates a new AgentEventStream with properly initialized channels.
func newAgentEventStream() AgentEventStream {
	return AgentEventStream{
		Events: make(chan AgentEvent),
		Result: make(chan []AgentMessage, 1), // Buffered to prevent deadlock
		Err:    make(chan error, 1),          // Buffered to prevent deadlock
	}
}

// runLoop is the core agent execution loop that handles turns, tool calls, and steering.
func runLoop(
	ctx context.Context,
	currentContext *AgentContext,
	newMessages *[]AgentMessage,
	config AgentLoopConfig,
	stream AgentEventStream,
) error {
	firstTurn := true
	var pendingMessages []AgentMessage

	if config.GetSteeringMessages != nil {
		var err error
		pendingMessages, err = config.GetSteeringMessages()
		if err != nil {
			return err
		}
	}

	for {
		hasMoreToolCalls := true
		var steeringAfterTools []AgentMessage

		for hasMoreToolCalls || len(pendingMessages) > 0 {
			if !firstTurn {
				if err := sendEvent(ctx, stream.Events, TurnStart{}); err != nil {
					return err
				}
			} else {
				firstTurn = false
			}

			if len(pendingMessages) > 0 {
				for _, message := range pendingMessages {
					if err := sendEvent(ctx, stream.Events, MessageStart{Message: message}); err != nil {
						return err
					}
					if err := sendEvent(ctx, stream.Events, MessageEnd{Message: message}); err != nil {
						return err
					}
					currentContext.Messages = append(currentContext.Messages, message)
					*newMessages = append(*newMessages, message)
				}
				pendingMessages = nil
			}

			message, err := streamAssistantResponse(ctx, currentContext, config, stream)
			if err != nil {
				return err
			}

			*newMessages = append(*newMessages, message)
			currentContext.Messages = append(currentContext.Messages, message)

			var toolCalls []types.ToolCall

			for _, c := range getContents(message) {
				switch t := c.(type) {
				case types.ToolCall:
					toolCalls = append(toolCalls, t)
				}
			}

			hasMoreToolCalls = len(toolCalls) > 0
			var toolResults []AgentToolResult

			if hasMoreToolCalls {
				executionResults, err := executeToolCalls(ctx, currentContext.Tools, toolCalls, stream, config.GetSteeringMessages)
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

			if err := sendEvent(ctx, stream.Events, TurnEnd{Message: message, ToolResults: toolResults}); err != nil {
				return err
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

	if err := sendEvent(ctx, stream.Events, AgentEnd{Messages: *newMessages}); err != nil {
		return err
	}

	// Send result - use select to handle cancellation
	select {
	case stream.Result <- *newMessages:
	case <-ctx.Done():
		return ctx.Err()
	}

	return nil
}

// getContents extracts the content slice from a message.
func getContents(msg AgentMessage) []types.Content {
	switch m := msg.(type) {
	case types.UserMessage:
		return m.Contents
	case types.AssistantMessage:
		return m.Contents
	case types.ToolMessage:
		return m.Contents
	default:
		return nil
	}
}

// streamAssistantResponse streams the LLM response and emits appropriate events.
func streamAssistantResponse(
	ctx context.Context,
	currentContext *AgentContext,
	config AgentLoopConfig,
	stream AgentEventStream,
) (AgentMessage, error) {
	var tools []types.Tool
	for _, t := range currentContext.Tools {
		tools = append(tools, t.Tool)
	}

	llmContext := types.Context{
		SystemPrompt: currentContext.SystemPrompt,
		Messages:     currentContext.Messages,
		Tools:        tools,
	}

	responseStream := config.Model.Stream(ctx, llmContext)

	messageStarted := false

Loop:
	for {
		select {
		case event, ok := <-responseStream.Events:
			if !ok {
				break Loop
			}

			switch e := event.(type) {
			case types.AssistantMessageEvent:
				var partial AgentMessage
				switch inner := e.(type) {
				case types.EventTextStart:
					partial = inner.Partial
				case types.EventTextDelta:
					partial = inner.Partial
				case types.EventTextEnd:
					partial = inner.Partial
				case types.EventToolcallStart:
					partial = inner.Partial
				case types.EventToolcallDelta:
					partial = inner.Partial
				case types.EventToolcallEnd:
					partial = inner.Partial
				case types.EventDone:
					partial = inner.Message
				case types.EventError:
					partial = inner.Error
				}

				// Send MessageStart on first AssistantMessageEvent
				if !messageStarted {
					if err := sendEvent(ctx, stream.Events, MessageStart{Message: partial}); err != nil {
						return nil, err
					}
					messageStarted = true
				}

				// Send MessageUpdate for all events after MessageStart
				if err := sendEvent(ctx, stream.Events, MessageUpdate{
					AssistantMessageEvent: e,
					Message:               partial,
				}); err != nil {
					return nil, err
				}

				switch e.(type) {
				case types.EventDone, types.EventError:
					break Loop
				}
			default:
				if ae, ok := event.(AgentEvent); ok {
					if err := sendEvent(ctx, stream.Events, ae); err != nil {
						return nil, err
					}
				}
			}

		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Wait for final result with cancellation support
	var finalMessage AgentMessage
	select {
	case finalMessage = <-responseStream.Result:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Check for errors
	select {
	case err := <-responseStream.Err:
		if err != nil {
			return nil, err
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Send MessageEnd with the final complete message
	// If we never started (no events), send both start and end
	if !messageStarted {
		if err := sendEvent(ctx, stream.Events, MessageStart{Message: finalMessage}); err != nil {
			return nil, err
		}
	}
	if err := sendEvent(ctx, stream.Events, MessageEnd{Message: finalMessage}); err != nil {
		return nil, err
	}

	return finalMessage, nil
}

// executionResult holds the results of tool execution along with any steering messages.
type executionResult struct {
	ToolResults      []AgentToolResult
	SteeringMessages []AgentMessage
}

// executeToolCalls executes a list of tool calls and returns the results.
func executeToolCalls(
	ctx context.Context,
	tools []AgentTool,
	toolCalls []types.ToolCall,
	stream AgentEventStream,
	getSteeringMessages func() ([]AgentMessage, error),
) (executionResult, error) {

	var results []AgentToolResult
	var steeringMessages []AgentMessage

	for i, toolCall := range toolCalls {
		// Check for cancellation before each tool
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

		if err := sendEvent(ctx, stream.Events, ToolExecutionStart{
			ToolCallId: toolCall.ID,
			ToolName:   toolCall.Name,
			Args:       toolCall.Arguments,
		}); err != nil {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, err
		}

		var result AgentToolResult
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

		if err := sendEvent(ctx, stream.Events, ToolExecutionEnd{
			ToolCallId: toolCall.ID,
			ToolName:   toolCall.Name,
			Result:     result,
			IsError:    isError,
		}); err != nil {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, err
		}

		results = append(results, result)

		if err := sendEvent(ctx, stream.Events, MessageStart{Message: result}); err != nil {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, err
		}
		if err := sendEvent(ctx, stream.Events, MessageEnd{Message: result}); err != nil {
			return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, err
		}

		if getSteeringMessages != nil {
			steering, err := getSteeringMessages()
			if err == nil && len(steering) > 0 {
				steeringMessages = steering
				remaining := toolCalls[i+1:]
				for _, skipped := range remaining {
					skippedResult, err := skipToolCall(ctx, skipped, stream)
					if err != nil {
						return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, err
					}
					results = append(results, skippedResult)
				}
				break
			}
		}
	}

	return executionResult{ToolResults: results, SteeringMessages: steeringMessages}, nil
}

// skipToolCall creates an error result for a skipped tool call.
func skipToolCall(ctx context.Context, toolCall types.ToolCall, stream AgentEventStream) (AgentToolResult, error) {
	result := createErrorToolResult(toolCall.ID, toolCall.Name, "Skipped due to queued user message.")

	if err := sendEvent(ctx, stream.Events, ToolExecutionStart{
		ToolCallId: toolCall.ID,
		ToolName:   toolCall.Name,
		Args:       toolCall.Arguments,
	}); err != nil {
		return result, err
	}

	if err := sendEvent(ctx, stream.Events, ToolExecutionEnd{
		ToolCallId: toolCall.ID,
		ToolName:   toolCall.Name,
		Result:     result,
		IsError:    true,
	}); err != nil {
		return result, err
	}

	if err := sendEvent(ctx, stream.Events, MessageStart{Message: result}); err != nil {
		return result, err
	}
	if err := sendEvent(ctx, stream.Events, MessageEnd{Message: result}); err != nil {
		return result, err
	}

	return result, nil
}

// createErrorToolResult creates a tool result indicating an error.
func createErrorToolResult(id, name, text string) AgentToolResult {
	return types.ToolMessage{
		ToolCallId: id,
		ToolName:   name,
		Timestamp:  time.Now(),
		Contents: []types.Content{
			types.TextContent{Text: text},
		},
		IsError: true,
	}
}
