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

		runAgentLoop(sctx, &currentContext, &newMessages, config, events)
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

	if lastMsg := agentContext.Messages[len(agentContext.Messages)-1]; lastMsg.Role() == "assistant" {
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

		runAgentLoop(sctx, &currentContext, &newMessages, config, events)
	}()

	return stream, nil
}

// runAgentLoop drives turns to completion and emits the terminal AgentError /
// AgentEnd events. Callers are responsible for AgentStart and the initial
// TurnStart.
func runAgentLoop(
	ctx context.Context,
	currentContext *AgentContext,
	newMessages *[]Message,
	config AgentLoopConfig,
	events chan<- AgentEvent,
) {
	if err := runTurns(ctx, currentContext, newMessages, config, events); err != nil {
		sendEvent(ctx, events, AgentError{Error: err})
	}
	sendEvent(ctx, events, AgentEnd{Messages: *newMessages})
}

// runTurns executes assistant generations until no tool calls remain or an
// error occurs. Each iteration emits exactly one TurnEnd; if the assistant
// requested tools, a TurnStart is emitted to open the next iteration.
func runTurns(
	ctx context.Context,
	currentContext *AgentContext,
	newMessages *[]Message,
	config AgentLoopConfig,
	events chan<- AgentEvent,
) error {
	for {
		message, err := streamAssistantResponse(ctx, currentContext, config, events)
		if message != nil {
			*newMessages = append(*newMessages, message)
			currentContext.Messages = append(currentContext.Messages, message)
		}
		if err != nil {
			sendEvent(ctx, events, TurnEnd{Message: message})
			return err
		}

		var toolCalls []gopiai.ToolCall
		for _, c := range message.GetContents() {
			if tc, ok := c.(gopiai.ToolCall); ok {
				toolCalls = append(toolCalls, tc)
			}
		}

		if len(toolCalls) == 0 {
			if !sendEvent(ctx, events, TurnEnd{Message: message}) {
				return ctx.Err()
			}
			return nil
		}

		toolResults, execErr := executeToolCalls(ctx, currentContext.Tools, toolCalls, events)
		for _, r := range toolResults {
			currentContext.Messages = append(currentContext.Messages, r)
			*newMessages = append(*newMessages, r)
		}
		if !sendEvent(ctx, events, TurnEnd{Message: message, ToolResults: toolResults}) {
			return ctx.Err()
		}
		if execErr != nil {
			return execErr
		}

		if !sendEvent(ctx, events, TurnStart{}) {
			return ctx.Err()
		}
	}
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

	llmStream, err := config.Provider.Stream(ctx, req)
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
			if e.Err != nil {
				return finalMessage, e.Err
			}
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

// executeToolCalls runs each tool call and emits ToolExecutionStart/End. The
// returned slice contains every result produced before the first error (if any).
func executeToolCalls(
	ctx context.Context,
	tools []AgentTool,
	toolCalls []gopiai.ToolCall,
	events chan<- AgentEvent,
) ([]ToolMessage, error) {

	var results []ToolMessage

	for _, toolCall := range toolCalls {
		select {
		case <-ctx.Done():
			return results, ctx.Err()
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
			return results, ctx.Err()
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

		results = append(results, result)

		if !sendEvent(ctx, events, ToolExecutionEnd{
			ToolCallID: toolCall.ID,
			ToolName:   toolCall.Name,
			Result:     result,
			IsError:    isError,
		}) {
			return results, ctx.Err()
		}
	}

	return results, nil
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
