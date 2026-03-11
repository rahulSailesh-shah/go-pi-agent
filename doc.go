// Package agent provides a framework for building AI agents with tool execution,
// streaming responses, and event-driven architecture.
//
// This package is designed as a core building block for creating specialized
// AI agents. It provides both high-level and low-level APIs.
//
// # Overview
//
// The package offers two main APIs:
//
//   - High-level API: The [Agent] struct provides state management, event subscription,
//     and lifecycle control. Recommended for most use cases.
//
//   - Low-level API: The [AgentLoop] function provides direct control over the
//     execution loop for advanced use cases.
//
// # Quick Start
//
//	import (
//	    agent "github.com/rahulSailesh-shah/go-pi-agent"
//	    "github.com/rahulSailesh-shah/go-pi-ai/openai"
//	)
//
//	// Create a provider
//	provider, _ := openai.NewProvider(openai.Config{
//	    APIKey: os.Getenv("OPENAI_API_KEY"),
//	})
//
//	// Define tools
//	tools := []agent.AgentTool{
//	    {
//	        Tool: agent.Tool{
//	            Name:        "getWeather",
//	            Description: "Get the weather for a location",
//	            Parameters: map[string]any{
//	                "type": "object",
//	                "properties": map[string]any{
//	                    "location": map[string]string{"type": "string"},
//	                },
//	                "required": []string{"location"},
//	            },
//	        },
//	        Execute: func(toolCallID string, params map[string]any) (agent.ToolMessage, error) {
//	            return agent.ToolMessage{
//	                ToolCallID: toolCallID,
//	                ToolName:   "getWeather",
//	                Contents:   []agent.Content{agent.TextContent{Text: "Sunny, 72F"}},
//	            }, nil
//	        },
//	    },
//	}
//
//	// Create the agent
//	myAgent := agent.NewAgent(&agent.AgentOptions{
//	    InitialState: &agent.AgentState{
//	        SystemPrompt: "You are a helpful assistant.",
//	        Model:        provider,
//	        ModelName:    "gpt-4o",
//	        Tools:        tools,
//	    },
//	})
//
//	// Subscribe to events for streaming
//	unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
//	    switch ev := e.(type) {
//	    case agent.MessageUpdate:
//	        if delta, ok := ev.Event.(agent.EventTextDelta); ok {
//	            fmt.Print(delta.Delta)
//	        }
//	    }
//	})
//	defer unsubscribe()
//
//	// Send a prompt and wait
//	myAgent.Prompt(context.Background(), "What's the weather in Tokyo?")
//	<-myAgent.WaitForIdle()
//
// # Core Concepts
//
// ## Agent
//
// The [Agent] struct manages conversation state, handles events, and provides
// methods for sending prompts, steering, and follow-up messages.
//
// ## Tools
//
// Tools are defined using [AgentTool], which combines a schema (name, description,
// parameters) with an Execute function.
//
// ## Events
//
// The agent emits events during execution through the [AgentEvent] interface:
//   - [AgentStart] / [AgentEnd]: Agent lifecycle events
//   - [TurnStart] / [TurnEnd]: Turn lifecycle events
//   - [MessageStart] / [MessageUpdate] / [MessageEnd]: Message streaming events
//   - [ToolExecutionStart] / [ToolExecutionEnd]: Tool execution events
//
// ## Steering and Follow-up
//
//   - Steering: Interrupt the agent mid-execution (e.g., during tool calls)
//   - Follow-up: Queue messages to be processed after the current execution
//
// # Thread Safety
//
// The [Agent] struct is thread-safe. All public methods can be called
// concurrently from multiple goroutines.
//
// # Dependencies
//
// This package depends on [github.com/rahulSailesh-shah/go-pi-ai] for the
// Provider interface and core types (Message, Tool, Content, etc.).
//
// See the examples directory for complete working examples.
package agent
