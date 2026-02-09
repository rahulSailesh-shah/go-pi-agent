// Package agent provides a framework for building AI agents with tool execution,
// streaming responses, and event-driven architecture.
//
// This package is designed to be used as a core building block for creating
// specialized AI agents in other projects. It provides both high-level and
// low-level APIs for different use cases.
//
// # Overview
//
// The package offers two main APIs:
//
//   - High-level API: The [Agent] struct provides state management, event subscription,
//     and lifecycle control. This is recommended for most use cases.
//
//   - Low-level API: The [AgentLoop] function provides direct control over the
//     execution loop for advanced use cases.
//
// # Quick Start
//
// Here's a minimal example using the high-level API:
//
//	// Create tools
//	tools := []agent.AgentTool{
//	    {
//	        Tool: types.Tool{
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
//	        Execute: func(toolCallId string, params map[string]any) (agent.AgentToolResult, error) {
//	            // Implement tool logic here
//	            return types.ToolMessage{
//	                ToolCallId: toolCallId,
//	                ToolName:   "getWeather",
//	                Contents:   []types.Content{types.TextContent{Text: "Sunny, 72F"}},
//	            }, nil
//	        },
//	    },
//	}
//
//	// Get a model provider
//	model, _ := provider.GetModel(types.ProviderOpenAI, "gpt-4")
//
//	// Create the agent
//	myAgent := agent.NewAgent(&agent.AgentOptions{
//	    InitialState: &agent.AgentState{
//	        SystemPrompt: "You are a helpful assistant.",
//	        Model:        model,
//	        Tools:        tools,
//	    },
//	})
//
//	// Subscribe to events for streaming
//	unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
//	    switch ev := e.(type) {
//	    case agent.MessageUpdate:
//	        if delta, ok := ev.AssistantMessageEvent.(types.EventTextDelta); ok {
//	            fmt.Print(delta.Delta)
//	        }
//	    }
//	})
//	defer unsubscribe()
//
//	// Send a prompt
//	myAgent.Prompt("What's the weather in Tokyo?")
//
//	// Wait for completion
//	<-myAgent.WaitForIdle()
//
// # Core Concepts
//
// ## Agent
//
// The [Agent] struct is the main interface for interacting with the agent loop.
// It manages conversation state, handles events, and provides methods for
// sending prompts, steering, and follow-up messages.
//
// ## Tools
//
// Tools are defined using [AgentTool], which combines a schema (name, description,
// parameters) with an Execute function. When the LLM decides to use a tool,
// the Execute function is called with the parsed arguments.
//
// ## Events
//
// The agent emits events during execution through the [AgentEvent] interface.
// Events include:
//   - [AgentStart] / [AgentEnd]: Agent lifecycle events
//   - [TurnStart] / [TurnEnd]: Turn lifecycle events
//   - [MessageStart] / [MessageUpdate] / [MessageEnd]: Message streaming events
//   - [ToolExecutionStart] / [ToolExecutionEnd]: Tool execution events
//
// ## Steering and Follow-up
//
// The agent supports two mechanisms for injecting messages during execution:
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
// This package depends on [github.com/rahulSailesh-shah/go-pi-ai] for:
//   - Provider abstraction ([provider.Provider])
//   - Type definitions ([types.Message], [types.Tool], etc.)
//
// See the examples directory for complete working examples.
package agent
