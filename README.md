# go-pi-agent

A Go framework for building AI agents with tool execution, streaming responses, and event-driven architecture.

## Overview

`go-pi-agent` provides a flexible foundation for creating specialized AI agents. It handles the complexities of LLM interaction, tool execution, and streaming, allowing you to focus on building your agent's specific capabilities.

### Key Features

- **Tool Execution**: Define tools with schemas and execute functions that the LLM can call
- **Streaming Support**: Real-time event streaming for assistant responses
- **Event-Driven Architecture**: Subscribe to events for fine-grained control over agent behavior
- **Steering and Follow-up**: Interrupt or queue messages during agent execution
- **Thread-Safe**: Safe for concurrent access from multiple goroutines
- **Provider Agnostic**: Works with multiple LLM providers (OpenAI, NVIDIA, etc.)

## Installation

```bash
go get github.com/rahulSailesh-shah/go-pi-agent
```

## Quick Start

```go
package main

import (
    "fmt"
    "time"

    agent "github.com/rahulSailesh-shah/go-pi-agent"
    "github.com/rahulSailesh-shah/go-pi-ai/provider"
    "github.com/rahulSailesh-shah/go-pi-ai/types"
)

func main() {
    // Define tools
    tools := []agent.AgentTool{
        {
            Tool: types.Tool{
                Name:        "getWeather",
                Description: "Get the weather for a location",
                Parameters: map[string]any{
                    "type": "object",
                    "properties": map[string]any{
                        "location": map[string]string{"type": "string"},
                    },
                    "required": []string{"location"},
                },
            },
            Execute: func(toolCallId string, params map[string]any) (agent.AgentToolResult, error) {
                location := params["location"].(string)
                return types.ToolMessage{
                    ToolCallId: toolCallId,
                    ToolName:   "getWeather",
                    Contents:   []types.Content{types.TextContent{Text: fmt.Sprintf("Weather in %s: Sunny, 72F", location)}},
                    Timestamp:  time.Now(),
                }, nil
            },
        },
    }

    // Get a model provider
    model, err := provider.GetModel(types.ProviderOpenAI, "gpt-4")
    if err != nil {
        panic(err)
    }

    // Create the agent
    myAgent := agent.NewAgent(&agent.AgentOptions{
        InitialState: &agent.AgentState{
            SystemPrompt: "You are a helpful assistant.",
            Model:        model,
            Tools:        tools,
        },
    })

    // Subscribe to events for streaming output
    unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
        switch ev := e.(type) {
        case agent.MessageUpdate:
            if delta, ok := ev.AssistantMessageEvent.(types.EventTextDelta); ok {
                fmt.Print(delta.Delta)
            }
        case agent.AgentEnd:
            fmt.Println("\n--- Agent finished ---")
        }
    })
    defer unsubscribe()

    // Send a prompt
    err = myAgent.Prompt("What's the weather in Tokyo?")
    if err != nil {
        panic(err)
    }

    // Wait for completion
    <-myAgent.WaitForIdle()
}
```

## Core Concepts

### Agent

The `Agent` struct is the main interface for interacting with the agent loop. It provides:

- State management (system prompt, model, tools, messages)
- Event subscription for streaming
- Lifecycle control (prompt, abort, reset)
- Steering and follow-up message queuing

```go
// Create an agent
myAgent := agent.NewAgent(&agent.AgentOptions{
    InitialState: &agent.AgentState{
        SystemPrompt: "You are a helpful assistant.",
        Model:        model,
        Tools:        tools,
    },
    SteeringMode: "one-at-a-time",  // or "all"
    FollowUpMode: "one-at-a-time",  // or "all"
})

// Send a prompt
myAgent.Prompt("Hello!")

// Wait for completion
<-myAgent.WaitForIdle()

// Get the final state
state := myAgent.State()
```

### Tools

Tools are defined using `AgentTool`, which combines a JSON schema with an execution function:

```go
tool := agent.AgentTool{
    Tool: types.Tool{
        Name:        "searchDatabase",
        Description: "Search the database for records",
        Parameters: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "query": map[string]string{
                    "type":        "string",
                    "description": "The search query",
                },
                "limit": map[string]any{
                    "type":        "integer",
                    "description": "Maximum results to return",
                    "default":     10,
                },
            },
            "required": []string{"query"},
        },
    },
    Label: "Database Search",  // Optional human-readable label
    Execute: func(toolCallId string, params map[string]any) (agent.AgentToolResult, error) {
        query := params["query"].(string)
        // Execute the search...
        return types.ToolMessage{
            ToolCallId: toolCallId,
            ToolName:   "searchDatabase",
            Contents:   []types.Content{types.TextContent{Text: "Found 5 results..."}},
            Timestamp:  time.Now(),
        }, nil
    },
}
```

### Events

Subscribe to events for real-time updates:

```go
unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
    switch ev := e.(type) {
    case agent.AgentStart:
        fmt.Println("Agent started")
    case agent.TurnStart:
        fmt.Println("Turn started")
    case agent.MessageStart:
        fmt.Println("Message started")
    case agent.MessageUpdate:
        // Handle streaming content
        switch inner := ev.AssistantMessageEvent.(type) {
        case types.EventTextDelta:
            fmt.Print(inner.Delta)
        case types.EventToolcallStart:
            fmt.Printf("Calling tool: %s\n", inner.ToolCall.Name)
        }
    case agent.MessageEnd:
        fmt.Println("Message complete")
    case agent.ToolExecutionStart:
        fmt.Printf("Executing tool: %s\n", ev.ToolName)
    case agent.ToolExecutionEnd:
        fmt.Printf("Tool result: %v\n", ev.Result)
    case agent.TurnEnd:
        fmt.Println("Turn ended")
    case agent.AgentEnd:
        fmt.Printf("Agent finished with %d messages\n", len(ev.Messages))
    }
})
defer unsubscribe()
```

### Steering and Follow-up

Interrupt or queue messages during execution:

```go
// Steering: Interrupt mid-execution (e.g., during tool calls)
myAgent.Steer(types.UserMessage{
    Contents: []types.Content{types.TextContent{Text: "Actually, cancel that and do this instead..."}},
    Timestamp: time.Now(),
})

// Follow-up: Queue for after current execution completes
myAgent.FollowUp(types.UserMessage{
    Contents: []types.Content{types.TextContent{Text: "Now do this next thing..."}},
    Timestamp: time.Now(),
})
```

## Low-Level API

For advanced use cases, use `AgentLoop` directly:

```go
ctx := context.Background()

agentContext := agent.AgentContext{
    SystemPrompt: "You are a helpful assistant.",
    Messages:     prompts,
    Tools:        tools,
}

config := agent.AgentLoopConfig{
    Model: model,
    GetSteeringMessages: func() ([]agent.AgentMessage, error) {
        return nil, nil
    },
    GetFollowUpMessages: func() ([]agent.AgentMessage, error) {
        return nil, nil
    },
}

stream := agent.AgentLoop(ctx, prompts, agentContext, config)

// Process events
for event := range stream.Events {
    // Handle events...
}

// Get result
result := <-stream.Result
```

## Configuration

### Environment Variables

The underlying `go-pi-ai` package uses environment variables for provider configuration:

```bash
# OpenAI
OPENAI_API_KEY=your-api-key

# NVIDIA
NVIDIA_API_KEY=your-api-key
```

### Agent Options

```go
agent.NewAgent(&agent.AgentOptions{
    InitialState: &agent.AgentState{
        SystemPrompt: "...",
        Model:        model,
        Tools:        tools,
        Messages:     existingMessages,  // Optional: restore conversation
    },
    
    // Message processing modes
    SteeringMode: "one-at-a-time",  // Process steering messages one at a time
    FollowUpMode: "all",            // Process all follow-up messages at once
    
    // Optional session identifier
    SessionId: "session-123",
    
    // Optional: Custom message filtering before LLM calls
    ConvertToLlm: func(msgs []agent.AgentMessage) ([]types.Message, error) {
        // Filter or transform messages...
        return msgs, nil
    },
    
    // Optional: Transform context before each LLM call
    TransformContext: func(msgs []agent.AgentMessage, ctx context.Context) ([]agent.AgentMessage, error) {
        // Add context, filter messages, etc.
        return msgs, nil
    },
})
```

## API Reference

Full API documentation is available at [pkg.go.dev](https://pkg.go.dev/github.com/rahulSailesh-shah/go-pi-agent).

## Examples

See the [examples](./examples) directory for complete working examples:

- [basic](./examples/basic) - Basic usage with both high-level and low-level APIs

## Dependencies

- [go-pi-ai](https://github.com/rahulSailesh-shah/go-pi-ai) - LLM provider abstraction

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
