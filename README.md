# go-pi-agent

A Go framework for building AI agents with tool execution, streaming responses, and event-driven architecture.

> **Inspired by [pi-mono](https://github.com/badlogic/pi-mono)** - A Go implementation inspired by Mario Zechner's excellent TypeScript AI agent toolkit.

## Overview

`go-pi-agent` provides a flexible foundation for creating specialized AI agents. It handles the complexities of LLM interaction, tool execution, and streaming, allowing you to focus on building your agent's specific capabilities.

### Key Features

- **Tool Execution**: Define tools with schemas and execute functions that the LLM can call
- **Streaming Support**: Real-time event streaming for assistant responses
- **Event-Driven Architecture**: Subscribe to events for fine-grained control over agent behavior
- **Steering and Follow-up**: Interrupt or queue messages during agent execution
- **Thread-Safe**: Safe for concurrent access from multiple goroutines
- **Provider Agnostic**: Works with any [gopiai](https://github.com/rahulSailesh-shah/go-pi-ai) provider (OpenAI, NVIDIA, etc.)

## Installation

```bash
go get github.com/rahulSailesh-shah/go-pi-agent
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "os"
    "time"

    agent "github.com/rahulSailesh-shah/go-pi-agent"
    "github.com/rahulSailesh-shah/go-pi-ai/openai"
)

func main() {
    // Create a provider
    provider, err := openai.NewProvider(openai.Config{
        APIKey: os.Getenv("OPENAI_API_KEY"),
    })
    if err != nil {
        panic(err)
    }

    // Define tools
    tools := []agent.AgentTool{
        {
            Tool: agent.Tool{
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
            Execute: func(toolCallID string, params map[string]any) (agent.ToolMessage, error) {
                location := params["location"].(string)
                return agent.ToolMessage{
                    ToolCallID: toolCallID,
                    ToolName:   "getWeather",
                    Contents:   []agent.Content{agent.TextContent{Text: fmt.Sprintf("Weather in %s: Sunny, 72F", location)}},
                    Timestamp:  time.Now(),
                }, nil
            },
        },
    }

    // Create the agent
    myAgent := agent.NewAgent(&agent.AgentOptions{
        InitialState: &agent.AgentState{
            SystemPrompt: "You are a helpful assistant.",
            Model:        provider,
            ModelName:    "gpt-4o",
            Tools:        tools,
        },
    })

    // Subscribe to events for streaming output
    unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
        switch ev := e.(type) {
        case agent.MessageUpdate:
            if delta, ok := ev.Event.(agent.EventTextDelta); ok {
                fmt.Print(delta.Delta)
            }
        case agent.AgentEnd:
            fmt.Println("\n--- Agent finished ---")
        }
    })
    defer unsubscribe()

    // Send a prompt and wait for completion
    err = myAgent.Prompt(context.Background(), "What's the weather in Tokyo?")
    if err != nil {
        panic(err)
    }
    <-myAgent.WaitForIdle()
}
```

## Core Concepts

### Agent

The `Agent` struct is the main interface for interacting with the agent loop:

```go
myAgent := agent.NewAgent(&agent.AgentOptions{
    InitialState: &agent.AgentState{
        SystemPrompt: "You are a helpful assistant.",
        Model:        provider,
        ModelName:    "gpt-4o",
        Tools:        tools,
    },
    SteeringMode: "one-at-a-time",  // or "all"
    FollowUpMode: "one-at-a-time",  // or "all"
})

myAgent.Prompt(ctx, "Hello!")
<-myAgent.WaitForIdle()
state := myAgent.State()
```

### Tools

Tools are defined using `AgentTool`, which combines a JSON schema with an execution function:

```go
tool := agent.AgentTool{
    Tool: agent.Tool{
        Name:        "searchDatabase",
        Description: "Search the database for records",
        Parameters: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "query": map[string]string{"type": "string"},
            },
            "required": []string{"query"},
        },
    },
    Label: "Database Search",
    Execute: func(toolCallID string, params map[string]any) (agent.ToolMessage, error) {
        query := params["query"].(string)
        // Execute the search...
        return agent.ToolMessage{
            ToolCallID: toolCallID,
            ToolName:   "searchDatabase",
            Contents:   []agent.Content{agent.TextContent{Text: "Found 5 results..."}},
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
    case agent.MessageUpdate:
        switch inner := ev.Event.(type) {
        case agent.EventTextDelta:
            fmt.Print(inner.Delta)
        case agent.EventToolcallStart:
            fmt.Println("Tool call started")
        case agent.EventDone:
            fmt.Printf("Done (reason: %s)\n", inner.Reason)
        }
    case agent.ToolExecutionStart:
        fmt.Printf("Executing tool: %s\n", ev.ToolName)
    case agent.ToolExecutionEnd:
        fmt.Printf("Tool result: %v\n", ev.Result)
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
myAgent.Steer(agent.UserMessage{
    Contents:  []agent.Content{agent.TextContent{Text: "Actually, cancel that and do this instead..."}},
    Timestamp: time.Now(),
})

// Follow-up: Queue for after current execution completes
myAgent.FollowUp(agent.UserMessage{
    Contents:  []agent.Content{agent.TextContent{Text: "Now do this next thing..."}},
    Timestamp: time.Now(),
})
```

## Low-Level API

For advanced use cases, use `AgentLoop` directly:

```go
stream := agent.AgentLoop(ctx, prompts, agentContext, agent.AgentLoopConfig{
    Model:     provider,
    ModelName: "gpt-4o",
    GetSteeringMessages: func() ([]agent.Message, error) { return nil, nil },
    GetFollowUpMessages: func() ([]agent.Message, error) { return nil, nil },
})
defer stream.Close()

for {
    event, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    // Handle events...
}
```

## Using with Different Providers

Any [gopiai](https://github.com/rahulSailesh-shah/go-pi-ai) provider works:

```go
// OpenAI (default)
provider, _ := openai.NewProvider(openai.Config{
    APIKey: os.Getenv("OPENAI_API_KEY"),
})

// NVIDIA
provider, _ := openai.NewProvider(openai.Config{
    APIKey:  os.Getenv("NVIDIA_API_KEY"),
    BaseURL: "https://integrate.api.nvidia.com/v1",
})

// Any OpenAI-compatible endpoint
provider, _ := openai.NewProvider(openai.Config{
    APIKey:  "key",
    BaseURL: "https://my-endpoint.com/v1",
})
```

## Configuration

### Environment Variables

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
        Model:        provider,
        ModelName:    "gpt-4o",
        Tools:        tools,
        Messages:     existingMessages,  // Optional: restore conversation
    },
    SteeringMode: "one-at-a-time",
    FollowUpMode: "all",
    SessionID:    "session-123",
})
```

## Re-exported Types

This package re-exports all core types from [gopiai](https://github.com/rahulSailesh-shah/go-pi-ai) so consumers typically only need to import the `agent` package:

- `agent.Message`, `agent.Content`, `agent.TextContent`, `agent.ImageContent`, `agent.ToolCall`
- `agent.UserMessage`, `agent.AssistantMessage`, `agent.ToolMessage`
- `agent.Tool`, `agent.Request`, `agent.Provider`
- `agent.Event`, `agent.EventTextDelta`, `agent.EventDone`, etc.
- `agent.StopReasonStop`, `agent.StopReasonToolUse`, etc.

## Examples

See the [examples](./examples) directory for complete working examples.

## Dependencies

- [go-pi-ai](https://github.com/rahulSailesh-shah/go-pi-ai) v0.2.1 - LLM provider abstraction

## License

MIT License - see [LICENSE](./LICENSE) for details.
