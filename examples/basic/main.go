// Package main demonstrates basic usage of the go-pi-agent package.
//
// This example shows how to:
//   - Create an agent with tools
//   - Subscribe to events for streaming responses
//   - Send prompts and wait for completion
//
// To run this example from the project root:
//
//	go run ./examples/basic
//
// Or from the examples/basic directory (with .env in project root):
//
//	go run main.go
//
// Make sure to set the required environment variables for your LLM provider.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/joho/godotenv"
	agent "github.com/rahulSailesh-shah/go-pi-agent"
	"github.com/rahulSailesh-shah/go-pi-ai/provider"
	"github.com/rahulSailesh-shah/go-pi-ai/types"
)

func init() {
	// Try to load .env from current directory first
	if err := godotenv.Load(); err != nil {
		// If not found, try to find it in parent directories (for running from examples/basic)
		dir, _ := os.Getwd()
		for i := 0; i < 3; i++ { // Check up to 3 parent directories
			envPath := filepath.Join(dir, ".env")
			if _, err := os.Stat(envPath); err == nil {
				godotenv.Load(envPath)
				break
			}
			dir = filepath.Dir(dir)
		}
	}
}

// RunAgentLoopExample demonstrates the low-level AgentLoop API.
// This provides direct control over the agent execution loop.
func RunAgentLoopExample() {
	ctx := context.Background()
	prompts := []agent.AgentMessage{
		types.UserMessage{
			Timestamp: time.Now(),
			Contents: []types.Content{
				types.TextContent{Text: "What is the weather in Tokyo, Japan?"},
			},
		},
	}
	tools := createTools()

	agentContext := agent.AgentContext{
		SystemPrompt: "You are a helpful assistant. Answer the user's query and use tools if needed.",
		Messages:     prompts,
		Tools:        tools,
	}

	model, err := provider.GetModel(types.ProviderNvidia, "openai/gpt-oss-20b")
	if err != nil {
		log.Fatalf("Failed to get model: %v", err)
	}

	agentLoopConfig := agent.AgentLoopConfig{
		Model: model,
		GetSteeringMessages: func() ([]agent.AgentMessage, error) {
			return []agent.AgentMessage{}, nil
		},
		GetFollowUpMessages: func() ([]agent.AgentMessage, error) {
			return []agent.AgentMessage{}, nil
		},
	}

	stream := agent.AgentLoop(ctx, prompts, agentContext, agentLoopConfig)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for event := range stream.Events {
			fmt.Printf("[DEBUG] Received event type: %T\n", event)
			switch e := event.(type) {
			case agent.MessageUpdate:
				switch inner := e.AssistantMessageEvent.(type) {
				case types.EventStart:
					fmt.Printf("\n[Stream Start]\n")
				case types.EventTextStart:
					// Start of a text block
				case types.EventTextDelta:
					fmt.Print(inner.Delta)
				case types.EventTextEnd:
					// End of a text block
				case types.EventToolcallStart:
					fmt.Printf("\n[Tool Call Start]\n")
				case types.EventToolcallEnd:
					fmt.Printf("\n[Tool Call End]\n")
				case types.EventDone:
					fmt.Printf("\n[Stream Done (Reason: %s)]\n", inner.Reason)
				default:
					fmt.Printf("[DEBUG] Unhandled inner event: %T\n", inner)
				}
			}
		}
		fmt.Println()
	}()

	fmt.Println("[DEBUG] Waiting for stream result/error")
	var finalMessage []agent.AgentMessage
	select {
	case finalMessage = <-stream.Result:
		fmt.Println("[DEBUG] Got result")
	case err := <-stream.Err:
		if err != nil {
			log.Fatalf("Error from stream: %v", err)
		}
	}

	wg.Wait()
	fmt.Println("[DEBUG] Events drained")
	data, err := json.MarshalIndent(finalMessage, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal final message: %v", err)
	}
	log.Printf("Final message: %s", string(data))
}

// RunAgentExample demonstrates the high-level Agent API.
// This is the recommended way to use the package for most use cases.
func RunAgentExample() {
	tools := createTools()

	// Get the model
	model, err := provider.GetModel(types.ProviderNvidia, "openai/gpt-oss-20b")
	if err != nil {
		log.Fatalf("Failed to get model: %v", err)
	}

	// Create the Agent with initial state
	myAgent := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.AgentState{
			SystemPrompt: "You are a helpful assistant. Answer the user's query and use tools if needed.",
			Model:        model,
			Tools:        tools,
		},
		SteeringMode: "one-at-a-time",
		FollowUpMode: "one-at-a-time",
	})

	// Subscribe to events
	unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
		fmt.Printf("[AGENT] Event: %T\n", e)
		switch ev := e.(type) {
		case agent.MessageUpdate:
			switch inner := ev.AssistantMessageEvent.(type) {
			case types.EventStart:
				fmt.Printf("\n[Stream Start]\n")
			case types.EventTextDelta:
				fmt.Print(inner.Delta)
			case types.EventToolcallStart:
				fmt.Printf("\n[Tool Call Start]\n")
			case types.EventToolcallEnd:
				fmt.Printf("\n[Tool Call End]\n")
			case types.EventDone:
				fmt.Printf("\n[Stream Done (Reason: %s)]\n", inner.Reason)
			}
		case agent.AgentEnd:
			fmt.Printf("\n[Agent End] Messages count: %d\n", len(ev.Messages))
		}
	})
	defer unsubscribe()

	// Send a prompt
	fmt.Println("=== Sending prompt: What is the weather in Tokyo, Japan? ===")
	err = myAgent.Prompt(context.Background(), "What is the weather in Tokyo, Japan?")
	if err != nil {
		log.Fatalf("Failed to send prompt: %v", err)
	}

	// Wait for the agent to finish
	<-myAgent.WaitForIdle()

	// Get the final state
	state := myAgent.State()
	fmt.Printf("\n=== Agent finished ===\n")
	fmt.Printf("Messages count: %d\n", len(state.Messages))
	fmt.Printf("Is streaming: %v\n", state.IsStreaming)
	if state.Error != nil {
		fmt.Printf("Error: %s\n", *state.Error)
	}

	// Print all messages
	data, err := json.MarshalIndent(state.Messages, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal messages: %v", err)
	}
	fmt.Printf("\nFinal messages:\n%s\n", string(data))
}

// RunAgentWithTimeoutExample demonstrates using context with timeout.
func RunAgentWithTimeoutExample() {
	tools := createTools()

	// Get the model
	model, err := provider.GetModel(types.ProviderNvidia, "openai/gpt-oss-20b")
	if err != nil {
		log.Fatalf("Failed to get model: %v", err)
	}

	// Create the Agent with initial state
	myAgent := agent.NewAgent(&agent.AgentOptions{
		InitialState: &agent.AgentState{
			SystemPrompt: "You are a helpful assistant. Answer the user's query and use tools if needed.",
			Model:        model,
			Tools:        tools,
		},
		SteeringMode: "one-at-a-time",
		FollowUpMode: "one-at-a-time",
	})

	// Subscribe to events
	unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
		switch ev := e.(type) {
		case agent.MessageUpdate:
			switch inner := ev.AssistantMessageEvent.(type) {
			case types.EventTextDelta:
				fmt.Print(inner.Delta)
			case types.EventDone:
				fmt.Printf("\n[Stream Done (Reason: %s)]\n", inner.Reason)
			}
		case agent.AgentEnd:
			fmt.Printf("\n[Agent End]\n")
		}
	})
	defer unsubscribe()

	// Create a context with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Send a prompt with timeout
	fmt.Println("=== Sending prompt with 10s timeout: What is the meaning of life? ===")
	err = myAgent.Prompt(ctx, "What is the meaning of life? Please provide a very detailed philosophical answer.")
	if err != nil {
		log.Printf("Prompt failed or timed out: %v", err)
	}

	// Wait for the agent to finish or timeout
	select {
	case <-myAgent.WaitForIdle():
		fmt.Println("\n=== Agent completed normally ===")
	case <-ctx.Done():
		fmt.Println("\n=== Context cancelled (timeout) ===")
		myAgent.Abort() // Ensure agent is stopped
	}

	// Get the final state
	state := myAgent.State()
	if state.Error != nil {
		fmt.Printf("Error: %s\n", *state.Error)
	}
}

// createTools returns a list of example tools for demonstration.
func createTools() []agent.AgentTool {
	return []agent.AgentTool{
		{
			Tool: types.Tool{
				Name:        "getWeather",
				Description: "Get the weather for a given location",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]string{"type": "string"},
					},
					"required": []string{"location"},
				},
			},
			Label: "getWeather",
			Execute: func(toolCallId string, params map[string]any) (agent.AgentToolResult, error) {
				return agent.AgentToolResult{
					ToolCallId: toolCallId,
					ToolName:   "getWeather",
					Contents: []types.Content{
						types.TextContent{Text: "Weather in Tokyo, Japan: 72F (22C), partly cloudy"},
					},
					IsError:   false,
					Timestamp: time.Now(),
				}, nil
			},
		},
		{
			Tool: types.Tool{
				Name:        "getStockPrice",
				Description: "Get the stock price for a given company",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"company": map[string]string{"type": "string"},
					},
					"required": []string{"company"},
				},
			},
			Label: "getStockPrice",
			Execute: func(toolCallId string, params map[string]any) (agent.AgentToolResult, error) {
				return agent.AgentToolResult{
					ToolCallId: toolCallId,
					ToolName:   "getStockPrice",
					Contents: []types.Content{
						types.TextContent{Text: "Stock price for Apple: $150.75"},
					},
					IsError:   false,
					Timestamp: time.Now(),
				}, nil
			},
		},
	}
}

func main() {
	// low-level AgentLoop example:
	// RunAgentLoopExample()

	// Run the high-level Agent example (recommended):
	// RunAgentExample()

	// Run the timeout example:
	// RunAgentWithTimeoutExample()
}
