package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/joho/godotenv"
	agent "github.com/rahulSailesh-shah/go-pi-agent"
	"github.com/rahulSailesh-shah/go-pi-ai/openai"
)

func init() {
	if err := godotenv.Load(); err != nil {
		dir, _ := os.Getwd()
		for range 3 {
			envPath := filepath.Join(dir, ".env")
			if _, err := os.Stat(envPath); err == nil {
				godotenv.Load(envPath)
				break
			}
			dir = filepath.Dir(dir)
		}
	}
}

// getProvider creates an OpenAI-compatible provider.
func getProvider() agent.Provider {
	provider, err := openai.NewProvider(openai.Config{
		APIKey:  os.Getenv("NVIDIA_API_KEY"),
		BaseURL: os.Getenv("NVIDIA_BASE_URL"),
	})
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	return provider
}

// RunAgentLoopExample demonstrates the low-level AgentLoop API
// using the Stream iterator (Recv/Close) pattern.
func RunAgentLoopExample() {
	ctx := context.Background()
	prompts := []agent.Message{
		agent.UserMessage{
			Timestamp: time.Now(),
			Contents: []agent.Content{
				agent.TextContent{Text: "What is the weather in Tokyo, Japan?"},
			},
		},
	}
	tools := createTools()

	agentContext := agent.AgentContext{
		SystemPrompt: "You are a helpful assistant. Answer the user's query and use tools if needed.",
		Messages:     prompts,
		Tools:        tools,
	}

	config := agent.AgentLoopConfig{
		Provider:  getProvider(),
		ModelName: "gpt-oss-120b",
	}

	stream := agent.AgentLoop(ctx, prompts, agentContext, config)
	defer stream.Close()

	var finalMessages []agent.Message

	for {
		event, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Error from stream: %v", err)
		}

		switch e := event.(type) {
		case agent.MessageUpdate:
			switch inner := e.Event.(type) {
			case agent.EventTextDelta:
				fmt.Print(inner.Delta)
			case agent.EventToolcallStart:
				fmt.Printf("\n[Tool Call Start]\n")
			case agent.EventToolcallEnd:
				fmt.Printf("\n[Tool Call End]\n")
			case agent.EventDone:
				fmt.Printf("\n[Stream Done (Reason: %s)]\n", inner.Reason)
			}
		case agent.AgentEnd:
			finalMessages = e.Messages
		}
	}

	fmt.Println()
	data, err := json.MarshalIndent(finalMessages, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal final messages: %v", err)
	}
	log.Printf("Final messages: %s", string(data))
}

// RunAgentExample demonstrates the high-level Agent API.
func RunAgentExample() {
	tools := createTools()

	myAgent := agent.NewAgent(
		agent.WithInitialState(&agent.AgentState{
			SystemPrompt: "You are a helpful assistant. Answer the user's query and use tools if needed.",
			Provider:     getProvider(),
			ModelName:    "openai/gpt-oss-120b",
			Tools:        tools,
		}),
	)

	unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
		switch ev := e.(type) {
		case agent.AgentStart:
			fmt.Printf("\n>> agent_start\n")
		case agent.TurnStart:
			fmt.Printf("\n>> turn_start\n")
		case agent.MessageStart:
			fmt.Printf("\n>> message_start (role=%s)\n", ev.Message.Role())
		case agent.MessageUpdate:
			switch inner := ev.Event.(type) {
			case agent.EventTextDelta:
				fmt.Print(inner.Delta)
			case agent.EventToolcallStart:
				fmt.Printf("\n   [tool_call_start]\n")
			case agent.EventToolcallEnd:
				fmt.Printf("\n   [tool_call_end]\n")
			case agent.EventDone:
				fmt.Printf("\n   [stream_done reason=%s]\n", inner.Reason)
			}
		case agent.MessageEnd:
			fmt.Printf("\n>> message_end (role=%s)\n", ev.Message.Role())
		case agent.ToolExecutionStart:
			fmt.Printf("\n>> tool_execution_start (name=%s id=%s)\n", ev.ToolName, ev.ToolCallID)
		case agent.ToolExecutionEnd:
			fmt.Printf("\n>> tool_execution_end (name=%s id=%s isError=%t)\n", ev.ToolName, ev.ToolCallID, ev.IsError)
		case agent.TurnEnd:
			fmt.Printf("\n>> turn_end (toolResults=%d)\n", len(ev.ToolResults))
		case agent.AgentEnd:
			fmt.Printf("\n>> agent_end (messages=%d)\n", len(ev.Messages))
		}
	})
	defer unsubscribe()

	fmt.Println("=== Sending prompt ===")
	err := myAgent.Prompt(context.Background(),
		"What is the weather in Tokyo, Japan? and based on the weather, what is the best activity to do in Tokyo? and after you are done fetch the stock price for Apple")
	if err != nil {
		log.Fatalf("Failed to send prompt: %v", err)
	}

	<-myAgent.WaitForIdle()

	state := myAgent.State()
	fmt.Printf("\n=== Agent finished ===\n")
	fmt.Printf("Messages count: %d\n", len(state.Messages))
	if state.Error != nil {
		fmt.Printf("Error: %s\n", *state.Error)
	}

	data, err := json.MarshalIndent(state.Messages, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal messages: %v", err)
	}
	fmt.Printf("\nFinal messages:\n%s\n", string(data))
}

// RunAgentWithTimeoutExample demonstrates using context with timeout.
func RunAgentWithTimeoutExample() {
	tools := createTools()

	myAgent := agent.NewAgent(
		agent.WithInitialState(&agent.AgentState{
			SystemPrompt: "You are a helpful assistant. Answer the user's query and use tools if needed.",
			Provider:     getProvider(),
			ModelName:    "openai/gpt-oss-20b",
			Tools:        tools,
		}),
	)

	unsubscribe := myAgent.Subscribe(func(e agent.AgentEvent) {
		switch ev := e.(type) {
		case agent.MessageUpdate:
			switch inner := ev.Event.(type) {
			case agent.EventTextDelta:
				fmt.Print(inner.Delta)
			case agent.EventDone:
				fmt.Printf("\n[Stream Done (Reason: %s)]\n", inner.Reason)
			}
		case agent.AgentEnd:
			fmt.Printf("\n[Agent End]\n")
		}
	})
	defer unsubscribe()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fmt.Println("=== Sending prompt with 10s timeout ===")
	err := myAgent.Prompt(ctx, "What is the meaning of life? Please provide a very detailed philosophical answer.")
	if err != nil {
		log.Printf("Prompt failed: %v", err)
	}

	select {
	case <-myAgent.WaitForIdle():
		fmt.Println("\n=== Agent completed normally ===")
	case <-ctx.Done():
		fmt.Println("\n=== Context cancelled (timeout) ===")
		myAgent.Abort()
	}

	state := myAgent.State()
	if state.Error != nil {
		fmt.Printf("Error: %s\n", *state.Error)
	}
}

func createTools() []agent.AgentTool {
	return []agent.AgentTool{
		{
			Tool: agent.Tool{
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
			Execute: func(toolCallID string, params map[string]any) (agent.ToolMessage, error) {
				return agent.ToolMessage{
					ToolCallID: toolCallID,
					ToolName:   "getWeather",
					Contents: []agent.Content{
						agent.TextContent{Text: "Weather in Tokyo, Japan: 72F (22C), partly cloudy"},
					},
					Timestamp: time.Now(),
				}, nil
			},
		},
		{
			Tool: agent.Tool{
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
			Execute: func(toolCallID string, params map[string]any) (agent.ToolMessage, error) {
				return agent.ToolMessage{
					ToolCallID: toolCallID,
					ToolName:   "getStockPrice",
					Contents: []agent.Content{
						agent.TextContent{Text: "Stock price for Apple: $150.75"},
					},
					Timestamp: time.Now(),
				}, nil
			},
		},
	}
}

func main() {
	// Uncomment the example you want to run:

	// Low-level AgentLoop example:
	// RunAgentLoopExample()

	// High-level Agent example (recommended):
	RunAgentExample()

	// Timeout example:
	// RunAgentWithTimeoutExample()
}
