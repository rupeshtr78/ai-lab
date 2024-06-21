package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

func OllamaMain() {
	llm, err := ollama.New(ollama.WithModel("llama3"))
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	resultStream := llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return nil
	})

	completion, err := llm.Call(ctx, "Human: Who was the first man to walk on the moon?\nAssistant:",
		llms.WithTemperature(0.8),
		resultStream,
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Answer\n", completion)
	// _ = completion
}
