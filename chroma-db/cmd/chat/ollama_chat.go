package chat

import (
	"chroma-db/internal/constants"
	ollamamodel "chroma-db/internal/ollama"
	"context"
	"log"

	"github.com/tmc/langchaingo/llms"
)

func ChatOllama(ctx context.Context) {

	l, err3 := ollamamodel.GetOllamaModel(constants.OllamaUrl, constants.OllamaModel)
	if err3 != nil {
		log.Default().Println(err3)
	}

	prompt := "Why is Sky Blue?"
	s, err4 := l.Call(ctx, prompt,
		llms.WithMaxTokens(2048),
		llms.WithSeed(52),
		llms.WithTemperature(0.5), // 0.5 0.9
		llms.WithTopP(0.9),
		// llms.WithTopK(40),
	)
	if err4 != nil {
		log.Default().Println(err4)
	}
	log.Default().Println(s)
}
