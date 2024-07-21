package ollamamodel

import (
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
)

func GetOllamaModel(model string) (*ollama.LLM, error) {
	option := ollama.WithModel(model)

	ollamaEmbeder, err := ollama.New(option)
	if err != nil {
		return nil, err
	}
	return ollamaEmbeder, nil

}

func GetOllamaEmbedding(ollamaLLM *ollama.LLM) (*embeddings.EmbedderImpl, error) {

	ollamaEmbeder, err := embeddings.NewEmbedder(ollamaLLM)
	if err != nil {
		return nil, err
	}
	return ollamaEmbeder, nil
}
