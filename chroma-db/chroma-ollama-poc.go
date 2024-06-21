package main

import (
	"context"
	"fmt"
	"log"

	// chroma "github.com/amikos-tech/chroma-go"
	chroma "github.com/amikos-tech/chroma-go"
	"github.com/tmc/langchaingo/llms/ollama"
)

func PocMain() {
	// Initialize Ollama client
	ollamaClient, err := ollama.New(ollama.WithModel("llama2"))
	if err != nil {
		log.Fatalf("Failed to create Ollama client: %v", err)
	}

	// Initialize ChromaDB client
	chromaClient, err := chroma.NewClient("http://localhost:8000")
	if err != nil {
		log.Fatalf("Failed to create ChromaDB client: %v", err)
	}

	ctx := context.Background()

	// Create a collection
	collection, err := chromaClient.CreateCollection(ctx, "my_collection")
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Function to get embeddings using Ollama
	getEmbeddings := func(texts []string) ([][]float32, error) {
		embeddings := make([][]float32, len(texts))
		for i, text := range texts {
			embedding, err := ollamaClient.CreateEmbedding(context.Background(), text)
			if err != nil {
				return nil, fmt.Errorf("failed to create embedding: %v", err)
			}
			embeddings[i] = embedding
		}
		return embeddings, nil
	}

	// Add documents to the collection
	documents := []string{"This is a document", "This is another document"}
	embeddings, err := getEmbeddings(documents)
	if err != nil {
		log.Fatalf("Failed to get embeddings: %v", err)
	}

	ids := []string{"id1", "id2"}
	metadatas := []map[string]interface{}{
		{"source": "my_source"},
		{"source": "my_source"},
	}

	err = collection.Add(ids, embeddings, metadatas, documents)
	if err != nil {
		log.Fatalf("Failed to add documents: %v", err)
	}

	// Query the collection
	queryText := "This is a query document"
	queryEmbedding, err := getEmbeddings([]string{queryText})
	if err != nil {
		log.Fatalf("Failed to get query embedding: %v", err)
	}

	results, err := collection.Query(queryEmbedding[0], 2, nil, nil, nil)
	if err != nil {
		log.Fatalf("Failed to query collection: %v", err)
	}

	fmt.Println("Query results:")
	for i, id := range results.Ids[0] {
		fmt.Printf("ID: %s, Distance: %f\n", id, results.Distances[0][i])
	}
}
