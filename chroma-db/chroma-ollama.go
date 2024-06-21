package main

import (
	"context"
	"fmt"
	"log"
	"time"

	chroma "github.com/amikos-tech/chroma-go"
	"github.com/amikos-tech/chroma-go/collection"
	"github.com/amikos-tech/chroma-go/ollama"
	"github.com/amikos-tech/chroma-go/types"
)

func ChromaOllamaMain(collectionName string) {

	duration := time.Duration(5) * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	// Create a new Ollama embedding function
	ollamaEf, err := GetOllamEmbeddingFunction("ollama-model")
	if err != nil {
		log.Fatalf("Error getting embedding function: %s \n", err)
	}

	// Create a new Chroma client
	client, err := getChromaClient("http://0.0.0.0:8070")
	if err != nil {
		log.Fatalf("Error creating client: %s \n", err)
	}

	// Delete the collection if it already exists
	// err = deleteCollection(ctx, collectionName, client)
	// if err != nil {
	// log.Fatalf("Error deleting collection: %s \n", err)
	// }

	// Create a new collection
	newCollection, err := createOllamaCollection(ctx, collectionName, client, ollamaEf)
	if err != nil {
		log.Fatalf("Error creating collection: %s \n", err)
	}

	// Create a new record set
	rs, err := createOllamaRecordSet(ollamaEf)
	if err != nil {
		log.Fatalf("Error creating record set: %s \n", err)
	}

	// Add records to the collection
	err = addRecords(rs, ctx, newCollection)
	if err != nil {
		log.Fatalf("Error adding records: %s \n", err)
	}

	// Count the number of documents in the collection
	countDocs, qrerr := newCollection.Count(ctx)
	if qrerr != nil {
		log.Fatalf("Error counting documents: %s \n", qrerr)
	}
	fmt.Printf("countDocs: %v\n", countDocs) //this should result in 2

	err = queryRecords(ctx, newCollection)
	if err != nil {
		log.Fatalf("Error querying records: %s \n", err)
	}
}

// GetOllamaEmbedding returns an Ollama embedding function
func GetOllamaClient(model string) (*ollama.OllamaClient, error) {
	ollamaClient, err := ollama.NewOllamaClient(ollama.WithModel(model))
	if err != nil {
		return nil, err
	}

	return ollamaClient, nil
}

func GetOllamEmbeddingFunction(model string) (*ollama.OllamaEmbeddingFunction, error) {

	enbeddingFn, err := ollama.NewOllamaEmbeddingFunction(
		ollama.WithModel(model))
	if err != nil {
		log.Fatalf("Error getting embedding function: %s \n", err)
		return nil, err
	}
	return enbeddingFn, nil

}

func createOllamaRecordSet(ollamaEf *ollama.OllamaEmbeddingFunction) (*types.RecordSet, error) {

	// Create a new record set with to hold the records to insert
	rs, err := types.NewRecordSet(
		types.WithEmbeddingFunction(ollamaEf),
		types.WithIDGenerator(types.NewULIDGenerator()),
	)
	if err != nil {
		log.Default().Printf("Error creating record set: %s \n", err)
		return nil, err
	}
	return rs, nil
}

func createOllamaCollection(ctx context.Context, collectionName string, client *chroma.Client, ollamaEf *ollama.OllamaEmbeddingFunction) (*chroma.Collection, error) {
	// Create a new collection with options
	newCollection, err := client.NewCollection(
		ctx,
		collection.WithName(collectionName),
		collection.WithMetadata("key1", "value1"),
		collection.WithEmbeddingFunction(ollamaEf),
		collection.WithHNSWDistanceFunction(types.L2),
	)
	if err != nil {
		log.Default().Printf("Error creating collection: %s \n", err)
		return nil, err
	}
	return newCollection, nil
}
