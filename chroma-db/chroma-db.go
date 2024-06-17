package main

import (
	"context"
	"fmt"
	chroma "github.com/amikos-tech/chroma-go"
	"github.com/amikos-tech/chroma-go/collection"
	openai "github.com/amikos-tech/chroma-go/openai"
	"github.com/amikos-tech/chroma-go/types"
	"log"
	"os"
	"time"
)

func ChromaMain() {

	duration := time.Duration(5) * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	// Create a new OpenAI embedding function
	openaiEf, err := getEmbeddingFunction("OPENAI_API_KEY")
	if err != nil {
		log.Fatalf("Error getting embedding function: %s \n", err)
	}

	// Create a new Chroma client
	client, err := getChromaClient("http://0.0.0.0:8070")
	if err != nil {
		log.Fatalf("Error creating client: %s \n", err)
	}

	// Delete the collection if it already exists
	err = deleteCollection(ctx, "test-collection", client)
	if err != nil {
		log.Fatalf("Error deleting collection: %s \n", err)
	}

	// Create a new collection
	newCollection, err := createCollection(ctx, "test-collection", client, openaiEf)
	if err != nil {
		log.Fatalf("Error creating collection: %s \n", err)
	}

	// Create a new record set
	rs, err := createRecordSet(openaiEf)
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

func addRecords(rs *types.RecordSet, ctx context.Context, newCollection *chroma.Collection) error {
	// Add a few records to the record set
	rs.WithRecord(types.WithDocument("My name is John. And I have two dogs."), types.WithMetadata("key1", "value1"))
	rs.WithRecord(types.WithDocument("My name is Jane. I am a data scientist."), types.WithMetadata("key2", "value2"))

	// Build and validate the record set (this will create embeddings if not already present)
	_, err := rs.BuildAndValidate(ctx)
	if err != nil {
		log.Default().Println("Error building and validating record set: %s \n", err)
		return err
	}

	// Add the records to the collection
	_, err = newCollection.AddRecords(context.Background(), rs)
	if err != nil {
		log.Default().Println("Error adding records: %s \n", err)
		return err
	}
	return err
}

func queryRecords(ctx context.Context, newCollection *chroma.Collection) error {
	// Query the collection
	qr, qrerr := newCollection.Query(ctx,
		[]string{"I love dogs"},
		5,
		nil,
		nil,
		nil)

	if qrerr != nil {
		log.Default().Println("Error querying collection: %s \n", qrerr)
		return qrerr
	}
	fmt.Printf("qr: %v\n", qr.Documents[0][0]) //this should result in the document about dogs
	return nil
}

func createRecordSet(openaiEf *openai.OpenAIEmbeddingFunction) (*types.RecordSet, error) {
	// Create a new record set with to hold the records to insert
	rs, err := types.NewRecordSet(
		types.WithEmbeddingFunction(openaiEf),
		types.WithIDGenerator(types.NewULIDGenerator()),
	)
	if err != nil {
		log.Default().Println("Error creating record set: %s \n", err)
		return nil, err
	}
	return rs, nil
}

func createCollection(ctx context.Context, collectionName string, client *chroma.Client, openaiEf *openai.OpenAIEmbeddingFunction) (*chroma.Collection, error) {
	// Create a new collection with options
	newCollection, err := client.NewCollection(
		ctx,
		collection.WithName(collectionName),
		collection.WithMetadata("key1", "value1"),
		collection.WithEmbeddingFunction(openaiEf),
		collection.WithHNSWDistanceFunction(types.L2),
	)
	if err != nil {
		log.Default().Println("Error creating collection: %s \n", err)
		return nil, err
	}
	return newCollection, nil
}

func deleteCollection(ctx context.Context, collectionName string, client *chroma.Client) error {
	//collectionName := "test-collection"
	// Check if the collection already exists
	_, err := client.GetCollection(ctx, collectionName, nil)
	if err != nil {
		log.Default().Println("Error getting collection: %s \n", err)
		return err
	}

	// Collection already exists, Delete the collection
	_, err = client.DeleteCollection(ctx, collectionName)
	if err != nil {
		log.Default().Println("Error deleting collection: %s \n", err)
		return err
	}
	return nil
}

func getChromaClient(path string) (*chroma.Client, error) {
	// Create a new Chroma client "http://0.0.0.0:8070"
	client, err := chroma.NewClient(path)
	if err != nil {
		log.Default().Println("Error creating client: %s \n", err)
		return nil, err
	}
	return client, nil
}

func getEmbeddingFunction(env string) (*openai.OpenAIEmbeddingFunction, error) {
	// Create new OpenAI embedding function
	apiKey := os.Getenv(env)
	openaiEf, err := openai.NewOpenAIEmbeddingFunction(apiKey)
	if err != nil {
		log.Default().Println("Error creating embedding function: %s \n", err)
		return nil, err
	}
	return openaiEf, err
}
