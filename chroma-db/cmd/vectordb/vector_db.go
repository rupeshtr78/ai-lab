package vectordb

import (
	"chroma-db/internal/client"
	"chroma-db/internal/constants"
	"context"
	"log"
)

func RunVectorDb(ctx context.Context) {
	// ctx := context.Background()
	// ctx, cancel := context.WithCancel(ctx)
	// defer cancel()
	// Create a new instance of the server
	client, err := client.GetChromaClient(ctx, constants.ChromaUrl)
	if err != nil {
		panic(err)
	}

	// Get the list of all the available databases
	c, err2 := client.ListCollections(ctx)
	if err2 != nil {
		log.Default().Println(err2)
	}

	for _, v := range c {
		log.Default().Println(v.Name)
	}
}
