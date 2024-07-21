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
		log.Default().Println(err)
		return
	}

	// Get the list of all the available collections
	collections, err2 := client.ListCollections(ctx)
	if err2 != nil {
		log.Default().Println(err2)
	}

	// Print the list of databases

	for _, col := range collections {
		log.Default().Printf("Collection: %v\n", col.Name)
		log.Default().Printf("Database: %v\n", col.Database)
		log.Default().Printf("Tenant: %v\n", col.Tenant)
	}
}

// 2024/07/21 14:39:53 Collection: chroma-ollama
// 2024/07/21 14:39:53 Database: default_database
// 2024/07/21 14:39:53 Tenant: default_tenant
