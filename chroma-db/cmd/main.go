package main

import (
	"chroma-db/cmd/vectordb"
	"context"
	"time"
)

func main() {
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, time.Second*5)
	defer cancel()

	vectordb.RunVectorDb(ctx)

	// chat.ChatOllama(ctx)
}
