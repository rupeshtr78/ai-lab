package main

import (
	"chroma-db/cmd/chat"
	"chroma-db/cmd/vectordb"
	"context"
)

func main() {
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	vectordb.RunVectorDb(ctx)

	chat.ChatOllama(ctx)
}
