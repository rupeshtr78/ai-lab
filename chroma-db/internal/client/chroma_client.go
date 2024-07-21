package client

import (
	"context"

	chromago "github.com/amikos-tech/chroma-go"
	"github.com/amikos-tech/chroma-go/types"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

func GetChromaClient(ctx context.Context, url string) (*chromago.Client, error) {
	// create the client connection and confirm that we can access the server with it
	chromaClient, err := chromago.NewClient(url)
	if err != nil {
		return nil, err
	}

	if _, errHb := chromaClient.Heartbeat(ctx); errHb != nil {
		return nil, errHb
	}

	return chromaClient, err
}

func CreateChromaStore(ctx context.Context,
	chromaUrl string,
	nameSpace string,
	embedder embeddings.Embedder,
	distanceFunction types.DistanceFunction) (*chroma.Store, error) {

	s, err := chroma.New(
		chroma.WithChromaURL(chromaUrl),
		chroma.WithNameSpace(nameSpace),
		chroma.WithEmbedder(embedder),
		chroma.WithDistanceFunction(distanceFunction),
	)
	if err != nil {
		return nil, err
	}

	return &s, nil

}
