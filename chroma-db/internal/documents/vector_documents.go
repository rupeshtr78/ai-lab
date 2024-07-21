package documents

import (
	"context"
	"errors"
	"log"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

func AddDocuments(ctx context.Context,
	store *chroma.Store,
	documents []schema.Document,
	namespace string) error {

	nsOption := vectorstores.WithNameSpace(namespace)

	// 	// Add documents to the vector store. returns the ids of the added documents.
	docIds, errAd := store.AddDocuments(ctx, documents, nsOption)
	if errAd != nil {
		log.Default().Printf("add documents: %v\n", errAd)
		return chroma.ErrAddDocument
	}
	if len(docIds) != len(documents) {
		log.Default().Printf("add documents: expected %d ids, got %d\n", len(documents), len(docIds))
		return chroma.ErrAddDocument
	}

	return nil

}

func SimilaritySearch(ctx context.Context,
	store *chroma.Store,
	query string,
	numDocuments int,
	namespace string) ([]schema.Document, error) {

	nsOption := vectorstores.WithNameSpace(namespace)

	// 	// Search for similar documents in the vector store.
	// 	// returns the most similar documents to the query.
	similarDocs, errSs := store.SimilaritySearch(ctx, query, numDocuments, nsOption)
	if errSs != nil {
		log.Default().Printf("similarity search: %v\n", errSs)
		return nil, errSs
	}
	if len(similarDocs) == 0 {
		log.Default().Printf("similarity search: no similar documents found\n")
		return nil, errors.New("no similar documents found")
	}

	return similarDocs, nil

}

func DeleteCollection(ctx context.Context,
	store *chroma.Store) error {
	// Delete collection and all documents in the vector store.
	errDc := store.RemoveCollection()
	if errDc != nil {
		log.Default().Printf("delete collection: %v\n", errDc)
		return errors.New("delete collection failed")
	}

	return nil

}
