from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders.generic import Document
from langchain_openai import OpenAIEmbeddings
from chromadb.api import AdminAPI, AsyncClientAPI, ClientAPI
import chromadb
from chromadb.config import Settings
import sys
from typing import TYPE_CHECKING, List

class ChromaStoreRetriever:
    def __init__(self, openAiEmbeddings: OpenAIEmbeddings, host: str, port: int):
        self.openAiEmbeddings = openAiEmbeddings or OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=())
        self.host = host or "0.0.0.0"
        self.port = port or 8000

    # start docker-compose for chroma db
    def start_chroma_db(self):
        pass

    def get_chroma_client(self) -> ClientAPI:
        client = None
        try:
            client =chromadb.HttpClient(
                        host=self.host,
                        port=self.port,
                        settings=Settings(
                            chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                            chroma_client_auth_credentials="admin:testDb@rupesh",
                        ),
                    )
        except ValueError:
        # We don't expect to be able to connect to Chroma. We just want to make sure
        # there isn't an ImportError.
            sys.exit(0)
        return client
    
    # async client for chroma db better for streamlit
    def get_chroma_async_client(self) -> AsyncClientAPI:
        client = None
        try:
            client = chromadb.AsyncHttpClient(
                        host=self.host,
                        port=self.port,
                        settings=Settings(
                            chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                            chroma_client_auth_credentials="admin:testDb@rupesh",
                        ),
                    )
        except ValueError:
        # We don't expect to be able to connect to Chroma. We just want to make sure
        # there isn't an ImportError.
            sys.exit(0)
        return client
    


    def get_retriever(self, documents: List[Document]) -> VectorStoreRetriever:
        db = Chroma.from_documents(documents, self.openAiEmbeddings)
        retriever = db.as_retriever(
            search_type="mmr",
            # search_kwargs={"k": 8},
            search_kwargs={'k': 6, 'lambda_mult': 0.25} # Useful if your dataset has many similar documents
        )
        return retriever