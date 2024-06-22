from langchain_community.document_loaders.generic import GenericLoader, Document
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import click
from typing import List
from logging import getLogger
logger = getLogger(__name__)

# load documents
def doc_loader(repo_path: str, language: str) -> List[Document]:
    # Load all documents from a repository.
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".go"],
        exclude=["Dockerfile", "vendor", "docker-compose.yml", "Makefile", "README.md"],
        parser=LanguageParser(language=language, parser_threshold=500),
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents.")
    return documents

# print documents for debugging
def print_documents(documents: List[Document]) -> None:
    # Print the content of all documents.
    for document in documents:
        print(document.page_content)
        print("\n\n")


# split documents into chunks
def split_documents(documents: List[Document], language: Language) -> List[Document]:
    # Split the content of all documents.
    doc_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=500, chunk_overlap=200
    )
    texts = doc_splitter.split_documents(documents)
    return texts


# print split documents for debugging
def print_split_documents(documents: List[Document]) -> None:
    # Print the content of all documents.
    for document in documents:
        print(document.page_content)
        print("\n\n")

# get retriever from documents embeddings using openai
def get_retriever(documents: List[Document]) -> VectorStoreRetriever:
    # Get the retriever.
    db = Chroma.from_documents(documents, OpenAIEmbeddings(model="text-embedding-3-large", disallowed_special=()))
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"

        search_kwargs={'k': 6, 'lambda_mult': 0.25}
        # search_type="similarity_score_threshold", # Only retrieve documents that have a relevance score above a certain threshold
        # search_kwargs={'score_threshold': 0.6}
    )
    return retriever

# create a chat prompt template for the QA model
def chat_prompt_template(model: str, retriever: VectorStoreRetriever):
    llm = ChatOpenAI(model=model)

    # First we need a prompt that we can pass into an LLM to generate this search query
    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the "
                "conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    # Now we need a prompt that we can pass into an LLM to generate the answer
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    qa = create_retrieval_chain(retriever_chain, document_chain)
    return qa


def chat():
    language = "go"
    model = "gpt-4"
    path = "/Users/rupeshraghavan/apl/gits/gits-rupesh/ai-lab/langchain-rag/nvidia-server-repo"
    documents = doc_loader(path, language)
    retriever = get_retriever(documents)
    qa = chat_prompt_template(model, retriever)

    question = "What is a FetchAllGpuInfo?"
    result = qa.invoke({"input": question})
    return result["answer"]


@click.command()
@click.option('--path', prompt='Enter the path to your documents', help='The path to your documents.', default='./nvidia-server-repo')
@click.option('--model', prompt='Enter the model you want to use', default='gpt-4', help='The model to use for QA.')
@click.option('--language', prompt='Enter the programming language of your documents', default='go', help='The programming language of the documents.')
def chat_cli(path, model, language):
    # Assuming the existence of doc_loader, get_retriever, and chat_prompt_template functions
    documents = doc_loader(path, language)  # Load documents
    chunks = split_documents(documents, language) # Split documents into chunks
    retriever = get_retriever(chunks)  # Initialize retriever
    qa = chat_prompt_template(model, retriever)  # Create QA model instance

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        result = qa.invoke({"input": question})  # Invoke the QA model with the question
        click.echo(result["answer"])


if __name__ == '__main__':
    chat_cli()
