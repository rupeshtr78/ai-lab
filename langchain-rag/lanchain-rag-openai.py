from typing import List
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
from logging import getLogger
import streamlit as st
logger = getLogger(__name__)


class LangChainHelper:
    def __init__(self, path: str, model: str, language: str):
        self.path = path
        self.model = model
        self.language = language

    def doc_loader(self) -> List[Document]:
        loader = GenericLoader.from_filesystem(
            self.path,
            glob="**/*",
            suffixes=[".go"],
            exclude=["Dockerfile", "vendor", "docker-compose.yml", "Makefile", "README.md"],
            parser=LanguageParser(language=self.language, parser_threshold=500),
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents.")
        return documents

    def document_chunks(self, documents: List[Document]) -> List[Document]:
        # Split the content of all documents.
        doc_splitter = RecursiveCharacterTextSplitter.from_language(
            language=self.language, chunk_size=512, chunk_overlap=200
        )
        texts = doc_splitter.split_documents(documents)
        return texts

    def get_retriever(self, documents: List[Document]) -> VectorStoreRetriever:
        db = Chroma.from_documents(documents, OpenAIEmbeddings(model="text-embedding-3-large", disallowed_special=()))
        retriever = db.as_retriever(
            search_type="mmr",
            # search_kwargs={"k": 8},
            search_kwargs={'k': 6, 'lambda_mult': 0.25} # Useful if your dataset has many similar documents
        )
        return retriever

    # create a chat prompt template for the QA model
    def chat_prompt_template(self, retriever: VectorStoreRetriever):
        llm = ChatOpenAI(model=self.model)

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

    def chat(self, question: str) -> str:
        documents = self.doc_loader()
        chunks = self.document_chunks(documents)
        retriever = self.get_retriever(chunks)
        qa = self.chat_prompt_template(retriever)
        result = qa.invoke({"input": question})
        return result["answer"]


@click.command()
@click.option('--path', prompt='Enter the path to your documents', help='The path to your documents.')
@click.option('--model', prompt='Enter the model you want to use', default='gpt-4', help='The model to use for QA.')
@click.option('--language', prompt='Enter the programming language of your documents', default='go',
              help='The programming language of the documents.')
def chat_cli(path, model, language):
    helper = LangChainHelper(path, model, language)
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = helper.chat(question)
        click.echo(answer)

def ChatUI():
    # Streamlit UI
    st.title("LangChain Chat Interface")

    path = st.text_input("Enter the path to your documents:", value="")
    model = st.text_input("Enter the model you want to use:", value="gpt-4")
    language = st.text_input("Enter the programming language of your documents:", value="go")

    if 'helper' not in st.session_state or path != st.session_state.get('path', '') or model != st.session_state.get(
            'model', '') or language != st.session_state.get('language', ''):
        st.session_state.helper = LangChainHelper(path, model, language)
        st.session_state.path = path
        st.session_state.model = model
        st.session_state.language = language

    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if question:
            answer = st.session_state.helper.chat(question)
            st.write(answer)
        else:
            st.write("Please enter a question.")


if __name__ == '__main__':
    # chat_cli()
    ChatUI()