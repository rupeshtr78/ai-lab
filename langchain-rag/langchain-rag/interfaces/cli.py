import click
from langchain_openai import OpenAIEmbeddings
from chroma_vector_db import ChromaStoreRetriever

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
    openAiEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=())
    helper = LangChainHelper(path, model, language, openAiEmbeddings)
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = helper.chat(question)
        click.echo(answer)