from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone_operations.operations import delete_pinecone_index, create_pinecone_index
# from constant import INDEX_NAME


def ingest_docs() -> None:
    index_name = "langchain-doc-index"
    loader = ReadTheDocsLoader("E:/python/documentation-helper/langchain-docs/langchain.readthedocs.io/en/latest",
                               encoding='UTF-8')
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted into {len(documents)} documents")
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    # delete_pinecone_index()
    # create_pinecone_index(index_name)

    Pinecone.from_documents(documents, embeddings, index_name=index_name)
    print("****Loading to vectorstore done ***")


if __name__ == '__main__':
    ingest_docs()
