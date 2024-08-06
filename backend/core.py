from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()

# from constant import INDEX_NAME

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: []) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff",
    #                                  retriever=docsearch.as_retriever(search_type='similarity'),
    #                                  return_source_documents=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(search_type='similarity'),
        return_source_documents=True
    )
    return qa({'question': query, "chat_history": chat_history})


if __name__ == '__main__':
    print(run_llm("What is langchain"))
