from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaLLM

from langchain_classic.indexes import SQLRecordManager, index

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

loader = PyPDFLoader("./test.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
split_docs = text_splitter.split_documents(docs)

embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "new_pdf_collection"

vector_store = PGVector(
    embeddings=embeddings_model,
    connection=connection,
    collection_name=collection_name,
    use_jsonb=True,
)

# SQLRecordManager 

namespace = f"pgvector/{collection_name}"
record_manager = SQLRecordManager(
    namespace=namespace,
    db_url=connection
)

record_manager.create_schema()

indexing_stats = index(

    docs_source=split_docs,
    record_manager=record_manager,
    vector_store=vector_store,
    cleanup="incremental",
    source_id_key="source",

)

print("indexing Status:", indexing_stats)

# retriever 
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = OllamaLLM(model="llama3.2", temperature=0.3)

template = """You are a helpful assistant. Use the following context from the document to answer the question.
If you do not know the answer based on the context, say that you do not know.

Context:
{context}

Question:
{question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Query Loop
while True:
    query = input("Enter your question: ")
    response = rag_chain.invoke(query)
    print("\n--- Llama 3.2 Answer ---")
    print(response)
