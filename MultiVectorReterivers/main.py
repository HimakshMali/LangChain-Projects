# In standard RAG. The chunk which we embed and retrieved is the same chunk we sent to LLM as a context to process and generate the response.

# While embedding works best on short semantically dense test segments like summaries , key sentences, or abstract question but it sometimes chunking especially raw and large chunks have mixed narratives , and tables and other things which can dilute the semantic search and cause data or table structure to lost during the chunking.

# Also llm needs complete contexts to generate a current response.
# In absence of complete contexts the llm can hallucinate.

# MultiVectorReterival address this by decoupling storage into two layered like (summary/sub-chunks and raw parent chunk) which are lined by same shared uuid

import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Indexing API & Ledger Imports
from langchain_classic.indexes import SQLRecordManager, index


# Load and create large parent chunks
loader = PyPDFLoader("./test.pdf")
docs = loader.load()

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
parent_chunks = parent_splitter.split_documents(docs)
print(f"Total parent chunks created: {len(parent_chunks)}")

#Parallel Summarization Chain via Ollama
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following document chunk concisely for semantic search retrieval. "
    "Focus on key facts, entities, and data points:\n\n{doc}"
)
summary_llm = ChatOllama(model="llama3.2", temperature=0)
summarize_chain = (
    {"doc": lambda x: x.page_content}
    | summary_prompt
    | summary_llm
    | StrOutputParser()
)

print("Generating summaries for parent chunks...")
chunk_summaries = summarize_chain.batch(parent_chunks, {"max_concurrency": 4})

#Configure Embeddings and Vector Store
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "pdf_multi_vector_summaries"

vectorstore = PGVector(
    embeddings=embeddings_model,
    connection=connection,
    collection_name=collection_name,
    use_jsonb=True,
)

docstore = InMemoryStore()
id_key = "doc_id"

# MultiVectorRetriever Setup
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)

# Deterministic ID Generation and Document Pairing
# Generate reproducible UUIDs based on chunk content so re-runs produce identical IDs
doc_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.page_content)) for chunk in parent_chunks]

# Create summary docs containing doc_id and source metadata for tracking
# Create summary docs containing doc_id and source metadata for tracking
summary_docs = [
    Document(
        page_content=summary,
        metadata={
            id_key: doc_ids[i],
            "source": parent_chunks[i].metadata.get("source", "./test.pdf"),
            "page": parent_chunks[i].metadata.get("page", 0)
        }
    )
    for i, summary in enumerate(chunk_summaries)
]

# Populate the in-memory document store with parent chunks
retriever.docstore.mset(list(zip(doc_ids, parent_chunks)))

# 6. SQLRecordManager Ledger & Conditional Indexing
namespace = f"pgvector/{collection_name}"
record_manager = SQLRecordManager(
    namespace=namespace,
    db_url=connection
)
record_manager.create_schema()

# Conditionally index summary docs into PGVector
indexing_stats = index(
    docs_source=summary_docs,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("Indexing Status:", indexing_stats)

# 7. RAG Generation Chain
def format_docs(retrieved_documents):
    return "\n\n".join(doc.page_content for doc in retrieved_documents)

rag_llm = OllamaLLM(model="llama3.2", temperature=0.3)

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
    | rag_llm
    | StrOutputParser()
)

# 8. Query Loop
while True:
    query = input("\nEnter your question (or 'exit' to quit): ")
    if query.strip().lower() == "exit":
        break
    response = rag_chain.invoke(query)
    print("\n--- Llama 3.2 Answer ---")
    print(response)