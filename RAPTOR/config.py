# this file will have all our configurations

import os
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
from langchain_postgres.vectorstores import PGVector
from langchain_classic.indexes import SQLRecordManager

PDF_FILES = [

    "./test2.pdf",
    "./test3.pdf",

]

DB_URL = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
COLLECTION_NAME = "raptor_multi_doc_collection"
NAMESPACE = f"pgvector/{COLLECTION_NAME}"

EMBEDDINGS_MODEL = OllamaEmbeddings(model = "mxbai-embed-large")
SUMMARY_LLM = ChatOllama(model="llama3.2", temperature=0)
RAG_LLM = OllamaLLM(model="llama3.2", temperature=0.3)


def get_vectorstore():
    return PGVector(
        embeddings=EMBEDDINGS_MODEL,
        connection=DB_URL,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )


def get_record_manager():
    record_manager = SQLRecordManager(namespace=NAMESPACE, db_url=DB_URL)
    record_manager.create_schema()
    return record_manager