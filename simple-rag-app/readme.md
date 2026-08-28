this readme is ai generated for the code ive wrote.

Simple RAG App

A minimal Retrieval-Augmented Generation (RAG) application built with LangChain, Ollama, PostgreSQL, and pgvector.

The app demonstrates the basic RAG pipeline:

PDF → Chunks → Embeddings → pgvector → Retriever → Prompt → LLM → Answer

The implementation is based on the concepts from Chapter 2: RAG Part I — Indexing Your Data of Learning LangChain: Building AI and LLM Applications with LangChain and LangGraph, with additional research and simplification for this project.

Requirements

Python 3.10+

Docker

Ollama

A PDF file named test.pdf

macOS/Linux/Windows

1. Clone the repository

git clone <your-repository-url>
cd simple-rag-app

2. Create a virtual environment

python -m venv .venv
source .venv/bin/activate

Windows:

.venv\Scripts\activate

3. Install Python dependencies

pip install -r requirements.txt

4. Start PostgreSQL + pgvector

Run the pgvector Docker container:

docker run --name pgvector-container   -e POSTGRES_USER=langchain   -e POSTGRES_PASSWORD=langchain   -e POSTGRES_DB=langchain   -p 6024:5432   -d pgvector/pgvector:pg16

Check that it is running:

docker ps

The application connects to:

postgresql+psycopg://langchain:langchain@localhost:6024/langchain

5. Install and prepare Ollama

Install Ollama from the official website, then pull the models used by this project:

ollama pull llama3.2
ollama pull mxbai-embed-large

The code currently uses:

LLM: llama3.2

Embedding model: mxbai-embed-large

Both run locally through Ollama, so your documents and queries do not need to be sent to a cloud LLM provider.

Choosing different models

You can use other Ollama models by changing the model names in simpleRAGapp.py.

For example:

llm = OllamaLLM(model="your-llm-model", temperature=0.3)
embeddings_model = OllamaEmbeddings(model="your-embedding-model")

For a 16 GB Mac, smaller models are generally more practical than very large models.

Cloud/proprietary models such as OpenAI can also be used, but that requires replacing the Ollama LLM/embedding classes and configuring the relevant API key.

6. Add your PDF

Place the PDF you want to query in the project directory and name it:

test.pdf

The current code loads it with:

loader = PyPDFLoader("./test.pdf")

7. Run the application

python simpleRAGapp.py

You should see:

Enter your question:

Ask questions about the PDF.

Press Ctrl+C to stop the application.

How it works

Load PDF — PyPDFLoader reads the document page by page.

Split text — RecursiveCharacterTextSplitter creates smaller chunks.

Create embeddings — OllamaEmbeddings converts chunks into vectors.

Store vectors — PGVector stores the vectors and document data in PostgreSQL.

Retrieve — the retriever finds the most relevant chunks for a question.

Build context — retrieved chunks are formatted into text.

Generate answer — the context and question are passed to llama3.2.

Current chunking configuration:

chunk_size = 1000
chunk_overlap = 100
retrieval k = 2

Check the stored vectors

You can inspect the data created by LangChain in PostgreSQL:

SELECT
    id,
    document,
    cmetadata,
    embedding
FROM langchain_pg_embedding
LIMIT 20;

Project structure

simple-rag-app/
├── simpleRAGapp.py
├── requirements.txt
├── test.pdf
└── README.md

Notes

This is intentionally a simple learning project, not a production-ready RAG system. It focuses on understanding the core indexing and retrieval flow before adding features such as persistent ingestion, metadata filtering, citations, reranking, hybrid search, APIs, or a web UI.

Reference

The project follows concepts from:

Learning LangChain: Building AI and LLM Applications with LangChain and LangGraph
Chapter 2 — RAG Part I: Indexing Your Data

Book reference:
https://github.com/HimakshMali/Books-Collection/blob/main/learning-langchain-building-ai-and-llm-applications-with-langchain-and-langgraph-1.pdf