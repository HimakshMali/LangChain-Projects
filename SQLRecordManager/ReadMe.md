**this read me is ai generated for this folder, in this section we added SQLRecordManager to our simple rag app . you can see the rag app in the previous repo.


**for the depedencies see the requirements.txt in the previous repo of SimpleRagApp.if some libraries are missing then pls download then manually 

# Simple RAG App with Deduplicated Ingestion

A local RAG (Retrieval-Augmented Generation) pipeline using LangChain, Ollama, and PostgreSQL (`pgvector`), featuring automated chunk deduplication.

---

## Key Feature: Deduplication via `SQLRecordManager`

This implementation uses LangChain's **`SQLRecordManager`** to prevent duplicate vector ingestion:
* **Ledger Tracking:** `SQLRecordManager` acts as a write ledger for all chunk hashes and metadata.
* **Skip Redundant Ingestions:** Chunks already present in the database are automatically skipped during subsequent runs using `cleanup="incremental"`.
* **Zero Duplication:** Ensures vector storage and compute are not wasted on re-indexing identical source files.

---

