# Here we are using the idea of  RAPTOR for broad or thementic reterival.
# what we are doing here is creating summaries of chunks and clustering them in multiple levels

# we are using GMM(gaussian Mixing model) to cluster the chunks semantically like this

'''
**by ai 
RAW CHUNKS (Points in Vector Space)        GMM CLUSTERING (num_clusters = 3)
       •   •                                   ┌─ Cluster 0 ──┐
     •   •   •                                 │  •   •   •   │
                                               └──────────────┘
         •   •                                 ┌─ Cluster 1 ──┐
       •   •   •                               │  •   •   •   │
                                               └──────────────┘
   •   •                                       ┌─ Cluster 2 ──┐
 •   •   •                                     │  •   •   •   │
                                               └──────────────┘

'''
# then we are summarizing these clusters using llm and storing them 



import os
import uuid
import numpy as np
from sklearn.mixture import GaussianMixture
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.indexes import index

from config import (
    PDF_FILES,
    EMBEDDINGS_MODEL,
    SUMMARY_LLM,
    get_vectorstore,
    get_record_manager,

)

summary_promt = ChatPromptTemplate.from_template(
    "Synthesize and summarize the core themes, relationships, and key facts across the following document sections:\n\n{doc}"
)

summarize_chain = summary_promt | SUMMARY_LLM | StrOutputParser()


# the function belows groups the documents into sementic clusters ussing Gaussian Misture Models
# Updated cluster_documents with consistent plural naming (num_clusters)
#this is Gaussian Mixture Model with scikit-learn

def cluster_documents(docs: list[Document], num_clusters: int = 3) -> list[list[Document]]:

# what is does is that it take the list of the documents , generates vector embeddig for their texts
# and clusters them into num_clusters group based on sementic similarities


    if len(docs) <= num_clusters: #this is gaurd clause: if there are fewer documents tan desired clusters, then retrun them as single cluster
        return [docs]
# extracting the raw string from all document
    text = [d.page_content for d in docs]
    # generated the dense numerical embeddingfor each document using ollama
    vectors = np.array(EMBEDDINGS_MODEL.embed_documents(texts=text))
    # fit gaussian mixuture model to find 'num_clusters' center points in the vector space
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    #  labels is an array of integer cluster assignments for each vector (e.g., [0, 2, 1, 0, 1])
    labels = gmm.fit_predict(vectors)
# it groups the original document objects according to the assigned cluster labels
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(docs[idx])

        # filter out any empty clusters and return the grouped lists
    return [c for c in clusters if c]

# recursively clusters and summrizes docuument nodes
def build_raptor_tree(docs: list[Document], current_level: int = 1, max_levels: int = 2) -> list[Document]:
    """
    Recursively clusters and summarizes document nodes until max_levels is reached.
    - Level 0: Raw Leaf Chunks (input)
    - Level 1: Summaries of clustered Level 0 chunks
    - Level 2: Summaries of clustered Level 1 summaries (Root/Global)
    """
# Base Case: Stop recursion if we exceed max allowed depth or have insufficient documents
    if current_level > max_levels or len(docs) <=1:
        return []
   
    print(f"--> Building RAPTOR Tree Level{current_level} across {len(docs)} nodes.... ")

# Calculate a proportional number of clusters based on document count
    num_clusters = max(2, len(docs) // 3)
    clusters = cluster_documents(docs, num_clusters = num_clusters)

    level_summaries = []
    # sumarize each cluster
    for cluster in clusters:
        # Merge all chunk texts in this cluster into a single text block
        merged_text = "\n\n".join([d.page_content for d in cluster])
        # Collect unique file sources contributing to this cluster summary
        sources = list(set([d.metadata.get("source", "unknown") for d in cluster]))
        # Invoke the LLM summarizer chain on the merged text
        summary_content = summarize_chain.invoke({"doc": merged_text})
        # Wrap the LLM summary in a new Document object with hierarchy metadata
        summary_doc = Document(
            page_content=summary_content,
            metadata={
                "source": ",".join(sources),
                "tree_level": current_level,
                "node_type": "raptor_summary"
            }
        )
        level_summaries.append(summary_doc)

    higher_levels = build_raptor_tree(level_summaries, current_level + 1, max_levels)
    return level_summaries + higher_levels

def run_ingestion():
    vectorstore = get_vectorstore()
    record_manager = get_record_manager()
    all_leaf_docs = []

    for pdf_path in PDF_FILES:
        if not os.path.exists(pdf_path):
            print(f"Warning: File{pdf_path} not found. skipping.")
            continue

        print(f"Loading and Chunking: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()

        leaf_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150 )
        chunks = leaf_splitter.split_documents(raw_docs)


        for c in chunks:
            c.metadata["source"] = pdf_path
            c.metadata["tree_level"] = 0
            c.metadata["node_type"] = "leaf_chunk"
            
        all_leaf_docs.extend(chunks)

        if not all_leaf_docs:
            print("All documents are up-to-date in the ledger. No new chunks to process.")
            return

    print(f"Total new leaf chunks generated: {len(all_leaf_docs)}")
    
    # Generate RAPTOR hierarchical tree
    tree_nodes = build_raptor_tree(all_leaf_docs, current_level=1, max_levels=2)
    full_corpus = all_leaf_docs + tree_nodes
    print(f"Total documents to index (Leaves + RAPTOR Summaries): {len(full_corpus)}")

    # Deterministic ID generation prevents duplicate keys
    for doc in full_corpus:
        content_hash = doc.page_content + str(doc.metadata.get("tree_level", 0)) + doc.metadata.get("source", "")
        doc.metadata["doc_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash))

    # Synchronize with PGVector via SQLRecordManager
    indexing_stats = index(
        docs_source=full_corpus,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )
    print("Indexing Complete! Status:", indexing_stats)



if __name__ == "__main__":
    run_ingestion()
'''
File Iteration & Pre-Check
   PDF_FILES ──► Check SQLRecordManager Ledger ──► If already indexed, skip file
                            │ (If new/modified)
                            ▼
2. Leaf Chunking (Level 0)
   PyPDFLoader ──► RecursiveCharacterTextSplitter ──► [Leaf Documents (tree_level=0)]
                            │
                            ▼
3. Recursive Tree Construction (build_raptor_tree)
   Embeddings (mxbai-embed-large) ──► Convert text chunks to numerical vectors
                            │
                            ▼
   Gaussian Mixture Model (GMM) ──► Group semantically close vectors into clusters
                            │
                            ▼
   ChatOllama (llama3.2) ──► Generate abstractive summary for each cluster (tree_level=1)
                            │
                            ▼
   Recursion Loop ────────► Cluster & summarize Level 1 nodes into Level 2 nodes
                            │
                            ▼
4. Unification & Synchronization
   Combine all nodes (Level 0 + Level 1 + Level 2)
   Generate Deterministic UUIDv5 IDs
   Run index() with SQLRecordManager ──► PGVector (PGVector syncs only new/modified vectors)





'''
