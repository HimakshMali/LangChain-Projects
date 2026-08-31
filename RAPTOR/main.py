# the right name of the file should be query.py but i am too lazy to rename the file 



# the main thing that RAPTOR does is 
# it Clusters the similar top(or leaf ) chunks and generates the cluster sumaries using the LLM,
# and recursively re-cluster and summarizes those sumries to form a hierarchical tree

# query.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import get_vectorstore, RAG_LLM

# Connect to existing PGVector store
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    formatted = []
    for d in docs:
        level = d.metadata.get("tree_level", 0)
        source = d.metadata.get("source", "unknown")
        node_type = d.metadata.get("node_type", "chunk")
        header = f"[Source: {source} | Type: {node_type} | Level: {level}]"
        formatted.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)

template = """You are an intelligent knowledge assistant. Use the provided multi-level document context (which contains both specific document chunks and high-level abstract summaries) to answer the question.
If the context does not contain the answer, say that you do not know.

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | RAG_LLM
    | StrOutputParser()
)

def main():
    print("\nRAPTOR Query Interface Ready. (Type 'exit' to quit)\n" + "="*50)
    while True:
        query = input("\nEnter Question: ")
        if query.strip().lower() == "exit":
            break
        if not query.strip():
            continue
        
        response = rag_chain.invoke(query)
        print("\n--- LLM Response ---")
        print(response)

if __name__ == "__main__":
    main()