# this is the code for a very simple rag application.
# its a simple flow of how the data (test.pdf) is converted into is splitted into chunks
# embeded
# and then store into the vector database
# and how we use it to ask questions about the pdf data

# ive wrote the detailed comments in this file about the code
# there is a requrirement.txt in this folder(simple-rag-app) to download all the necessaryy dependencies

# to download and run all the things please refer to readme file


from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaLLM

# for retriever and llm query 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# to load the doc
loader = PyPDFLoader("./test.pdf")
docs = loader.load()
# splitting the doc into chunks (1000 of size, with overlapping of 200)
test_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = test_splitter.split_documents(docs)
# emdinngs it and storing it in the vector database
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "pdf_collection"

vector_store = PGVector.from_documents(

    documents = split_docs,
    embedding = embeddings_model,
    connection=connection,
    collection_name=collection_name,
    use_jsonb=True,



)

# retriever to fetch relevant chunks based on user queries
# here k:2 means the retriever fetches the two chunks of data using mathematical probablietes functions to fetch chunks most relevant to the context
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# this function converts the postgresql converted data which is like [Document(...), Document(...)] into human redable formates
# as llm can read python objects and can only process the human redable texts
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = OllamaLLM(model="llama3.2", temperature=0.3)


# here we are adding 'context' and 'question' keys in the template and passing it to the prompt template
# we are adding {context} here because the llama3.2 is the general purpose models
# if we didnt pass the context or the datachunks , it will use the publically train data to give answer or it will just hallicunate
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


while True:
    query = input("Enter your question: ")
    response = rag_chain.invoke(query)
    print("\n--- Llama 3.2 Answer ---")
    print(response)




'''

the way this thing is working is that to use the documents and add it into the vector database and use llm to ask question
1.we first parse the document using PyPDFLoader , decodes it page by page 
2. Chunk the document into smaller pieces using RecursiveCharacterTextSplitter
3. Create embeddings for each chunk using OllamaEmbeddings
4. Store the embeddings in a PostgreSQL database using PGVector
5. Create a retriever to fetch relevant chunks based on user queries
6. Format the retrieved chunks into a single string to provide context for the LLM(using formate_docs())

 use this code to see if the documents are added into the vector database or not
 run it in query tool of postgresql and check the data in the table langchain_pg_embedding


 SELECT 
    id,
    document,
    cmetadata,
    embedding
FROM langchain_pg_embedding
LIMIT 20;
'''


'''
# somthing geimni made, dont know what it means

                     ──> retriever ──> format_docs ──> context ──┐
query (Input String)│                                            ├──> prompt ──> llm ──> StrOutputParser() ──> Final String
                    └──> RunnablePassthrough ────────> question ─┘

'''