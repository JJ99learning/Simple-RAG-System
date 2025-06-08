from embedder import JinaEmbedder
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import os
import dotenv

dotenv.load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
embedder = JinaEmbedder(api_key=os.getenv("JINA_API_KEY"))


chroma_client = Chroma(
    embedding_function=embedder,
    persist_directory="chromaDB"
)

#load with encoding utf-8 or else it will throw an error while laoding the cat facts txt
# loader = TextLoader('data/cat-facts.txt', encoding='utf-8')
loader = PyPDFLoader('data/User Manual_Acer_1.0_A_A.pdf')
documents = loader.load()

# Turn the text into several chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
chroma_client.add_documents(chunks)

# Create the RAG prompt template
template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the information provided."""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": chroma_client.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# main loop 
while True:
    query = input("\nEnter your question about the document: ")

    try:
        response = rag_chain.invoke(query)
        print("\nAnswer:", response.content)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")




