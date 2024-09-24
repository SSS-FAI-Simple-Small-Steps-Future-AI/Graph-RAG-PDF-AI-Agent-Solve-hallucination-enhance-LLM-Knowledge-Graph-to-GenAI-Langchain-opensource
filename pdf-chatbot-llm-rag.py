## PDF-AI-Agent - RAG, LLAMA 3.1 via Ollama, LangChain, SBERT
######################################
### Developed By
### Furqan Khan
#### Email: furqan.cloud.dev@gmail.com


from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
os.environ['USER_AGENT'] = 'myagent'
# # Load PDF documents from directory
loader = PyPDFDirectoryLoader("documents_data/")
docs = loader.load()
print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
# Split the documents into chunks
docs_splits = text_splitter.split_documents(docs)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
# Create embeddings for documents and store them in a vector store
vectorstore = SKLearnVectorStore.from_documents(
    documents=docs_splits,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
retriever = vectorstore.as_retriever(k=5)

from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)

# Example usage
question = "What is the role of obesity in type 2 diabetes?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)
