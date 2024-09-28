## PDF-AI-Agent - RAG, LLAMA 3.1 via Ollama, LangChain, SBERT,
## GraphRAG: Neo4j
######################################
#Graph RAG Part-4
######################################
### Developed By
### Furqan ( Software Developer - AI/ML Solution Architect )
#### Email: furqan.cloud.dev@gmail.com

import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
# neo4j database credentials
url = "bolt://localhost:7687"
database = "neo4j"
username = "neo4j"
password = "password"

from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph(url=url, database=database, username=username, password=password)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

question = "What is the role of obesity in type 2 diabetes?"
emb = embeddings.embed_query(question)

cypher = f"""
CALL db.index.vector.queryNodes('vector_index_entity', 2, {emb})
YIELD node AS vectorNode, score as vectorScore
WITH vectorNode, vectorScore
MATCH (vectorNode)-[r]->(d:Document)
WITH DISTINCT d, gds.similarity.cosine({emb}, d.embeddings) AS cosineSimilarity
WHERE cosineSimilarity > 0.6
RETURN d.full_text
ORDER BY cosineSimilarity DESC
LIMIT 10
"""
query_result = graph.query(cypher)
# print(f"Number of documents: {len(query_result)}")
# print("##############DOCS####################")
# print(query_result)

from langchain.prompts import PromptTemplate
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

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

llm_type = os.getenv("LLM_TYPE", "ollama")
if llm_type == "ollama":
    llm = ChatOllama(model="llama3.1", temperature=0)
else:
    llm = ChatOpenAI(model="gpt-4", temperature=0)


from langchain_core.output_parsers import StrOutputParser
# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()
# Extract content from retrieved documents
doc_texts = "\\n".join([doc['d.full_text'] for doc in query_result])
# print(doc_texts)

# Get the answer from the language model
answer = rag_chain.invoke({"question": question, "documents": doc_texts})
print("Question:", question)
print("Answer:", answer)