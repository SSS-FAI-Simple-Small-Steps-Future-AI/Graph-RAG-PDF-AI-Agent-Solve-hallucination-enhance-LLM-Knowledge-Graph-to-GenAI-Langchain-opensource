## PDF-AI-Agent - RAG, LLAMA 3.1 via Ollama, LangChain, SBERT,
## GraphRAG: Neo4j
######################################
#Graph RAG Part-3
######################################
### Developed By
### Furqan Khan
#### Email: furqan.cloud.dev@gmail.com


from langchain_community.graphs import Neo4jGraph
url = "bolt://localhost:7687"
database = "neo4j"
username = "neo4j"
password = "password"

graph = Neo4jGraph(url=url, database=database, username=username, password=password)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Semantic search for vector index
emb = embeddings.embed_query('type 2 diabetes')

cypher = f"""
CALL db.index.vector.queryNodes('vector_index_entity', 10, {emb})
YIELD node AS vectorNode, score as vectorScore
WITH vectorNode, vectorScore
ORDER BY vectorScore DESC
RETURN vectorNode.name AS label, vectorScore AS score
"""
result = graph.query(cypher)
print(result)

#############################################

##--------------DOCS Retrival -----------
cypher = f"""
CALL db.index.vector.queryNodes('vector_index_entity', 2, {emb})
YIELD node AS vectorNode, score as vectorScore
WITH vectorNode, vectorScore
MATCH (vectorNode)-[r]->(d:Document)
WITH DISTINCT d, gds.similarity.cosine({emb}, d.embeddings) AS cosineSimilarity
WHERE cosineSimilarity > 0.5
RETURN ID(d), left(d.full_text, 50)
ORDER BY cosineSimilarity DESC
LIMIT 10
"""
result = graph.query(cypher)
print("##############DOCS####################")
print(result)

