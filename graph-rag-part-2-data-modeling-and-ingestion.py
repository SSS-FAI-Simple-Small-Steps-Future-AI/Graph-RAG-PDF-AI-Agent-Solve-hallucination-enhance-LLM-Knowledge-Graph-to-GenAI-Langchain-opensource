## PDF-AI-Agent - RAG, LLAMA 3.1 via Ollama, LangChain, SBERT,
## GraphRAG: Neo4j
######################################
#Graph RAG Part-2
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
# Initialize the HuggingFace SentenceTransformer embeddings in LangChain
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the entities
entities = [
    {"label": "Condition", "name": "Diabetes", 'embedding_text': 'diabetes'},
    {"label": "Condition", "name": "Obesity",'embedding_text': 'obesity'},
    {"label": "Habit", "name": "Smoking", 'embedding_text': 'smoking'},
    {"label": "Condition", "name": "Prediabetes", 'embedding_text': 'prediabetes'},
    {"label": "Condition", "name": "Type-1 Diabetes", 'embedding_text': 'type 1 diabetes'},
    {"label": "Condition", "name": "Type-2 Diabetes", 'embedding_text': 'type 2 diabetes'}
]

# Generate the vector embeddings
# Update Entities
for entity in entities:
    v_embeddings = embeddings.embed_query(entity['embedding_text'])
    cypher = f"MATCH (n {{name: '{entity['name']}'}}) SET n.embeddings = {v_embeddings}"
    graph.query(cypher)


from langchain_community.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader("documents_data/")
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
# Split the documents into chunks
chunks = text_splitter.split_documents(docs)


# Add document chunks to database
#This code results in chunks being added as nodes labeled “Document”

prev_node_id = None  # Initialize prev_node_id before loop

for i, chunk in enumerate(chunks):
    # Create the chunk node
    query = f'''
    CREATE (d:Document {{chunkID: '{f"chunk_{i}"}', full_text: '{chunk.page_content}', embeddings: {embeddings.embed_query(chunk.page_content)}}})
    RETURN ID(d)
    '''
    result = graph.query(query)
    chunk_node_id = result[0]['ID(d)']
# If this is not the first chunk, create a NEXT relationship to the previous chunk
    if prev_node_id is not None:
        query = f'''
        MATCH (c1:Document), (c2:Document)
        WHERE ID(c1) = {prev_node_id} AND ID(c2) = {chunk_node_id}
        CREATE (c1)-[:NEXT]->(c2)
        CREATE (c2)-[:PREV]->(c1)
        '''
        graph.query(query)

    prev_node_id = chunk_node_id
