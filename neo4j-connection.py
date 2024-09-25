# Neo4j setup & connection
# Docker Command for local container instance:
# run shell script: neo4j-setup.sh OR just copy the docker command and run in terminal

import sys
import os
os.environ["NEO4J_URL"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

# Alternatively if using .env file for environment variables
# from dotenv import load_dotenv
# load_dotenv()

neo4j_url = os.environ["NEO4J_URL"]
neo4j_username = os.environ["NEO4J_USERNAME"]
neo4j_password = os.environ["NEO4J_PASSWORD"]
neo4j_database = "neo4j"

from langchain_community.graphs import Neo4jGraph
graph: Neo4jGraph
try:
    graph = Neo4jGraph(url=neo4j_url, database=neo4j_database, username=neo4j_username, password=neo4j_password)
except BaseException as error:
    print('An exception occurred: {}'.format(error))
    sys.exit()

cypher = """
CALL dbms.components() YIELD name, edition, versions
RETURN name, edition, versions
"""
result = graph.query(cypher)
print(result)

# [{'name': 'Neo4j Kernel', 'edition': 'community', 'versions': ['5.23.0']}]
