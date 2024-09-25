#!/bin/bash
#Setup Neo4j Docker Instance - Local

docker run --name neo4jDB \
  --restart always \
  --publish=7474:7474 --publish=7687:7687 \
  --env NEO4J_AUTH=neo4j/password \
  --env NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  --volume $HOME/neo4j/data:/data \
  --volume $HOME/neo4j/logs:/logs \
  --volume $HOME/neo4j/import:/var/lib/neo4j/import \
  --volume $HOME/neo4j/plugins:/plugins \
  neo4j:5.23.0