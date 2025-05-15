from neo4j import GraphDatabase
import os

class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USER"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

    def search_nodes(self, keyword):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n.name CONTAINS $kw RETURN n LIMIT 50",
                kw=keyword
            )
            return [dict(record["n"]) for record in result]
