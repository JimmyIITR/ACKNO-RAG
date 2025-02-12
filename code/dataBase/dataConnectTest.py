from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()


URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def test_connection(tx):
    result = tx.run("RETURN 'Connection Successful' AS message")
    for record in result:
        print(record["message"])

with driver.session() as session:
    session.read_transaction(test_connection)

driver.close()
