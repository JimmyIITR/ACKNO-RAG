from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
load_dotenv()


URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
parsed = urlparse(URI)


print("Host (instance) name:", parsed.hostname)

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def test_connection(tx):
    result = tx.run("RETURN 'Connection Successful' AS message")
    for record in result:
        print(record["message"])

with driver.session() as session:
    session.execute_read(test_connection)

driver.close()

# test_connection()