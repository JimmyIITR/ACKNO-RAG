from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
load_dotenv()

URI = os.getenv("NEO4J_URI_LOCAL")
USER = os.getenv("NEO4J_USERNAME_LOCAL")
PASSWORD = os.getenv("NEO4J_PASSWORD_LOCAL")


# URI = os.getenv("NEO4J_URI")
# USER = os.getenv("NEO4J_USERNAME")
# PASSWORD = os.getenv("NEO4J_PASSWORD")
# parsed = urlparse(URI)


# print("Host (instance) name:", parsed.hostname)

# driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# def test_connection(tx):
#     result = tx.run("RETURN 'Connection Successful' AS message")
#     for record in result:
#         print(record["message"])

# with driver.session() as session:
#     session.execute_read(test_connection)

# driver.close()

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "xcQAbjjhUJjRCDDhvhpv5mhvt4SyQcXnNYOHHmAWbyc"  # Replace with your actual password


def verify_gds():
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", PASSWORD)
    )
    try:
        with driver.session() as session:
            result = session.run("CALL gds.list() YIELD name RETURN count(name)")
            return f"GDS procedures available: {result.single()[0]}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        driver.close()

print(verify_gds())

# test_connection()