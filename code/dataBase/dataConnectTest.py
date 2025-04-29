from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
load_dotenv()

URI = os.getenv("NEO4J_URI_LOCAL")
USER = os.getenv("NEO4J_USERNAME_LOCAL")
PASSWORD = os.getenv("NEO4J_PASSWORD_LOCAL")

def driveOpen():
    driver = GraphDatabase.driver(uri = os.environ["NEO4J_URI_LOCAL"], 
                              auth = (os.environ["NEO4J_USERNAME_LOCAL"],os.environ["NEO4J_PASSWORD_LOCAL"]))
    return driver
driver = driveOpen()

def test_connection(tx):
    result = tx.run("RETURN 'Connection Successful' AS message")
    for record in result:
        print(record["message"])

with driver.session() as session:
    session.execute_read(test_connection)

def verify_gds():
    try:
        with driver.session() as session:
            result = session.run("CALL gds.list() YIELD name RETURN count(name)")
            return f"GDS procedures available: {result.single()[0]}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        driver.close()

print(verify_gds())

driver.close()

# test_connection()