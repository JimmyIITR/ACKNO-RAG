
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from itertools import combinations

load_dotenv()

def neo4j():
    return Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"]
    )


def driveOpen():
    driver = GraphDatabase.driver(uri = os.environ["NEO4J_URI"], 
                              auth = (os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"]))
    return driver

def clearDatabase(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def dropFulltextIndex(tx):
    tx.run("DROP INDEX fulltext_entity_id IF EXISTS")

def createFulltextIndex(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

def clearDataWithIndex(driver):
    with driver.session() as session:
        session.execute_write(clearDatabase)
    with driver.session() as session:
        session.execute_write(dropFulltextIndex)

def createIndex(driver):
    with driver.session() as session:
        session.execute_write(createFulltextIndex)
        print("Fulltext index created successfully.")


# def createFulltextIndex(tx):
#     tx.run("MATCH (n) DETACH DELETE n")
#     tx.run("DROP INDEX fulltext_entity_id IF EXISTS")
#     query = '''
#     CREATE FULLTEXT INDEX `fulltext_entity_id` 
#     FOR (n:__Entity__) 
#     ON EACH [n.id];
#     '''
#     tx.run(query)

# # Function to execute the query
# def createIndex(driver):
#     with driver.session() as session:
#         session.execute_write(createFulltextIndex)
#         print("Fulltext index created successfully.")


def driveClose(driver):
    driver.close()

# def getEntities(question: str, entityChain):
#     """
#     Collects the neighborhood of entities mentioned
#     in the question
#     """
#     result = []
#     entities = entityChain.invoke(question)
#     for entity in entities.names:
#         result.append(entity)
#     return result

def graphRetriever(question: str, entityChain, graph) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entityChain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def matchNodeRetriver(question: str, entityChain, graph) -> str:
    result = ""
    
    entities = entityChain.invoke(question).names
    # print(entities)
    entity_nodes = {}
    for entity in entities:
        node_query = """
        CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:1})
        YIELD node
        RETURN node.id AS id
        """
        response = graph.query(node_query, {"query": entity})
        if response:
            entity_nodes[entity] = response[0]['id']

    for en1, en2 in combinations(entities, 2):
        if en1 in entity_nodes and en2 in entity_nodes:
            relationship_query = """
            MATCH (a)-[r:!MENTIONS]-(b)
            WHERE a.id = $id1 AND b.id = $id2
            RETURN DISTINCT a.id + ' - ' + type(r) + ' - ' + b.id AS output
            """
            response = graph.query(relationship_query, {"id1": entity_nodes[en1], "id2": entity_nodes[en2]})
            for rec in response:
                result += rec['output'] + "\n"
                
    return result


# def generate_full_text_query(input: str) -> str:
#     words = [el for el in remove_lucene_chars(input).split() if el]
#     if not words:
#         return ""
#     full_text_query = " AND ".join([f"{word}~2" for word in words])
#     print(f"Generated Query: {full_text_query}")
#     return full_text_query.strip()