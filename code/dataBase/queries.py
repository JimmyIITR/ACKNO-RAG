
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from itertools import combinations
from typing import List, Dict

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


def driveClose(driver):
    driver.close()

def getEntities(question: str, entityChain) -> list:
    entities = entityChain.invoke(question)
    return list(entities.names)

def escapeLuceneQuery(query: str) -> str:
    """Escape special characters for Lucene queries"""
    specialChars = r'\+-&|!(){}[]^"~*?:\\/'
    return ''.join(['\\' + char if char in specialChars else char for char in query])

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

# def matchNodeRetriver(en1: str, en2: str, combinedNodesName, graph) -> str:
#     result = ""
#     # make mapping of node and name 
#     entity_nodes = {}
#     for entity in combinedNodesName:
#         node_query = """
#         CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:1})
#         YIELD node
#         RETURN node.id AS id
#         """
#         response = graph.query(node_query, {"query": entity})
#         if response:
#             entity_nodes[entity] = response[0]['id']
#     # just a verificaiotin before calling 
#     try:
#         id1 = entity_nodes[en1]
#         id2 = entity_nodes[en2]
#     except KeyError as e:
#         raise ValueError(f"Entity {e.args[0]!r} not found in entity_nodes") from None
#     # final call for database
#     relationship_query = """
#     MATCH (a)-[r:!MENTIONS]-(b)
#     WHERE a.id = $id1 AND b.id = $id2
#     RETURN DISTINCT a.id + ' - ' + type(r) + ' - ' + b.id AS output
#     """
#     response = graph.query(relationship_query, {"id1": id1, "id2": id2})
#     for rec in response:
#         result += rec['output'] + "\n"
                
#     return result
# def twoNodeConnection(en1: str, en2: str, combinedNodesName: List[str], graph ) -> str:
#     entityNode: Dict[str, str] = {}
#     for name in combinedNodesName:
#         result = graph.run(
#             """
#             CALL db.index.fulltext.queryNodes($indexName, $query, {limit:1})
#             YIELD node
#             RETURN node.id AS id
#             """,
#             indexName="fulltext_entity_id",
#             query=name
#         )
#         record = result.single()
#         if record:
#             entityNode[name] = record["id"]

#     if en1 not in entityNode:
#         raise ValueError(f"Entity {en1!r} not found in index")
#     if en2 not in entityNode:
#         raise ValueError(f"Entity {en2!r} not found in index")
#     id1, id2 = entityNode[en1], entityNode[en2]

#     response = graph.run(
#         """
#         MATCH (a)-[r]-(b)
#         WHERE a.id = $id1 AND b.id = $id2
#           AND type(r) <> 'MENTIONS'
#         RETURN DISTINCT a.id + ' - ' + type(r) + ' - ' + b.id AS output
#         """,
#         id1=id1,
#         id2=id2
#     )

#     return "\n".join(rec["output"] for rec in response)


def twoNodeConnection(en1: str, en2: str, combinedNodesName: List[str], graph) -> str:
    entityNode: Dict[str, str] = {}
    
    for name in combinedNodesName:
        node_query = """
        CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:1})
        YIELD node
        RETURN node.id AS id
        """
        result = graph.query(node_query, {"query": escapeLuceneQuery(name)})
        if result:
            entityNode[name] = result[0]["id"]

    if en1 not in entityNode:
        return f"{en1} - R - {en2}"
        # raise ValueError(f"Entity {en1!r} not found in index")
    if en2 not in entityNode:
        return f"{en1} - R - {en2}"
        # raise ValueError(f"Entity {en2!r} not found in index")
    
    id1, id2 = entityNode[en1], entityNode[en2]

    response = graph.query(
        """
        MATCH (a)-[r]-(b)
        WHERE a.id = $id1 AND b.id = $id2
          AND type(r) <> 'MENTIONS'
        RETURN DISTINCT a.id + ' - ' + type(r) + ' - ' + b.id AS output
        """,
        params={"id1": id1, "id2": id2}
    )

    return "\n".join(rec["output"] for rec in response)
# def generate_full_text_query(input: str) -> str:
#     words = [el for el in remove_lucene_chars(input).split() if el]
#     if not words:
#         return ""
#     full_text_query = " AND ".join([f"{word}~2" for word in words])
#     print(f"Generated Query: {full_text_query}")
#     return full_text_query.strip()