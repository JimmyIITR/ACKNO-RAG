
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from itertools import combinations
from typing import List, Dict

load_dotenv()

def neo4j():
    return Neo4jGraph(
        url=os.environ["NEO4J_URI_LOCAL"],
        username=os.environ["NEO4J_USERNAME_LOCAL"],
        password=os.environ["NEO4J_PASSWORD_LOCAL"]
    )


def driveOpen():
    driver = GraphDatabase.driver(uri = os.environ["NEO4J_URI_LOCAL"], 
                              auth = (os.environ["NEO4J_USERNAME_LOCAL"],os.environ["NEO4J_PASSWORD_LOCAL"]))
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

def setupGdsCentrality(graph, graph_name: str = "knowledgeGraph"):
    """
    Initial setup for Betweenness Centrality calculation
    Run this ONCE before using bridgeNodeConnector
    """
    setup_queries = [
        # Check GDS installation
        "CALL gds.list() YIELD libraryVersion RETURN libraryVersion",
        
        # Create graph projection
        f"""CALL gds.graph.project(
            '{graph_name}',
            '*',
            {{
                ALL: {{ 
                    type: '*', 
                    orientation: 'UNDIRECTED' 
                }}
            }}
        )""",
        
        # Calculate centrality scores
        f"""CALL gds.betweenness.stream('{graph_name}')
        YIELD nodeId, score
        MATCH (n)
        WHERE id(n) = nodeId
        SET n.centralityScore = score
        RETURN count(n) AS updatedNodes""",
        
        # Drop projection (optional)
        f"CALL gds.graph.drop('{graph_name}')"
    ]
    
    results = []
    for query in setup_queries:
        try:
            results.append(graph.query(query))
        except Exception as e:
            print(f"Setup error: {str(e)}")
            raise
    
    return results

def verifyCentrality(graph):
    verify_query = """
    MATCH (n)
    WHERE EXISTS(n.centralityScore)
    RETURN count(n) AS nodesWithScore
    """
    return graph.query(verify_query)

def bridgeNodeConnector(graph, centralityThreshold: float = 0.7, bridgeLimit: int = 3) -> str:
    """
    Connects isolated subgraphs using high-centrality bridge nodes
    """
    bridgeQuery = """
    // 1. Find disconnected node pairs with valid scores
    MATCH (a), (b)
    WHERE a <> b 
      AND NOT EXISTS((a)-[*1..3]-(b))
      AND EXISTS(a.centralityScore)
      AND EXISTS(b.centralityScore)
    
    // 2. Subquery with proper variable scope
    CALL {
      WITH a, b
      MATCH path = shortestPath((a)-[*..6]-(b))
      WHERE ALL(rel IN relationships(path) WHERE type(rel) <> 'MENTIONS'
      WITH path, 
           [n IN nodes(path)[1..-1] WHERE n.centralityScore >= $centralityThreshold] AS bridgeNodes
      UNWIND bridgeNodes AS bridgeNode
      RETURN bridgeNode, relationships(path) AS rels
      ORDER BY bridgeNode.centralityScore DESC
      LIMIT 1
    }
    
    // 3. Safe relationship formatting
    WITH a, b, bridgeNode, rels,
         CASE WHEN size(rels) > 0 THEN type(rels[0]) ELSE 'UNKNOWN' END AS rel1,
         CASE WHEN size(rels) > 1 THEN type(rels[-1]) ELSE 'UNKNOWN' END AS rel2
    
    RETURN DISTINCT
      COALESCE(a.name, '') + ' -[' + rel1 + ']-> ' +
      COALESCE(bridgeNode.name, '') + ' -[' + rel2 + ']-> ' +
      COALESCE(b.name, '') AS connection
    LIMIT $bridgeLimit
    """
    
    params = {
        "centralityThreshold": centralityThreshold,
        "bridgeLimit": bridgeLimit
    }
    
    try:
        result = graph.query(bridgeQuery, params)
        return '\n'.join([rec.get('connection', '') for rec in result])
    except Exception as e:
        return f"Bridge connection error: {str(e)}"


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


# def twoNodeConnection(en1: str, en2: str, combinedNodesName: List[str], graph) -> str:
#     entityNode: Dict[str, str] = {}
    
#     # Node ID lookup remains the same
#     for name in combinedNodesName:
#         node_query = """
#         CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:1})
#         YIELD node
#         RETURN node.id AS id
#         """
#         result = graph.query(node_query, {"query": escapeLuceneQuery(name)})
#         if result:
#             entityNode[name] = result[0]["id"]

#     if en1 not in entityNode or en2 not in entityNode:
#         return f"{en1} - R - {en2}"
    
#     id1, id2 = entityNode[en1], entityNode[en2]

#     # Modified query with degree check
#     response = graph.query(
#         """
#         MATCH (a {id: $id1}), (b {id: $id2})
#         WITH a, b, 
#             size([(a)-[r]-(c) WHERE type(r) <> 'MENTIONS' | r]) AS a_degree,
#             size([(b)-[r]-(d) WHERE type(r) <> 'MENTIONS' | r]) AS b_degree
#         WHERE a_degree > 3 OR b_degree > 3
#         MATCH (a)-[r]-(b)
#         WHERE type(r) <> 'MENTIONS'
#         RETURN DISTINCT a.id + ' - ' + type(r) + ' - ' + b.id AS output
#         """,
#         params={"id1": id1, "id2": id2}
#     )

#     return "\n".join(rec["output"] for rec in response) if response else ""
# def generate_full_text_query(input: str) -> str:
#     words = [el for el in remove_lucene_chars(input).split() if el]
#     if not words:
#         return ""
#     full_text_query = " AND ".join([f"{word}~2" for word in words])
#     print(f"Generated Query: {full_text_query}")
#     return full_text_query.strip()