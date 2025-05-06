
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from itertools import combinations
from typing import List, Dict
import time

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

def getAllNodeId(graph) -> List[str]:
    response = graph.query(
        """
        MATCH (n)
        WHERE n.id IS NOT NULL
        RETURN n.id AS node_id
        """
    )
    return [record['node_id'] for record in response]

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

def getTwoEntpaths(en1: str, en2: str, combinedNodesName: List[str], graph) -> str:
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
    if en2 not in entityNode:
        return f"{en1} - R - {en2}"
    if en1 == en2:
        return f"{en1} - R - {en2}"
    
    id1, id2 = entityNode[en1], entityNode[en2]

    try:
        response = graph.query(
            """
            MATCH path = shortestPath((a)-[*..7]-(b))
            WHERE a.id = $id1 AND b.id = $id2
            WITH nodes(path) AS nodes, relationships(path) AS rels
            RETURN 
            REDUCE(
                s = CASE WHEN 'Document' IN LABELS(HEAD(nodes)) 
                        THEN 'Document' ELSE HEAD(nodes).id END,
                i IN RANGE(0, size(rels)-1) | 
                s + ' - ' + type(rels[i]) + ' - ' + 
                CASE WHEN 'Document' IN LABELS(nodes[i+1]) 
                    THEN 'Document' ELSE nodes[i+1].id END
            ) AS output
            """,
            params={"id1": id1, "id2": id2}
        )
        if not response:
            raise ValueError("no path found")
        return "\n".join(rec["output"] for rec in response)
    except Exception:
        return f"{id1} - R - {id2}"

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
    if en2 not in entityNode:
        return f"{en1} - R - {en2}"
    if en1 == en2:
        return f"{en1} - R - {en2}"
    
    id1, id2 = entityNode[en1], entityNode[en2]

    try:
        response = graph.query(
            """
            MATCH path = (a)-[rels*..7]-(b)
            WHERE a.id = $id1 AND b.id = $id2
            WITH nodes(path) AS nodes, rels
            RETURN 
            REDUCE(
                s = CASE WHEN 'Document' IN LABELS(HEAD(nodes)) 
                        THEN 'Document' ELSE HEAD(nodes).id END,
                i IN RANGE(0, size(rels)-1) | 
                s + ' - ' + type(rels[i]) + ' - ' + 
                CASE WHEN 'Document' IN LABELS(nodes[i+1]) 
                    THEN 'Document' ELSE nodes[i+1].id END
            ) AS output
            """,
            params={"id1": id1, "id2": id2}
        )
        if not response:
            raise ValueError("no path found")
        return "\n".join(rec["output"] for rec in response)
    except Exception:
        return f"{id1} - R - {id2}"

def graphSetup(graph) -> str:
    """
    Prepares the graph for bridge detection using native Cypher
    Aligns with your existing index patterns and id properties
    """
    setup_queries = [
        # Centrality calculation using relationship count
        ("""
        MATCH (n)
        SET n.centralityScore = SIZE([(n)-[]->() | 1])
        RETURN count(n) AS scoredNodes
        """, "Centrality calculation"),

        # Ensure ID property exists
        ("""
        MATCH (n) WHERE n.id IS NULL
        SET n.id = COALESCE(n.name, 'Node_' + elementId(n))
        RETURN count(n) AS idFixed
        """, "ID property setup"),

        # Index for bridge connections
        ("""CREATE INDEX IF NOT EXISTS FOR (n:__Entity__) ON (n.centralityScore)""", 
         "Centrality index")
    ]

    results = []
    for query, desc in setup_queries:
        result = graph.query(query)
        results.append(f"{desc}: {result}")
    return "Setup completed:\n" + "\n".join(results)

def bridgeNodeConnector(graph, centralityThreshold: float = 1.0, bridgeLimit: int = 5) -> str:
    """
    Creates bridge connections using your existing ID property
    and avoids MENTIONS relationships
    """
    bridgeQuery = """
    MATCH (a), (b)
    WHERE a.id <> b.id
      AND NOT (a)-[*1..3]-(b)
      AND a.centralityScore >= $ct
      AND b.centralityScore >= $ct
    CALL {
      WITH a, b
      OPTIONAL MATCH path = shortestPath((a)-[*..6]-(b))
      WHERE ALL(rel IN relationships(path) WHERE type(rel) <> 'MENTIONS')
      WITH [n IN nodes(path)[1..-1] | n] AS candidates
      UNWIND candidates AS candidate
      RETURN candidate ORDER BY candidate.centralityScore DESC LIMIT 1
    }
    WITH a, b, candidate
    WHERE candidate IS NOT NULL
    MERGE (a)-[r:BRIDGED]->(b)
    SET r.bridgeNode = candidate.id,
        r.score = candidate.centralityScore
    RETURN a.id + ' -[:BRIDGED]-> ' + b.id + ' via ' + candidate.id AS connection
    LIMIT $limit
    """
    
    try:
        # Validate setup using your existing patterns
        valid = graph.query("""
            MATCH (n) 
            RETURN 
                n.id IS NOT NULL AS hasId,
                n.centralityScore IS NOT NULL AS hasScore
            LIMIT 1
            """)[0]
        
        if not valid["hasId"] or not valid["hasScore"]:
            return "Run graphSetup() first!"

        result = graph.query(bridgeQuery, {
            "ct": centralityThreshold,
            "limit": bridgeLimit
        })
        
        return "Connections:\n" + '\n'.join([rec['connection'] for rec in result]) if result else "No bridges found"
        
    except Exception as e:
        return f"Connection error: {str(e)}"
    
def autoGraphConnector(graph, relationship_type: str = 'BRIDGED') -> str:
    """
    Connects all graph components by projecting with GDS, streaming WCC,
    and merging BRIDGED relationships between representative nodes.
    """

    # 1. Drop and re-project using gds.graph.project
    graph.query("CALL gds.graph.drop('entityGraph', false)")
    graph.query("""
    CALL gds.graph.project(
      'entityGraph',
      ['__Entity__'],
      {
        MENTIONS: {
          type: 'MENTIONS',
          orientation: 'UNDIRECTED'
        }
      }
    )
    """)

    # 2. Stream weakly connected components
    components = graph.query("""
    CALL gds.wcc.stream('entityGraph')
    YIELD nodeId, componentId
    WITH componentId, collect(gds.util.asNode(nodeId).id) AS members
    RETURN componentId, members
    """)

    # 3. If already connected, exit early
    if len(components) <= 1:
        return "Graph is already fully connected."

    # 4. Choose the first member of each component as representative
    reps = [comp['members'][0] for comp in components]
    created_edges = []

    # 5. Merge bridging relationships between consecutive reps
    for i in range(len(reps) - 1):
        id1, id2 = reps[i], reps[i+1]
        result = graph.query(f"""
        MATCH (a {{id: $id1}}), (b {{id: $id2}})
        MERGE (a)-[r:{relationship_type}]->(b)
        RETURN a.id AS from, b.id AS to
        """, {'id1': id1, 'id2': id2})
        created_edges.append(f"{result[0]['from']} -[:{relationship_type}]-> {result[0]['to']}")

    return "autoGraphConnector created connections:\n" + "\n".join(created_edges)
