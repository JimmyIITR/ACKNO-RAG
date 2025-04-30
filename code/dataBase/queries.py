
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
    
def autoGraphConnector(graph) -> str:
    """
    Automatically connects graph components using GDS with validated syntax
    """
    connector_query = """
    // 1. Verify GDS installation
    CALL gds.list() 
    YIELD name 
    WITH count(*) AS gdsAvailable
    WHERE gdsAvailable > 0
    
    // 2. Create temporary in-memory graph
    CALL gds.graph.project(
        'bridgingGraph',
        { 
            __Entity__: { 
                properties: 'id' 
            } 
        },
        { 
            REL: { 
                orientation: 'UNDIRECTED',
                properties: ['weight']
            },
            MENTIONS: { 
                orientation: 'NATURAL' 
            }
        }
    )
    
    // 3. Find weakly connected components
    CALL gds.wcc.stream('bridgingGraph')
    YIELD nodeId, componentId
    WITH componentId, collect(gds.util.asNode(nodeId)) AS componentNodes
    ORDER BY size(componentNodes) DESC
    
    // 4. Process components in pairs
    WITH collect(componentNodes) AS components
    UNWIND range(0, size(components)-2) AS i
    UNWIND range(i+1, size(components)-1) AS j
    WITH components[i] AS compA, components[j] AS compB
    
    // 5. Find optimal bridge paths
    CALL {
        WITH compA, compB
        UNWIND compA[0..5] AS a  // Limit component sampling
        UNWIND compB[0..5] AS b
        MATCH path = shortestPath((a)-[*..6]-(b))
        WHERE NONE(rel IN relationships(path) WHERE type(rel) = 'MENTIONS')
        RETURN nodes(path) AS bridgeNodes
        ORDER BY length(path)
        LIMIT 1
    }
    
    // 6. Create bridge relationships (fixed syntax)
    WITH bridgeNodes
    WHERE bridgeNodes IS NOT NULL AND size(bridgeNodes) > 1
    WITH bridgeNodes[0] AS a, bridgeNodes[-1] AS b
    MERGE (a)-[:BRIDGED]->(b)

    // // 7. Cleanup
    // CALL gds.graph.drop('bridgingGraph')
    // RETURN count(*) AS bridgesCreated
    
    """
    
    try:
        # Verify GDS installation first
        # graph.query("CALL gds.list() YIELD name LIMIT 1")
        
        # Execute the bridging process
        result = graph.query(connector_query)
        bridges = result[0]["bridgesCreated"] if result else 0
        
        return f"Created {bridges} strategic bridge connections"
        
    except Exception as e:
        return f"Connection error: {str(e)}"
    

def autoGraphConnector2(graph) -> str:
    """
    Automatically connects disconnected components in the graph using GDS.
    """
    try:
        # 1. Drop existing in-memory graph if it exists
        graph.query("CALL gds.graph.drop('bridgingGraph', false) YIELD graphName")

        # 2. Create in-memory graph using Cypher projection
        graph.query("""
        CALL gds.graph.project.cypher(
            'bridgingGraph',
            'MATCH (n:__Entity__) RETURN id(n) AS id',
            'MATCH (n:__Entity__)-[r]->(m:__Entity__) RETURN id(n) AS source, id(m) AS target, type(r) AS type'
        )
        """)

        # 3. Compute weakly connected components
        components = graph.query("""
        CALL gds.wcc.stream('bridgingGraph')
        YIELD nodeId, componentId
        RETURN gds.util.asNode(nodeId).id AS nodeId, componentId
        """)

        # 4. Identify disconnected components
        component_map = {}
        for record in components:
            comp_id = record['componentId']
            node_id = record['nodeId']
            component_map.setdefault(comp_id, []).append(node_id)

        if len(component_map) <= 1:
            return "Graph is already connected."

        # 5. Connect components by creating bridge relationships
        bridge_count = 0
        comp_ids = list(component_map.keys())
        for i in range(len(comp_ids)):
            for j in range(i + 1, len(comp_ids)):
                comp_a_nodes = component_map[comp_ids[i]]
                comp_b_nodes = component_map[comp_ids[j]]
                # Select representative nodes from each component
                node_a = comp_a_nodes[0]
                node_b = comp_b_nodes[0]
                # Create bridge relationship
                graph.query("""
                MATCH (a:__Entity__ {id: $id1}), (b:__Entity__ {id: $id2})
                MERGE (a)-[:BRIDGED]->(b)
                """, {"id1": node_a, "id2": node_b})
                bridge_count += 1

        # 6. Drop the in-memory graph
        graph.query("CALL gds.graph.drop('bridgingGraph') YIELD graphName")

        return f"Created {bridge_count} bridge connections to unify the graph."

    except Exception as e:
        return f"Connection error: {str(e)}"
