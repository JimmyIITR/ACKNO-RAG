from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver
from langchain_ollama import ChatOllama


from dotenv import load_dotenv

load_dotenv()

def neo4j():
    graph = Neo4jGraph()
    return graph


def driveOpen():
    driver = GraphDatabase.driver(uri = os.environ["NEO4J_URI"], 
                              auth = (os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"]))
    return driver

def createFulltextIndex(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

# Function to execute the query
def createIndex(driver):
    with driver.session() as session:
        session.execute_write(createFulltextIndex)
        print("Fulltext index created successfully.")


def driveClose(driver):
    driver.close()

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


# def generate_full_text_query(input: str) -> str:
#     words = [el for el in remove_lucene_chars(input).split() if el]
#     if not words:
#         return ""
#     full_text_query = " AND ".join([f"{word}~2" for word in words])
#     print(f"Generated Query: {full_text_query}")
#     return full_text_query.strip()