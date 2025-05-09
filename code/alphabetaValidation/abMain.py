import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) #code path is set

import nltk
import json
from dataBase import queries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import TextLoader
from dataBase import queries
from langchain_ollama import ChatOllama
import json
from datetime import datetime
import selectData
from langchain.schema import Document
from selectData import tempFileFactText,tempFileFalseFactText,dataPath,llmModel,embeddingModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

FACT_DATA = tempFileFactText()
FALSE_FACT_DATA = tempFileFalseFactText()

DATA_PATH = dataPath()
LLM_MODEL = llmModel()
EMBEDDINGS_MODEL = embeddingModel()
sbertModel = SentenceTransformer('all-MiniLM-L6-v2')


nltk.download('punkt')

def loadData(dataPath):
    """Load and split documents from specified path"""
    textLoader = TextLoader(file_path=dataPath, autodetect_encoding=True)
    rawDocs = textLoader.load()
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=250, 
        chunk_overlap=24
    )
    return textSplitter.split_documents(documents=rawDocs)

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=4, max=10))
def processLLM(docs):
    """Process documents to create LLM instance and graph documents"""
    llm = ChatOllama(model=LLM_MODEL, temperature=0,request_timeout=300)
    graphTransformer = LLMGraphTransformer(llm=llm)
    graphDocs = graphTransformer.convert_to_graph_documents(docs)
    return llm, graphDocs

def findCLosestPair(aList, bList, model=sbertModel)  -> list[str]:
    a_embeddings = model.encode(aList, convert_to_tensor=True)
    b_embeddings = model.encode(bList, convert_to_tensor=True)
    cosine_scores = util.cos_sim(a_embeddings, b_embeddings)
    results = []
    for idx, a_item in enumerate(aList):
        best_match_idx = cosine_scores[idx].argmax()
        best_match = bList[best_match_idx]
        results.append(best_match)
    return results

def processLLMFromText(text: str):
    doc = Document(page_content=text)
    llm, graphDocs = processLLM([doc])
    return llm, graphDocs


def addToGraph(graph, graphDocs):
    """Add processed documents to Neo4j graph"""
    graph.add_graph_documents(
        graphDocs,
        baseEntityLabel=True,
        include_source=True
    )

def getNodesListIDs(graph_docs) -> list[str]:
    """Return a sorted list of all unique node IDs in the given GraphDocuments."""
    return sorted({ node.id for doc in graph_docs for node in doc.nodes })

def getPathsforAllEntPairs(graph, entitiesOfClaim, entities, index=0):
    graphData = ""
    for i, en1 in enumerate(entitiesOfClaim):
        for en2 in entitiesOfClaim[i+1:]:
            print(en1, en2)
            paths_str = queries.getTwoEntpaths(en1, en2, entities, graph)
            graphData = graphData + paths_str
    return graphData    


def log_entry(index, message, data=None, status="info"):
    """Helper function to log messages and data to a JSONL file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "index": index,
        "status": status,
        "message": message,
        "data": data
    }
    try:
        with open("data_ingestion_log.jsonl", "a") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Failed to write log entry: {str(e)}")

def handleDataIngestion(claim, PATH, index=0):
    log_entry(index, f"Loading and processing data for {index}")
    document = loadData(PATH)
    llmModel, graphDocuments = processLLMFromText(claim)
    entitesOfClaim = getNodesListIDs(graphDocuments)

    log_entry(index, f"BERT data completed for {index}")
    log_entry(index, "Nodes data", data=entitesOfClaim)

    llmModel, graphDocuments = processLLM(document)
    entites = getNodesListIDs(graphDocuments)
    
    entitiesSelectedByClaim = findCLosestPair(entitesOfClaim, entites, sbertModel)

    log_entry(index, f"BERT data completed for {index}")
    log_entry(index, "Nodes data", data=entites)

  
    graph = queries.neo4j()
    driver = queries.driveOpen()
    # entites = queries.getAllNodeId(graph) #temp
    try:
        queries.clearDataWithIndex(driver)
        log_entry(index, "Database Cleaned Successfully.")
    except Exception as e:
        log_entry(index, f"Data Clean error : {str(e)}", status="error")
    finally:
        queries.driveClose(driver)
    
    addToGraph(graph, graphDocuments)
    log_entry(index, f"Data added to Graph for {index}")
    
    driver = queries.driveOpen()
    try:
        queries.createIndex(driver)
        log_entry(index, "Indexing created successfully.")
    except Exception as e:
        log_entry(index, f"Index creation skipped: {str(e)}", status="error")
    finally:
        queries.driveClose(driver)
    
    log_entry(index, "Data ingestion completed successfully!")
    t1 = queries.graphSetup(graph)
    log_entry(index, f"Graph Setup for {index}", t1)
    t2 = queries.autoGraphConnector(graph)
    log_entry(index, f"Graph auto connector for {index}", t2)
    ans = getPathsforAllEntPairs(graph, entitiesSelectedByClaim, entites, index)
    return ans


def main(claim, PATH, index=0):
    graphData =  handleDataIngestion(claim, PATH, index)
    log_entry(index, f"Data retrived from graph for GraphData {index}", graphData)
    return graphData

if __name__ == "__main__":
    claim = "Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma."
    PATH = selectData.sbertDataPath()
    main(claim, PATH)