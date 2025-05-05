import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import TextLoader
from dataBase import queries
from langchain_ollama import ChatOllama
import json
from datetime import datetime
from selectData import tempFileFactText,tempFileFalseFactText,dataPath,llmModel,embeddingModel
from abMain import handleDataIngestion
from dotenv import load_dotenv

load_dotenv()

FACT_DATA = tempFileFactText()
FALSE_FACT_DATA = tempFileFalseFactText()

DATA_PATH = dataPath()
LLM_MODEL = llmModel()
EMBEDDINGS_MODEL = embeddingModel()

def loadData(dataPath):
    """Load and split documents from specified path"""
    textLoader = TextLoader(file_path=dataPath, autodetect_encoding=True)
    rawDocs = textLoader.load()
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=250, 
        chunk_overlap=24
    )
    return textSplitter.split_documents(documents=rawDocs)

def processLLM(docs):
    """Process documents to create LLM instance and graph documents"""
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    graphTransformer = LLMGraphTransformer(llm=llm)
    graphDocs = graphTransformer.convert_to_graph_documents(docs)
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


def extract_counts(paths_str):
    paths = [p for p in paths_str.split("\n") if p.strip()]
    counts = []
    for p in paths:
        parts = p.split(" - ")
        total_nodes = (len(parts) + 1) // 2 # for relation removalas rel will be on odd places
        intermediate = max(0, total_nodes - 2)
        counts.append(intermediate)
    return counts, len(paths)

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


def dataGeneratorForLogistic(graph, result, index=0):
    """ result = {true: [], false: [], combined: []} """
    #below lines are incomplete
    true_nodes = result["true"]
    false_nodes = result["false"]
    combined = result["combined"]

    tt_horiz, tt_vert = [], []
    tf_horiz, tf_vert = [], []

    # 1 True–True
    for i, en1 in enumerate(true_nodes):
        for en2 in true_nodes[i+1:]:
            print(en1, en2)
            paths_str = queries.twoNodeConnection(en1, en2, combined, graph)
            # log_entry(index, f"{en1}-{en2}", data=paths_str)
            counts, num_paths = extract_counts(paths_str)
            tt_horiz.extend(counts)
            tt_vert.append(num_paths)

    # 2 True–False
    for en1 in true_nodes:
        for en2 in false_nodes:
            print(en1, en2)
            paths_str = queries.twoNodeConnection(en1, en2, combined, graph)
            # log_entry(index, f"{en1}-{en2}", data=paths_str)
            counts, num_paths = extract_counts(paths_str)
            tf_horiz.extend(counts)
            tf_vert.append(num_paths)

    return {
        "true_true_horizontal":  tt_horiz,
        "true_true_vertical":    tt_vert,
        "true_false_horizontal": tf_horiz,
        "true_false_vertical":   tf_vert,
    }

def handleDataIngestion(index=1):
    log_entry(index, f"Loading and processing data for {index}")
    trueDocuements = loadData(FACT_DATA)
    llmModel, graphDocuments = processLLM(trueDocuements)
    trueNodes = getNodesListIDs(graphDocuments)
    log_entry(index, f"FACT data completed for {index}")
    log_entry(index, "True Nodes data", data=trueNodes)
    
    falseDocuments = loadData(FALSE_FACT_DATA)
    llmModel, graphDocuments = processLLM(falseDocuments)
    falseNodes = getNodesListIDs(graphDocuments)
    log_entry(index, f"FALSE data completed for {index}")
    log_entry(index, "False Nodes data", data=falseNodes)

    combinedData = trueDocuements + falseDocuments
    llmModel, graphDocuments = processLLM(combinedData)
    log_entry(index, f"Combined data completed for {index}")
    combinedNodes = getNodesListIDs(graphDocuments)
    log_entry(index, "Combined Nodes data (temp)", data=combinedNodes)
    
    result = {
        "true": trueNodes,
        "false": falseNodes,
        "combined": combinedNodes
    }
    graph = queries.neo4j()
    driver = queries.driveOpen()
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
    res = dataGeneratorForLogistic(graph, result, index)
    log_entry(index, f"Response generated for {index}", res)
    return res


if __name__ == "__main__":
    # x = handleDataIngestion()
    # print(x)
    graph = queries.neo4j()
    driver = queries.driveOpen()
    # ans = queries.graphSetup(graph)
    # print(ans)
    # print("----------------\n setupDone")
    # bridges = queries.autoGraphConnector(graph)
    # print(bridges)
    # print("----------------\n Bridge connected")
    result_small = {
        "true": [ 'Nasa','Catholic Council','Nasa'],
        "false": ['Isi Chief','India Today','Ljp'],
        "combined":['Nasa','Catholic Council','Isi Chief','India Today','Ljp']
    }
    x = dataGeneratorForLogistic(graph, result_small)
    print(x)
    print("----------------")
    # asn = handleDataIngestion()
    # print(asn)
    print("Session terminated.")