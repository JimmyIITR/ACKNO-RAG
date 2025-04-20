import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from dataBase import queries
import prompts
import articleExtrection
from selectData import tempFileFactText,tempFileFalseFactText,dataPath,llmModel,embeddingModel

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
    llm = OllamaFunctions(model=LLM_MODEL, temperature=0, format="json")
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


def handleDataIngestion():
    """Handle the data loading and graph population process"""
    print("Loading and processing data...")
    documents = loadData(FACT_DATA)
    llmModel, graphDocuments = processLLM(documents)
    
    graph = queries.neo4j()
    #try to clean whole data first
    driver = queries.driveOpen()
    try:
        queries.clearDataWithIndex(driver)
        print(f"Database Cleaned Successfuly.")
    except Exception as e:
         print(f"Data Clean error : {str(e)}")
    finally:
        queries.driveClose(driver)
    #add data to database
    addToGraph(graph, graphDocuments)
    print("Data added to Graph")
    #create index of the database
    driver = queries.driveOpen()
    try:
        queries.createIndex(driver)
        print("Indexing created successfully.")
    except Exception as e:
        print(f"Index creation skipped: {str(e)}")
    finally:
        queries.driveClose(driver)
    print("Data ingestion completed successfully!\n")



if __name__ == "__main__":
    handleDataIngestion()
    print("Session terminated.")