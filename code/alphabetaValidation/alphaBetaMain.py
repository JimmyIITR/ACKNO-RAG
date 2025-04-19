import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) #code path is set

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
import selectData
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
import nltk
import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import nltk
import re
import pandas as pd
from collections import defaultdict
from gammaValidation.dataFetch import getCrossAndSelfURLsWithClaims, getKCrossAndSelfURLsWithClaims
from sentence_transformers import SentenceTransformer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

nltk.download('punkt')
load_dotenv()

DATA_PATH = selectData.dataPath()
LLM_MODEL = selectData.llmModel()
EMBEDDINGS_MODEL = selectData.embeddingModel()

def getSession():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetchArticleText(url, session):
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=' ')
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        else:
            return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def tokenize(text):
    return nltk.word_tokenize(text)


def main():
    llm = OllamaFunctions(model=LLM_MODEL, temperature=0, format="json")
    entityChain = llm.with_structured_output(prompts.Entities)
    session = getSession()
    d = getKCrossAndSelfURLsWithClaims(7,1) # experiment shows that 7 is the value for mixup and 1 is number of samples needed
    print(d)
    for data in d: #working as if condition for now
        
        print("Retrieving search results...")
        urlSelf = data["main_claim"]["fact_checking_article"]
        urlOther = []
        for i in range(0,7):
            urlOther.append(data["related_articles"][i]["fact_checking_article"])
       
        combinedDocumentForGraph = ""
        text = fetchArticleText(urlSelf,session)
        combinedDocumentForGraph += text
        print(text)
        trueEntities = entityChain.invoke(text)
        print(trueEntities.names)
        # trueEntNames = trueEntities.names
        # falseEntNames = []
        # time.sleep(2)
        # for url in urlOther:
        #     text = fetchArticleText(url,session)
        #     combinedDocumentForGraph += text
        #     falseEntities = entityChain.invoke(text)
        #     falseEntNames += falseEntities.names
        #     time.sleep(2)

        # print(f"\n\n\n Combined data corpus: ")
        # print(combinedDocumentForGraph)
        # print(f"\n\n\n True Entities Name: ")
        # print(trueEntNames)
        # print(f"\n\n\n False Entities Name: ")
        # print(falseEntNames)

        #have to segregate true and false now by first constructing map and then 

if __name__ == "__main__":
    main()


# def loadData(dataPath):
#     """Load and split documents from specified path"""
#     textLoader = TextLoader(file_path=dataPath, autodetect_encoding=True)
#     rawDocs = textLoader.load()
#     textSplitter = RecursiveCharacterTextSplitter(
#         chunk_size=250, 
#         chunk_overlap=24
#     )
#     return textSplitter.split_documents(documents=rawDocs)

# def processLLM(docs):
#     """Process documents to create LLM instance and graph documents"""
#     llm = OllamaFunctions(model=LLM_MODEL, temperature=0, format="json")
#     graphTransformer = LLMGraphTransformer(llm=llm)
#     graphDocs = graphTransformer.convert_to_graph_documents(docs)
#     return llm, graphDocs

# def addToGraph(graph, graphDocs):
#     """Add processed documents to Neo4j graph"""
#     graph.add_graph_documents(
#         graphDocs,
#         baseEntityLabel=True,
#         include_source=True
#     )

# def initializeEmbeddings():
#     """Initialize and return vector retriever"""
#     embeddingModel = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
#     vectorIndex = Neo4jVector.from_existing_graph(
#         embedding=embeddingModel,
#         search_type="hybrid",
#         node_label="Document",
#         text_node_properties=["text"],
#         embedding_node_property="embedding"
#     )
#     return vectorIndex.as_retriever()

# def retrieveContext(question, vectorRetriever, entityChain, graph):
#     """Retrieve combined context from graph and vector store"""
#     graphData = queries.graphRetriever(question, entityChain, graph)
#     vectorResults = [doc.page_content for doc in vectorRetriever.invoke(question)]
#     res = f"""Graph Data:
# {graphData}

# Vector Data:
# {"#Document ".join(vectorResults)}"""
#     print(res)
#     return res

# def setupAnswerChain():
#     """Set up and return the question answering chain"""
#     graph = queries.neo4j()
#     llm = OllamaFunctions(model=LLM_MODEL, temperature=0, format="json")
#     entityChain = llm.with_structured_output(prompts.Entities)
#     vectorRetriever = initializeEmbeddings()
    
#     promptTemplate = ChatPromptTemplate.from_template(prompts.template)
#     return (
#         {
#             "context": RunnableLambda(
#                 lambda question: retrieveContext(
#                     question=question,
#                     vectorRetriever=vectorRetriever,
#                     entityChain=entityChain,
#                     graph=graph
#                 )
#             ),
#             "question": RunnablePassthrough() 
#         }
#         | promptTemplate
#         | llm
#         | StrOutputParser()
#     )

# def handleDataIngestion():
#     """Handle the data loading and graph population process"""
#     print("Loading and processing data...")
#     documents = loadData(DATA_PATH)
#     llmModel, graphDocuments = processLLM(documents)
    
#     graph = queries.neo4j()
#     #try to clean whole data first
#     driver = queries.driveOpen()
#     try:
#         queries.clearDataWithIndex(driver)
#         print(f"Database Cleaned Successfuly.")
#     except Exception as e:
#          print(f"Data Clean error : {str(e)}")
#     finally:
#         queries.driveClose(driver)
#     #add data to database
#     addToGraph(graph, graphDocuments)
#     print("Data added to Graph")
#     #create index of the database
#     driver = queries.driveOpen()
#     try:
#         queries.createIndex(driver)
#         print("Indexing created successfully.")
#     except Exception as e:
#         print(f"Index creation skipped: {str(e)}")
#     finally:
#         queries.driveClose(driver)
#     print("Data ingestion completed successfully!\n")

# def queryInterface(answerChain):
#     """Handle user queries in a loop"""
#     print("\nQuery system ready. Type 'exit' to quit.\n")
#     while True:
#         userInput = input("Enter your question: ").strip()
#         if userInput.lower() in ('exit', 'quit'):
#             break
#         if not userInput:
#             continue
            
#         response = answerChain.invoke(userInput)
#         print("\nResponse:")
#         print(response)
#         print("\n" + "="*50 + "\n")

# if __name__ == "__main__":
#     initialChoice = input("Initialize new data in graph? (yes/no): ").lower().strip()
#     if initialChoice == 'yes':
#         handleDataIngestion()
    
#     qaChain = setupAnswerChain()
#     queryInterface(qaChain)
#     print("Session terminated.")