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

load_dotenv()

DATA_PATH = selectData.dataPath()
LLM_MODEL = selectData.llmModel()
EMBEDDINGS_MODEL = selectData.embeddingModel()

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

def initializeEmbeddings():
    """Initialize and return vector retriever"""
    embeddingModel = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    vectorIndex = Neo4jVector.from_existing_graph(
        embedding=embeddingModel,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vectorIndex.as_retriever()

def retrieveContext(question, vectorRetriever, entityChain, graph):
    """Retrieve combined context from graph and vector store"""
    graphData = queries.graphRetriever(question, entityChain, graph)
    vectorResults = [doc.page_content for doc in vectorRetriever.invoke(question)]
    res = f"""Graph Data:
{graphData}

Vector Data:
{"#Document ".join(vectorResults)}"""
    print(res)
    return res

def setupAnswerChain():
    """Set up and return the question answering chain"""
    graph = queries.neo4j()
    llm = OllamaFunctions(model=LLM_MODEL, temperature=0, format="json")
    entityChain = llm.with_structured_output(prompts.Entities)
    vectorRetriever = initializeEmbeddings()
    
    promptTemplate = ChatPromptTemplate.from_template(prompts.template)
    return (
        {
            "context": RunnableLambda(
                lambda question: retrieveContext(
                    question=question,
                    vectorRetriever=vectorRetriever,
                    entityChain=entityChain,
                    graph=graph
                )
            ),
            "question": RunnablePassthrough() 
        }
        | promptTemplate
        | llm
        | StrOutputParser()
    )

def handleDataIngestion():
    """Handle the data loading and graph population process"""
    print("Loading and processing data...")
    documents = loadData(DATA_PATH)
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

def queryInterface(answerChain):
    """Handle user queries in a loop"""
    print("\nQuery system ready. Type 'exit' to quit.\n")
    while True:
        userInput = input("Enter your question: ").strip()
        if userInput.lower() in ('exit', 'quit'):
            break
        if not userInput:
            continue
            
        response = answerChain.invoke(userInput)
        print("\nResponse:")
        print(response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    initialChoice = input("Initialize new data in graph? (yes/no): ").lower().strip()
    if initialChoice == 'yes':
        handleDataIngestion()
    
    qaChain = setupAnswerChain()
    queryInterface(qaChain)
    print("Session terminated.")