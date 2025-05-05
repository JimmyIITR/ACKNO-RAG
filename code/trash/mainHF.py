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
from langchain_core.runnables import RunnableLambda
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# ← NEW: use the Hugging Face integration instead of OllamaFunctions
from langchain_huggingface.llms import HuggingFacePipeline

from dataBase import queries
import prompts
import selectData
from dotenv import load_dotenv

load_dotenv()

DATA_PATH       = selectData.dataPath()
HF_MODEL_ID     = selectData.llmModel()        # e.g. "mistralai/Mistral-7B-Instruct"
EMBEDDINGS_MODEL= selectData.embeddingModel()

def loadData(dataPath):
    textLoader   = TextLoader(file_path=dataPath, autodetect_encoding=True)
    rawDocs      = textLoader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    return textSplitter.split_documents(documents=rawDocs)

def processLLM(docs):
    """
    Replace OllamaFunctions with a HuggingFacePipeline instance.
    """
    llm = HuggingFacePipeline.from_model_id(
        model_id=HF_MODEL_ID,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.0},
    )  # :contentReference[oaicite:0]{index=0}

    graphTransformer = LLMGraphTransformer(llm=llm)
    graphDocs = graphTransformer.convert_to_graph_documents(docs)
    return llm, graphDocs

def addToGraph(graph, graphDocs):
    graph.add_graph_documents(graphDocs, baseEntityLabel=True, include_source=True)

def initializeEmbeddings():
    # 1. Load a Sentence‑Transformer model
    embeddingModel = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},             # or "cuda" if available
        encode_kwargs={"normalize_embeddings": True} # optional normalization
    )

    # 2. Build the retriever with the HF embeddings
    vectorIndex = Neo4jVector.from_existing_graph(
        embedding=embeddingModel,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vectorIndex.as_retriever()

def retrieveContext(question, vectorRetriever, graph):
    """
    No more entityChain: we'll rely on plain-text or a custom JSON prompt.
    """
    # If you still need entity extraction, call llm.invoke() with a JSON prompt manually here.
    graphData     = queries.graphRetriever(question, None, graph)
    vectorResults = [doc.page_content for doc in vectorRetriever.invoke(question)]
    return f"""Graph Data:
{graphData}

Vector Data:
{"#Document ".join(vectorResults)}"""

def setupAnswerChain():
    """
    Build the chain using HuggingFacePipeline in place of OllamaFunctions.
    """
    graph           = queries.neo4j()
    llm, _          = processLLM([])  # we only need llm here
    vectorRetriever = initializeEmbeddings()

    # Use a simple prompt → llm → StrOutputParser
    promptTemplate = ChatPromptTemplate.from_template(prompts.template)

    return (
        {
            "context": RunnableLambda(
                lambda question: retrieveContext(
                    question=question,
                    vectorRetriever=vectorRetriever,
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
    print("Loading and processing data...")
    documents   = loadData(DATA_PATH)
    llmModel, graphDocs = processLLM(documents)

    graph  = queries.neo4j()
    driver = queries.driveOpen()
    try:
        queries.clearDataWithIndex(driver)
        print("Database cleaned successfully.")
    finally:
        queries.driveClose(driver)

    addToGraph(graph, graphDocs)
    print("Data added to graph.")

    driver = queries.driveOpen()
    try:
        queries.createIndex(driver)
        print("Index created successfully.")
    finally:
        queries.driveClose(driver)

    print("Data ingestion completed successfully!\n")

def queryInterface(answerChain):
    print("\nQuery system ready. Type 'exit' to quit.\n")
    while True:
        userInput = input("Enter your question: ").strip()
        if userInput.lower() in ('exit','quit'):
            break
        if not userInput:
            continue
        response = answerChain.invoke(userInput)
        print("\nResponse:")
        print(response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    if input("Initialize new data in graph? (yes/no): ").lower().strip() == 'yes':
        handleDataIngestion()
    qaChain = setupAnswerChain()
    queryInterface(qaChain)
    print("Session terminated.")
