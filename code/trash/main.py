from langchain_core.runnables import  RunnablePassthrough
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

from dotenv import load_dotenv

load_dotenv()

def dataLoader(dataPath):
    loader = TextLoader(file_path=dataPath,autodetect_encoding=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    documents = text_splitter.split_documents(documents=docs)
    return documents


def LLM(docs):
    llm = OllamaFunctions(model="llama3.1", temperature=0, format="json")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graphDoc = llm_transformer.convert_to_graph_documents(docs)
    return llm,graphDoc

def addToGraph(graph, graphDoc):
    graph.add_graph_documents(
        graphDoc,
        baseEntityLabel=True,
        include_source=True
    )
    return

def embeddings():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vecIndex = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vectorRetriver = vecIndex.as_retriever()
    return vectorRetriver

def fullRetriver(question: str, vectorRetriver, entityChain, graph):
    graphData = queries.graphRetriever(question, entityChain, graph)
    vectorData = [el.page_content for el in vectorRetriver.invoke(question)]
    finalData = f"""Graph data:
                    {graphData}
                    vector data:
                    {"#Document ". join(vectorData)}
                    """
    return finalData


dataPath = "/Users/jimmyaghera/Downloads/Thesis/ACKNO-RAG/data/dummytext.txt"
userQuery = "Who are Nonna Lucia and Giovanni Caruso?"
userQueryT1 = "Who is Nonna Lucia?"
userQueryF1 = "Who is Nonna Lucia? Did she teach anyone about restaurants or cooking?"

graph = queries.neo4j()
docs = dataLoader(dataPath=dataPath)
llm,graphDoc = LLM(docs)
print(graphDoc[0])
addToGraph(graph,graphDoc)
driver = queries.driveOpen()
try:
    queries.createIndex(driver=driver)
except:
    pass

queries.driveClose(driver=driver)
entityChain = llm.with_structured_output(prompts.Entities)
entityChain.invoke(userQuery)
print(queries.graphRetriever(userQuery, entityChain=entityChain, graph=graph))
vectorRetriver = embeddings()

prompt = ChatPromptTemplate.from_template(prompts.template)
contextInput = fullRetriver(question=userQueryF1, vectorRetriver=vectorRetriver, entityChain=entityChain, graph=graph)
print(contextInput)
print("\n\n\n")

chain = (
        {
        "context": RunnableLambda(lambda question: fullRetriver(
            question=question,
            vectorRetriver=vectorRetriver,
            entityChain=entityChain,
            graph=graph
        )),
        "question": RunnablePassthrough() 
        }
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke(userQueryF1)

print(answer)



