import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableLambda
from dataBase import queries
import prompts
from langchain_ollama import ChatOllama
import selectData
import splitClaim
import pandas as pd
import csv
from gammaValidation import gammaMain
from code.alphabetaValidation import abMain
import queryLog

from dotenv import load_dotenv

load_dotenv()

SBERT_DATA_PATH = selectData.sbertDataPath()
TFIDF_DATA_PATH = selectData.tfidfDataPath()
BM25_DATA_PATH = selectData.bm25DataPath()
LLM_MODEL = selectData.llmModel()
EMBEDDINGS_MODEL = selectData.embeddingModel()
RESULT_PATH = selectData.finalCSV() #csv file
DATA_PATH = selectData.getTrainAVeriTecData() #json file

def initializeEmbeddings():
    embeddingModel = OllamaEmbeddings(model=selectData.embeddingModel())
    vectorIndex = Neo4jVector.from_existing_graph(
        embedding=embeddingModel,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vectorIndex.as_retriever()

def retrieveContext(question, vectorRetriever, graphData):
    vectorResults = [doc.page_content for doc in vectorRetriever.invoke(question)]
    return f"""Graph Data:
{graphData}

Vector Data:
{"#Document ".join(vectorResults)}"""

def setupAnswerChain(graphData):
    graph = queries.neo4j()
    llm = ChatOllama(model=selectData.llmModel(), temperature=0)
    entityChain = llm.with_structured_output(prompts.Entities)
    vectorRetriever = initializeEmbeddings()
    promptTemplate = ChatPromptTemplate.from_template(prompts.template)

    return (
        {
            "context": RunnableLambda(
                lambda q: retrieveContext(q, vectorRetriever, graphData)
            ),
            "question": RunnablePassthrough()
        }
        | promptTemplate
        | llm
        | StrOutputParser()
    )

def main():
    df = pd.read_json(DATA_PATH)

    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(['index', 'claim', 'response'])

    for idx, row in df.iterrows():
        claim = row['claim']
        queryLog.log_entry(idx, "START claim", data=claim)

        atomicClaims = splitClaim.processParagraph(claim)
        for sc in atomicClaims:
            gammaMain.main(sc, idx)

        queryLog.log_entry(idx, "Files Generated", data=None, status="info")
        
        graphData = abMain.main(selectData.sbertDataPath(), idx)

        queryLog.log_entry(idx, "Graph generated", data=graphData, status="info")
        
        qaChain  = setupAnswerChain(graphData)
        response = qaChain.invoke(claim)

        with open(RESULT_PATH, 'a', newline='', encoding='utf-8') as outf:
            writer = csv.writer(outf)
            writer.writerow([idx, claim, response])

        queryLog.log_entry(idx, "SAVED result", data=response, status="info")

    print("All done, results written to", RESULT_PATH)

if __name__ == "__main__":
    main()