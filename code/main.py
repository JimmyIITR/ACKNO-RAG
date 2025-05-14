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
import os
from gammaValidation import gammaMainBrave
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

NEO4J_URI = os.getenv("NEO4J_URI_LOCAL")
NEO4J_USER = os.getenv("NEO4J_USERNAME_LOCAL")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_LOCAL")


def initializeEmbeddings():
    embeddingModel = OllamaEmbeddings(model=selectData.embeddingModel())
    vectorIndex = Neo4jVector.from_existing_graph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        embedding=embeddingModel,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vectorIndex.as_retriever()

def summarize(file_path):
    try:
        with open(file_path, 'r') as file:
            original_text = file.read()

        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        summarize_chain = prompts.summerizePrompt | llm | StrOutputParser()
        
        summary = summarize_chain.invoke({"text": original_text})
        with open(file_path, 'w') as file:
            file.write(summary)
            
        return summary
    except Exception as e:
        queryLog.log_entry("SUMMARIZER", "SUMMARIZE_FAIL", data=str(e), status="error")
        return None

def retrieveContext(question, vectorRetriever, graphData=None):
    vectorResults = [doc.page_content for doc in vectorRetriever.invoke(question)]
    res = f"""Graph Data:
        {graphData}

        Vector Data:
        {"#Document ".join(vectorResults)}"""
    print(res)
    return res

def setupAnswerChain(graphData = None):
    llm = ChatOllama(model=selectData.llmModel(), temperature=0)
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

def remove_commas(text):
    return text.replace(",", "")

def main():
    try:
        df = pd.read_json(DATA_PATH)
        queryLog.log_entry("LOAD_DATA", "DATA_LOAD_SUCCESS", data={"rows": len(df)})
    except Exception as e:
        queryLog.log_entry("LOAD_DATA", "DATA_LOAD_FAIL", data=str(e), status="error")
        return

    try:
        with open(RESULT_PATH, 'a', newline='', encoding='utf-8') as outf:
            writer = csv.writer(outf)
            writer.writerow(['index', 'claim', 'response','label'])
        queryLog.log_entry("INIT_CSV", "CSV_INIT_SUCCESS", data={"path": RESULT_PATH})
    except Exception as e:
        queryLog.log_entry("INIT_CSV", "CSV_INIT_FAIL", data=str(e), status="error")
        return

    for idx, row in df.iloc[548:].iterrows():
        claim = row.get('claim', '')
        label = row.get('label', '')
        queryLog.log_entry(idx, "CLAIM_PROCESS_START", data=claim)

        try:
            atomicClaims = splitClaim.processParagraph(claim)
            for sc in atomicClaims:
                gammaMainBrave.main(sc, idx)
            queryLog.log_entry(idx, "GAMMA_VALIDATION_SUCCESS", data={"count": len(atomicClaims)})
        except Exception as e:
            queryLog.log_entry(idx, "GAMMA_VALIDATION_FAIL", data=str(e), status="error")

        try:
            summary = summarize(SBERT_DATA_PATH)
            if summary:
                queryLog.log_entry(idx, "SBERT_SUMMARY_SUCCESS", data={"summary": summary[:200] + "..."})
            else:
                queryLog.log_entry(idx, "SBERT_SUMMARY_EMPTY", status="warning")
        except Exception as e:
            queryLog.log_entry(idx, "SBERT_SUMMARY_FAIL", data=str(e), status="error")

        try:
            graphData = abMain.main(claim, SBERT_DATA_PATH, idx)
            queryLog.log_entry(idx, "GRAPH_BUILD_SUCCESS", data={"summary": graphData[:200] + "..."})
        except Exception as e:
            queryLog.log_entry(idx, "GRAPH_BUILD_FAIL", data=str(e), status="error")
            graphData = None

        response = ""
        if graphData is not None:
            try:
                qaChain = setupAnswerChain(graphData)
                response = qaChain.invoke(claim)
                queryLog.log_entry(idx, "QA_INVOKE_SUCCESS", data=response)
            except Exception as e:
                queryLog.log_entry(idx, "QA_INVOKE_FAIL", data=str(e), status="error")
                response = f"ERROR during QA: {e}"
        else:
            response = "SKIPPED QA due to graph build failure."
            queryLog.log_entry(idx, "QA_SKIPPED", data=response)

        try:
            with open(RESULT_PATH, 'a', newline='', encoding='utf-8') as outf:
                writer = csv.writer(outf)
                writer.writerow([idx, remove_commas(claim), remove_commas(response), label])
            queryLog.log_entry(idx, "CSV_APPEND_SUCCESS", data={"response": response})
        except Exception as e:
            queryLog.log_entry(idx, "CSV_APPEND_FAIL", data=str(e), status="error")

    queryLog.log_entry("MAIN_LOOP", "PROCESSING_COMPLETE", data={"output": RESULT_PATH})
    print("All done, results written to", RESULT_PATH)

if __name__ == "__main__":
    main()