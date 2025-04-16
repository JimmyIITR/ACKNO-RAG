import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) #code path is set

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import prompts
import selectData
from dataBase import queries

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = selectData.dataPath()
LLM_MODEL = selectData.llmModel()
EMBEDDINGS_MODEL = selectData.embeddingModel()

def retrieveContext(question, entityChain, graph):
    graphData = queries.matchNodeRetriver(question, entityChain, graph)
    res = f"""Graph Data: {graphData}"""
    print(res)
    return res

def setupAnswerChain():
    graph = queries.neo4j()
    llm = OllamaFunctions(model=LLM_MODEL, temperature=0, format="json")
    entityChain = llm.with_structured_output(prompts.Entities)
    
    promptTemplate = ChatPromptTemplate.from_template(prompts.template)
    return (
        {
            "context": RunnableLambda(
                lambda question: retrieveContext(
                    question=question,
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
    qaChain = setupAnswerChain()
    queryInterface(qaChain)
    print("Session terminated.")