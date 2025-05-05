import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

def llmModel():
    return "gemma3:4b" #mistral-small3.1

def embeddingModel():
    return "mxbai-embed-large"

def dataPath():
    return abspath(join(dirname(__file__), '../data/dummytext.txt')) # "/Users/jimmyaghera/Downloads/Thesis/ACKNO-RAG/data/dummytext.txt"

def tempFileFactText():
    return abspath(join(dirname(__file__), './dataBase/temp/factText.txt'))

def tempFileFalseFactText():
    return abspath(join(dirname(__file__), './dataBase/temp/falseFactText.txt'))

def resultsAlphaBeta(i=0):
    return abspath(join(dirname(__file__), f'../results/alphaBeta/results{i}.json'))

def resultGamma(i=0):
    return abspath(join(dirname(__file__), f'../results/gamma/results{i}.csv'))

def getTestAVeriTecData():
    return abspath(join(dirname(__file__), f'../data/AVeriTecData/dataTest.json'))

def getTrainAVeriTecData():
    return abspath(join(dirname(__file__), f'../data/AVeriTecData/dataTrain.json'))

def getDevAVeriTecData():
    return abspath(join(dirname(__file__), f'../data/AVeriTecData/dataDev.json'))

def dataInLogs():
    return abspath(join(dirname(__file__), f'../results/data_ingestion_log.jsonl')) #data_ingestion_log.jsonl

def dataInLogsMain():
    return abspath(join(dirname(__file__), f'../data_ingestion_log.jsonl')) #data_ingestion_log.jsonl

def getStrData():
    return abspath(join(dirname(__file__), f'../results/stringData.csv')) #data_ingestion_log.jsonl

def sbertDataPath():
    return abspath(join(dirname(__file__), f'./dataBase/temp/gammaExtrected/SBERT.txt')) #data_ingestion_log.jsonl

def tfidfDataPath():
    return abspath(join(dirname(__file__), f'./dataBase/temp/gammaExtrected/TFIDF.txt')) #data_ingestion_log.jsonl

def bm25DataPath():
    return abspath(join(dirname(__file__), f'./dataBase/temp/gammaExtrected/BM25.txt')) #data_ingestion_log.jsonl

def finalCSV():
    return abspath(join(dirname(__file__), f'../results/finalCSV.csv')) 
