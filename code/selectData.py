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