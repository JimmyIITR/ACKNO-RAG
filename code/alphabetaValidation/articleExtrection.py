import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) #code path is set

import selectData
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
import nltk
import re
import pandas as pd
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
    session = getSession()
    d = getKCrossAndSelfURLsWithClaims(2,1) # experiment shows that 7 is the value for mixup and 1 is number of samples needed
    #7 is the ideal as we can see from hyperparameter training of gaama but then prompt is getting bigger which is unable to handel by LLM so taking 2 only for now
    print(d)
    for data in d: #working as if condition for now
        
        print("Retrieving search results...")
        urlSelf = data["main_claim"]["fact_checking_article"]
        urlOther = []
        for i in range(0,2):
            urlOther.append(data["related_articles"][i]["fact_checking_article"])
       
        falseFactText = ""
        factText = fetchArticleText(urlSelf,session)
        with open(abspath(join(dirname(__file__), '../dataBase/temp/factText.txt')), "w", encoding="utf-8") as f:
            f.write(factText + f'\n')
        time.sleep(2)
        for url in urlOther:
            text = fetchArticleText(url,session)
            falseFactText += text + f'\n'
            time.sleep(2)
        with open(abspath(join(dirname(__file__), '../dataBase/temp/falseFactText.txt')), "w", encoding="utf-8") as f:
            f.write(falseFactText)
        print("Retrieving completed")

if __name__ == "__main__":
    main()
