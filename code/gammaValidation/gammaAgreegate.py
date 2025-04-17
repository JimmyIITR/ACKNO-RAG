import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import nltk
import re
import pandas as pd
from collections import defaultdict
from dataFetch import getCrossAndSelfURLsWithClaims, getTestDataCrossAndSelfURLsWithClaims
from BM25GammaValidation import compute_bm25_scores
from TFIDFGammaValidation import computeTfIdfScores
from SBERTGammaValidation import computeSbertScores
from sentence_transformers import SentenceTransformer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

nltk.download('punkt')

RESULT_PATH = "/Users/jimmyaghera/Downloads/Thesis/ACKNO-RAG/results/gamma"

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
    sbertModel = SentenceTransformer('all-MiniLM-L6-v2')  # (Consider all-mpnet-base-v2 in the future)
    session = getSession()  # Create a session with retries
    columns = []
    for i in range(2, 10):
        columns.extend([
            f'BM25_{i}_first',
            f'BM25_{i}_max',
            f'SBERT_{i}_first',
            f'SBERT_{i}_max',
            f'TFIDF_{i}_first',
            f'TFIDF_{i}_max'
        ])
    
    # Create DataFrame
    df = pd.DataFrame(columns=['Claim'] + columns)
    
    for i in range(8, 11):
        dataList = getCrossAndSelfURLsWithClaims(i)  # 5 related articles + 1 main = 6 total
        for data in dataList: #working as if condition
            claim = data["main_claim"]["text"]
            if claim not in df['Claim'].values:
                new_row = {'Claim': claim}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            rowIndex = df.index[df['Claim'] == claim].tolist()[0]
            
            urlSelf = data["main_claim"]["fact_checking_article"]
            urlOther = [data["related_articles"][j]["fact_checking_article"] for j in range(i)]
            
            documents = []
            textSelf = fetchArticleText(urlSelf, session)
            documents.append(textSelf)
            # Delay between requests to avoid rate limiting
            time.sleep(2)
            for url in urlOther:
                textOther = fetchArticleText(url, session)
                documents.append(textOther)
                time.sleep(2)
            
            scoresBM25 = compute_bm25_scores(claim, documents)
            scoresTFIDF = computeTfIdfScores(claim, documents)
            scoresSBERT = computeSbertScores(claim, documents, sbertModel)
            print("BM25->",scoresBM25, "\n TFIDF->",scoresTFIDF, "\n SBERT->",scoresSBERT, "\n")
            if len(scoresBM25) > 0:
                df.at[rowIndex, f'BM25_{i}_first'] = scoresBM25[0]
                df.at[rowIndex, f'BM25_{i}_max'] = max(scoresBM25[1:i+1])
            if len(scoresSBERT) > 0:
                df.at[rowIndex, f'SBERT_{i}_first'] = scoresSBERT[0]
                df.at[rowIndex, f'SBERT_{i}_max'] = max(scoresSBERT[1:i+1])
            if len(scoresTFIDF) > 0:
                df.at[rowIndex, f'TFIDF_{i}_first'] = scoresTFIDF[0]
                df.at[rowIndex, f'TFIDF_{i}_max'] = max(scoresTFIDF[1:i+1])
    
    df.to_excel(f"{RESULT_PATH}/validation_results(4).xlsx", index=False)
    print("Results saved to validation_results.xlsx")


def testMain():
    sbertModel = SentenceTransformer('all-MiniLM-L6-v2')  # (Consider all-mpnet-base-v2 in the future)
    session = getSession()  
    columns = []
    for i in range(7, 9):
        columns.extend([
            f'BM25_{i}_first',
            f'BM25_{i}_max',
            f'SBERT_{i}_first',
            f'SBERT_{i}_max',
            f'TFIDF_{i}_first',
            f'TFIDF_{i}_max'
        ])
    
    df = pd.DataFrame(columns=['Claim'] + columns)
    
    for i in range(7, 9):
        dataList = getTestDataCrossAndSelfURLsWithClaims(i)  # 5 related articles + 1 main = 6 total
        print(dataList)
        for data in dataList:
            claim = data["main_claim"]["text"]
            if claim not in df['Claim'].values:
                new_row = {'Claim': claim}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            rowIndex = df.index[df['Claim'] == claim].tolist()[0]
            
            urlSelf = data["main_claim"]["fact_checking_article"]
            urlOther = [data["related_articles"][j]["fact_checking_article"] for j in range(i)]
            
            documents = []
            textSelf = fetchArticleText(urlSelf, session)
            documents.append(textSelf)
            time.sleep(2)
            for url in urlOther:
                textOther = fetchArticleText(url, session)
                documents.append(textOther)
                time.sleep(2)
            
            scoresBM25 = compute_bm25_scores(claim, documents)
            scoresTFIDF = computeTfIdfScores(claim, documents)
            scoresSBERT = computeSbertScores(claim, documents, sbertModel)
            print("BM25->",scoresBM25, "\n TFIDF->",scoresTFIDF, "\n SBERT->",scoresSBERT, "\n")
            if len(scoresBM25) > 0:
                df.at[rowIndex, f'BM25_{i}_first'] = scoresBM25[0]
                df.at[rowIndex, f'BM25_{i}_max'] = max(scoresBM25[1:i+1])
            if len(scoresSBERT) > 0:
                df.at[rowIndex, f'SBERT_{i}_first'] = scoresSBERT[0]
                df.at[rowIndex, f'SBERT_{i}_max'] = max(scoresSBERT[1:i+1])
            if len(scoresTFIDF) > 0:
                df.at[rowIndex, f'TFIDF_{i}_first'] = scoresTFIDF[0]
                df.at[rowIndex, f'TFIDF_{i}_max'] = max(scoresTFIDF[1:i+1])
    
    df.to_excel(f"{RESULT_PATH}/testdata_results.xlsx", index=False)
    print("Results saved to testdata_results.xlsx")

if __name__ == "__main__":
    testMain() #yet to be done have to test and check which one performs best among all of them.
