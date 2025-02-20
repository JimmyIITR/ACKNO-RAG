from BM25GammaValidation import compute_bm25_scores
from SBERTGammaValidation import computeSbertScores
from TFIDFGammaValidation import computeTfIdfScores
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import nltk
import re
import pandas as pd
from collections import defaultdict
from dataFetch import getCrossAndSelfURLsWithClaims
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
results = defaultdict(dict)
def fetchArticleText(url):
    try:
        response = requests.get(url, timeout=10)
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
    sbertModel = SentenceTransformer('all-MiniLM-L6-v2') #user all-mpnet-base-v2 insted infuture
    columns = []
    for i in range(1, 10):
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
    for i in range(1,10):
        dataList = getCrossAndSelfURLsWithClaims(i)  # 5 related articles + 1 main = 6 total
        for data in dataList:
            claim = data["main_claim"]["text"]
            if claim not in df['Claim'].values:
                new_row = {'Claim': claim}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            row_index = df.index[df['Claim'] == claim].tolist()[0]
            urlSelf = data["main_claim"]["fact_checking_article"]
            urlOther = [data["related_articles"][j]["fact_checking_article"] for j in range(i)]
            
            documents = []
            textSelf = fetchArticleText(urlSelf)
            documents.append(textSelf)
            for url in urlOther:
                textOther = fetchArticleText(url)
                documents.append(textOther)
            
            scoresBM25 = compute_bm25_scores(claim, documents)
            scoresTFIDF = computeTfIdfScores(claim, documents)
            scoresSBERT = computeSbertScores(claim, documents, sbertModel)
            
            # if len(scoresBM25) > 0:
            #     print("BM25@{i}", scoresBM25[0], max(scoresBM25[1:i]))
            # else:
            #     print("No scores computed for BM25")
            # if len(scoresSBERT) > 0:
            #     print("SBERT@{i}", scoresSBERT[0], max(scoresSBERT[1:i]))
            # else:
            #     print("No scores computed for SBERT")
            # if len(scoresTFIDF) > 0:
            #     print("TFIDF@{i}", scoresTFIDF[0], max(scoresTFIDF[1:i]))
            # else:
            #     print("No scores computed for TFIDF")
            if len(scoresBM25) > 0:
                df.at[row_index, f'BM25_{i}_first'] = scoresBM25[0]
                df.at[row_index, f'BM25_{i}_max'] = max(scoresBM25[1:i+1])
            
            if len(scoresSBERT) > 0:
                df.at[row_index, f'SBERT_{i}_first'] = scoresSBERT[0]
                df.at[row_index, f'SBERT_{i}_max'] = max(scoresSBERT[1:i+1])
            
            if len(scoresTFIDF) > 0:
                df.at[row_index, f'TFIDF_{i}_first'] = scoresTFIDF[0]
                df.at[row_index, f'TFIDF_{i}_max'] = max(scoresTFIDF[1:i+1])

    df.to_excel('validation_results.xlsx', index=False)
    print("Results saved to validation_results.xlsx")

if __name__ == "__main__":
    main()