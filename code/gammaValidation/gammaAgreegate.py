import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

nltk.download('punkt')

RESULT_PATH = abspath(join(dirname(__file__), '../../results/gamma'))

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

import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

def testMain():
    sbertModel = SentenceTransformer('all-MiniLM-L6-v2')
    session = getSession()  # Ensure this function is defined elsewhere

    BM25_THRESHOLD = [14.48266273, 15.89477579, 16.15302568]
    TFIDF_THRESHOLD = [0.2531110585, 0.2726348417, 0.2719207645]
    SBERT_THRESHOLD = [0.3493147966, 0.3647371782, 0.3641520826]

    # Initialize metric collectors
    all_scores = {'BM25': [], 'TFIDF': [], 'SBERT': []}
    all_labels = []
    threshold_metrics = {
        'BM25': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'TFIDF': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'SBERT': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    }

    for i in range(6, 9):
        idx = i - 6  # Index for threshold lists
        collected = 0
        max_attempts = 10  # Prevent infinite retries
        attempts = 0
        
        while collected < 50 and attempts < max_attempts:
            print(f"Processing i={i}, collected={collected}")
            dataList = getTestDataCrossAndSelfURLsWithClaims(i)
            print(dataList)
            # Skip empty/invalid dataList
            if not dataList:
                attempts += 1
                print(f"No data for i={i}, attempt {attempts}")
                time.sleep(1)  # Add delay to avoid spamming
                continue
            
            for data in dataList:
                if collected >= 250:
                    break  # Exit if target reached
                claim = data["main_claim"]["text"]
                urlSelf = data["main_claim"]["fact_checking_article"]
                urlOther = [data["related_articles"][j]["fact_checking_article"] for j in range(i)]

                # Fetch documents
                documents = []
                try:
                    textSelf = fetchArticleText(urlSelf, session)  # Ensure this function is defined elsewhere
                    documents.append(textSelf)
                    time.sleep(2)
                    for url in urlOther:
                        textOther = fetchArticleText(url, session)
                        documents.append(textOther)
                        time.sleep(2)
                except Exception as e:
                    print(f"Error fetching articles: {e}")
                    continue  # Skip this iteration if fetching fails

                # Compute scores
                try:
                    scoresBM25 = compute_bm25_scores(claim, documents)  # Ensure this function is defined elsewhere
                    scoresTFIDF = computeTfIdfScores(claim, documents)  # Ensure this function is defined elsewhere
                    scoresSBERT = computeSbertScores(claim, documents, sbertModel)  # Ensure this function is defined elsewhere
                except Exception as e:
                    print(f"Error computing scores: {e}")
                    continue  # Skip this iteration if computation fails

                # Labels: 1 for real article, 0 for false articles
                labels = [1] + [0] * i

                # Update threshold-based metrics
                for method, scores, threshold in zip(
                    ['BM25', 'TFIDF', 'SBERT'],
                    [scoresBM25, scoresTFIDF, scoresSBERT],
                    [BM25_THRESHOLD[idx], TFIDF_THRESHOLD[idx], SBERT_THRESHOLD[idx]]
                ):
                    for label, score in zip(labels, scores):
                        if score >= threshold:
                            if label == 1:
                                threshold_metrics[method]['TP'] += 1
                            else:
                                threshold_metrics[method]['FP'] += 1
                        else:
                            if label == 0:
                                threshold_metrics[method]['TN'] += 1
                            else:
                                threshold_metrics[method]['FN'] += 1

                # Collect scores and labels for curve plotting
                all_scores['BM25'].extend(scoresBM25)
                all_scores['TFIDF'].extend(scoresTFIDF)
                all_scores['SBERT'].extend(scoresSBERT)
                all_labels.extend(labels)
                print(threshold_metrics)               
                collected += 1
                print(f"Collected {collected}/250 for i={i}")
            
            attempts = 0  # Reset attempts if data is found
        if collected < 250:
            print(f"Warning: Only collected {collected} items for i={i}")
    print(threshold_metrics)

    # Convert labels to numpy array
    all_labels = np.array(all_labels)

    # Plot ROC and Precision-Recall curves
    for method in ['BM25', 'TFIDF', 'SBERT']:
        scores = np.array(all_scores[method])

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {method}')
        plt.legend()
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(all_labels, scores)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f'{method} (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {method}')
        plt.legend()
        plt.show()

    # Print threshold-based metrics
    for method in ['BM25', 'TFIDF', 'SBERT']:
        metrics = threshold_metrics[method]
        print(f"\n{method} Threshold Metrics:")
        print(f"True Positives: {metrics['TP']}")
        print(f"False Positives: {metrics['FP']}")
        print(f"True Negatives: {metrics['TN']}")
        print(f"False Negatives: {metrics['FN']}")

if __name__ == "__main__":
    testMain() #yet to be done have to test and check which one performs best among all of them.
