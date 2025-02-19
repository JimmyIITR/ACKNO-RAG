import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
import nltk
import re
from dataFetch import getCrossAndSelfURLsWithClaims

nltk.download('punkt')

def fetch_article_text(url):
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

def compute_bm25_scores(claim, documents):
    # Filter out documents that are empty after stripping whitespace
    tokenized_docs = [tokenize(doc) for doc in documents if doc.strip()]
    # Use explicit length check to avoid ambiguity
    if len(tokenized_docs) == 0:
        return []
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = tokenize(claim)
    scores = bm25.get_scores(query_tokens)
    return scores

def main():
    d = getCrossAndSelfURLsWithClaims(5) # current set for 5 others articls total will be +1 = 6
    for data in d:
        claim = data["main_claim"]["text"]
        
        print("Retrieving search results...")
        urlSelf = data["main_claim"]["fact_checking_article"]
        urlOther = []
        for i in range(0,5):
            urlOther.append(data["related_articles"][i]["fact_checking_article"])

        documents = []
        text = fetch_article_text(urlSelf)
        documents.append(text)
        for url in urlOther:
            text = fetch_article_text(url)
            documents.append(text)
        
        print("\nComputing BM25 relevance scores...")
        scores = compute_bm25_scores(claim, documents)
        
        if scores.any():
            print(scores)
        else:
            print("No scores computed.")

if __name__ == "__main__":
    main()
