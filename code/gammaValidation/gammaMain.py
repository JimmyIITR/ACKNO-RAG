import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
import nltk
import re

nltk.download('punkt')

def get_top_urls(query, num_results=10):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
        urls = [result.get('href') for result in results if result.get('href')]
        return urls
    except Exception as e:
        print(f"Search error: {e}")
        return []

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
    claim = "The government announced a major tax cut in 2024."
    
    print("Retrieving search results...")
    urls = get_top_urls(claim, num_results=10)
    if not urls:
        print("No URLs found.")
        return

    print("Found URLs:")
    for url in urls:
        print(url)
    
    print("\nFetching articles...")
    documents = []
    for url in urls:
        text = fetch_article_text(url)
        documents.append(text)
    
    for idx, doc in enumerate(documents):
        snippet = doc[:200] + "..." if len(doc) > 200 else doc
        print(f"\nDocument {idx+1} snippet:\n{snippet}")
    
    print("\nComputing BM25 relevance scores...")
    scores = compute_bm25_scores(claim, documents)
    
    if scores.any():
        for url, score in zip(urls, scores):
            print(f"{url} -> BM25 Score: {score:.2f}")
    else:
        print("No scores computed.")

if __name__ == "__main__":
    main()
