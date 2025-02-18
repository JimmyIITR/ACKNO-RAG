import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
import nltk
import re

# Ensure you have the necessary NLTK data
nltk.download('punkt')

def get_top_urls(query, num_results=10):
    # Use DuckDuckGo to retrieve URLs for the query
    results = DDGS(query, max_results=num_results)
    urls = [result['href'] for result in results if 'href' in result]
    return urls

def fetch_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Use BeautifulSoup to extract text content
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ')
            # Clean up whitespace and non-alphabetic characters
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        else:
            return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def tokenize_text(text):
    # Tokenize the text into words
    return nltk.word_tokenize(text)

def compute_bm25_scores(claim, documents):
    # Tokenize claim (query) and each document
    tokenized_docs = [tokenize_text(doc) for doc in documents if doc]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = tokenize_text(claim)
    scores = bm25.get_scores(query_tokens)
    return scores

def main():
    claim = "The government announced a major tax cut in 2024."
    # Step 1: Retrieve top 10 URLs using DuckDuckGo
    urls = get_top_urls(claim, num_results=10)
    print("Retrieved URLs:")
    for url in urls:
        print(url)
    
    # Step 2: Fetch and clean text from each URL
    documents = [fetch_article_text(url) for url in urls]
    
    # Optionally, print a summary of each document
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1} snippet: {doc[:200]}...")
    
    # Step 3: Compute BM25 scores for each document given the claim
    scores = compute_bm25_scores(claim, documents)
    
    print("\nBM25 relevance scores for each document:")
    for url, score in zip(urls, scores):
        print(f"{url} -> Score: {score:.2f}")

if __name__ == "__main__":
    main()
