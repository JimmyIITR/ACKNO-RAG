import os
import sys
import time
import re
import nltk
import requests

from os.path import dirname, join, abspath
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from duckduckgo_search import DDGS
from gammaValidation import BM25GammaValidation, SBERTGammaValidation, TFIDFGammaValidation
import queryLog

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
if not SERPAPI_KEY:
    raise RuntimeError("Please set the SERPAPI_KEY environment variable")  

# Download tokenizer
nltk.download('punkt', quiet=True)
sbertModel = SentenceTransformer('all-MiniLM-L6-v2')

# Threshold constants
BM25_THRESHOLD  = 4.06336433559004
SBERT_THRESHOLD = 0.253111058547908
TFIDF_THRESHOLD = 0.349314796641909

SOCIAL_MEDIA_DOMAINS = {
    'twitter.com', 'facebook.com', 'instagram.com',
    'youtube.com', 'tiktok.com', 'linkedin.com',
    'pinterest.com', 'reddit.com', 'snapchat.com',
    'whatsapp.com', 'wechat.com', 'tumblr.com',
    'flickr.com', 'vk.com', 'quora.com'
}

def is_social_media(url):
    """Check if URL belongs to social media domain"""
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc.lower()
        return any(sm_domain in domain for sm_domain in SOCIAL_MEDIA_DOMAINS)
    except:
        return False

def getSession():
    """
    Returns a requests.Session() with retry logic for transient HTTP errors.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_top_urls_ddg(query, session, num_results=1):
    """
    Fetch top URLs from DuckDuckGo (non-social, non-doc).
    """
    try:
        urls = []
        seen = set()

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            time.sleep(2)
            for result in results:
                time.sleep(2)
                url = result.get('href') or result.get('url')
                if not url:
                    continue

                parsed = urlparse(url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

                if (clean_url not in seen and 
                    not is_social_media(clean_url) and 
                    not clean_url.endswith(('.pdf', '.doc', '.docx'))):
                    seen.add(clean_url)
                    urls.append(clean_url)

        return urls[:num_results]

    except Exception as e:
        print(f"Error fetching from DuckDuckGo: {str(e)}")
        return []

def fetchArticleText(url, session):
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def tokenize(text):
    return nltk.word_tokenize(text)

def main(claim, index=1) -> str:
    session = getSession()
    queryLog.log_entry(index, "Starting URL fetch", data=claim)

    try:
        urls = get_top_urls_ddg(claim, session, num_results=3)
        if not urls:
            queryLog.log_entry(index, "No URLs found after filtering", status="warning")
            return "Error: No valid URLs found after social media filtering"
            
        queryLog.log_entry(index, "Filtered URLs", data=urls, status="info")
        queryLog.log_entry(index, "Fetching articles...", status="info")

        # Prepare lists to hold up to 7 successful texts per method
        bm25_texts, tfidf_texts, sbert_texts = [], [], []

        for url in urls:
            # Stop early if all three have 7 items
            if len(bm25_texts) >= 1 and len(tfidf_texts) >= 1 and len(sbert_texts) >= 1:
                break

            time.sleep(2)  # throttle article fetches
            text = fetchArticleText(url, session)
            if not text:
                continue

            # Compute scores
            scoresBM25  = BM25GammaValidation.compute_bm25_scores(claim, [text])
            scoresTFIDF = TFIDFGammaValidation.computeTfIdfScores(claim, [text])
            scoresSBERT = SBERTGammaValidation.computeSbertScores(claim, [text], model=sbertModel)
            
            # Append only if above threshold and we still need more
            if abs(scoresBM25)  >= BM25_THRESHOLD and len(bm25_texts) < 1:
                bm25_texts.append(text)
            if scoresTFIDF >= TFIDF_THRESHOLD and len(tfidf_texts) < 1:
                tfidf_texts.append(text)
            if scoresSBERT >= SBERT_THRESHOLD and len(sbert_texts) < 1:
                sbert_texts.append(text)

        if len(bm25_texts)  < 1 or len(tfidf_texts) < 1 or len(sbert_texts) < 1:
            queryLog.log_entry(
                index,
                "Warning: fewer than 7 articles found for some methods",
                data={
                    "bm25": len(bm25_texts),
                    "tfidf": len(tfidf_texts),
                    "sbert": len(sbert_texts)
                },
                status="warning"
            )

        # Write exactly 7 (or fewer, if not found) to each file
        base = abspath(join(dirname(__file__), "../dataBase/temp/gammaExtrected"))
        with open(join(base, "BM25.txt"),  "a", encoding="utf-8") as f:
            f.write("\n\n".join(bm25_texts))
        with open(join(base, "TFIDF.txt"), "a", encoding="utf-8") as f:
            f.write("\n\n".join(tfidf_texts))
        with open(join(base, "SBERT.txt"), "a", encoding="utf-8") as f:
            f.write("\n\n".join(sbert_texts))

        queryLog.log_entry(index, "Fetching completed and saved.", status="info")
        return "Data fetch and file generation completed."
    except Exception as e:
        queryLog.log_entry(index, "Critical Error", data=str(e), status="critical")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    claim = "The government announced a major tax cut in 2024."
    print(main(claim, 1))
