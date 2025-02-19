import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataFetch import getCrossAndSelfURLsWithClaims

nltk.download('punkt')

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

def computeSbertScores(claim, documents, model):
    nonEmptyDocs = [doc for doc in documents if doc.strip()]
    if len(nonEmptyDocs) == 0:
        return []
    
    claimEmbedding = model.encode([claim])
    docEmbeddings = model.encode(nonEmptyDocs)
    
    scoresMatrix = cosine_similarity(claimEmbedding, docEmbeddings)
    scores = scoresMatrix.flatten()
    return scores

def main():
    sbertModel = SentenceTransformer('all-MiniLM-L6-v2') #user all-mpnet-base-v2 insted infuture
    
    dataList = getCrossAndSelfURLsWithClaims(5)  # 5 related articles + 1 main = 6 total
    for data in dataList:
        claim = data["main_claim"]["text"]
        print("Claim:", claim)
        
        urlSelf = data["main_claim"]["fact_checking_article"]
        urlOther = [data["related_articles"][i]["fact_checking_article"] for i in range(5)]
        
        documents = []
        textSelf = fetchArticleText(urlSelf)
        documents.append(textSelf)
        for url in urlOther:
            textOther = fetchArticleText(url)
            documents.append(textOther)
        
        print("\nComputing SBERT relevance scores...")
        scores = computeSbertScores(claim, documents, sbertModel)
        
        if len(scores) > 0:
            print(scores)
            # for url, score in zip([urlSelf] + urlOther, scores):
            #     print(f"{url} -> SBERT Score: {score:.2f}")
        else:
            print("No scores computed.")

if __name__ == "__main__":
    main()
