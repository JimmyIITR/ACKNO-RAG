import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import nltk
import re
from dataFetch import getCrossAndSelfURLsWithClaims
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def tokenize(text):
    return nltk.word_tokenize(text)

def computeTfIdfScores(claim, documents):
    nonEmptyDocs = [doc for doc in documents if doc.strip()]
    if len(nonEmptyDocs) == 0:
        return []
    
    vectorizer = TfidfVectorizer()
    docVectors = vectorizer.fit_transform(nonEmptyDocs)
    queryVector = vectorizer.transform([claim])
    
    scoresMatrix = cosine_similarity(queryVector, docVectors)
    scores = scoresMatrix.flatten()
    return scores

def main():
    dataList = getCrossAndSelfURLsWithClaims(5)  # 5 related articles + 1 main = 6 total
    for data in dataList:
        claim = data["main_claim"]["text"]
        urlSelf = data["main_claim"]["fact_checking_article"]
        urlOther = [data["related_articles"][i]["fact_checking_article"] for i in range(5)]
        
        documents = []
        textSelf = fetchArticleText(urlSelf)
        documents.append(textSelf)
        for url in urlOther:
            textOther = fetchArticleText(url)
            documents.append(textOther)
        
        scores = computeTfIdfScores(claim, documents)
        
        if len(scores) > 0:
            print(scores)
        else:
            print("No scores computed.")

if __name__ == "__main__":
    main()
