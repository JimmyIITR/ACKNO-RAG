import json
from typing import List, Dict, Tuple

VALIDATIONDATAPATH = "data/AVeriTeCData/dataDev.json"
TRAINDATAPATH = "data/AVeriTecData/dataTrain.json"
TESTDATAPATH = "data/AVeriTecData/dataTest.json"

def loadData() -> List[Dict]:
    with open(VALIDATIONDATAPATH, 'r') as file:
        data = json.load(file)
    return data

def loadTestData() -> List[Dict]:
    with open(TESTDATAPATH, 'r') as file:
        data = json.load(file)
    return data

def getURLS() -> List[str]:
    data = loadData()
    res = []
    for claim in data:
        if claim.get('fact_checking_article'):
            res.append(claim['fact_checking_article'])
    return res

def getClaimsAndVerdict() -> List[Tuple[str, str]]:
    data = loadData()
    res = []
    for claim in data:
        if claim.get('claim') and claim.get('label'):
            res.append((claim['claim'], claim['label']))
    return res

def getClaimsURLsPair() -> List[Tuple[str, str]]:
    data = loadData()
    res = []
    for claim in data:
        if claim.get('claim') and claim.get('fact_checking_article'):
            res.append((claim['claim'], claim['fact_checking_article']))
    return res

def getClaimsURLsAndVerdict() -> List[Tuple[str, str, str]]:
    data = loadData()
    res = []
    for claim in data:
        if all(claim.get(key) for key in ['claim', 'fact_checking_article', 'label']):
            res.append((
                claim['claim'],
                claim['fact_checking_article'],
                claim['label']
            ))
    return res

def getCrossAndSelfURLsWithClaims(k) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    data = loadData()
    result = []
    
    for claim_data in data:
        if not (claim_data.get('claim') and claim_data.get('fact_checking_article')):
            continue
            
        related_articles = []
        for other_claim in data:
            if (other_claim['claim'] != claim_data['claim'] and 
                other_claim.get('fact_checking_article') and 
                other_claim['fact_checking_article'] != claim_data['fact_checking_article']):
                related_articles.append({
                    'related_claim': other_claim['claim'],
                    'fact_checking_article': other_claim['fact_checking_article'],
                })
            if len(related_articles) == k:
                break
                
        claim_set = {
            "main_claim": {
                "text": claim_data['claim'],
                "fact_checking_article": claim_data['fact_checking_article']
            },
            "related_articles": related_articles
        }
        
        result.append(claim_set)
    
    return result

def getKCrossAndSelfURLsWithClaims(k,T) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    data = loadData()
    result = []
    
    for claim_data in data:
        if not (claim_data.get('claim') and claim_data.get('fact_checking_article')):
            continue
            
        related_articles = []
        for other_claim in data:
            if (other_claim['claim'] != claim_data['claim'] and 
                other_claim.get('fact_checking_article') and 
                other_claim['fact_checking_article'] != claim_data['fact_checking_article']):
                related_articles.append({
                    'related_claim': other_claim['claim'],
                    'fact_checking_article': other_claim['fact_checking_article'],
                })
            if len(related_articles) == k:
                break
                
        claim_set = {
            "main_claim": {
                "text": claim_data['claim'],
                "fact_checking_article": claim_data['fact_checking_article']
            },
            "related_articles": related_articles
        }
        
        result.append(claim_set)
    
    return result[0:T]

def getTestDataCrossAndSelfURLsWithClaims(k) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    data = loadTestData()
    result = []
    
    for claim_data in data:
        if not (claim_data.get('claim') and claim_data.get('fact_checking_article')):
            continue
            
        related_articles = []
        for other_claim in data:
            if (other_claim['claim'] != claim_data['claim'] and 
                other_claim.get('fact_checking_article') and 
                other_claim['fact_checking_article'] != claim_data['fact_checking_article']):
                related_articles.append({
                    'related_claim': other_claim['claim'],
                    'fact_checking_article': other_claim['fact_checking_article'],
                })
            if len(related_articles) == k:
                break
                
        claim_set = {
            "main_claim": {
                "text": claim_data['claim'],
                "fact_checking_article": claim_data['fact_checking_article']
            },
            "related_articles": related_articles
        }
        
        result.append(claim_set)
    
    return result

def main():
    # urls = getURLS()
    # print(f"Total number of fact checking URLs: {len(urls)}")
    
    # claims_verdicts = getClaimsAndVerdict()
    # print(f"\nTotal number of claim-verdict pairs: {len(claims_verdicts)}")
    
    # verdicts = [v for _, v in claims_verdicts]
    # verdict_dist = {v: verdicts.count(v) for v in set(verdicts)}
    # print("\nVerdict distribution:")
    # for verdict, count in verdict_dist.items():
    #     print(f"{verdict}: {count}")
    
    result = getCrossAndSelfURLsWithClaims(5)
    print(result[0])

if __name__ == "__main__":
    main()