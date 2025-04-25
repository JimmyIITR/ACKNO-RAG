import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 
import selectData
import json
from typing import List, Dict, Tuple
import random

VALIDATIONDATAPATH = selectData.getDevAVeriTecData()
TRAINDATAPATH = selectData.getTrainAVeriTecData()
TESTDATAPATH = selectData.getTestAVeriTecData()

def loadData() -> List[Dict]:
    with open(VALIDATIONDATAPATH, 'r') as file:
        data = json.load(file)
    return data

def loadTestData() -> List[Dict]:
    with open(TESTDATAPATH, 'r') as file:
        data = json.load(file)
    return data

def getLengthOfData() -> int:
    data = loadData()
    valid = [
        entry for entry in data
        if entry.get("claim") and entry.get("fact_checking_article")
    ]
    return len(valid)


def getIthDataFromTrainData(I: int, K: int) -> dict:
    """
    Fetch the I-th valid claim from the TRAIN data and K random others.
    """
    data = loadData()
    
    valid = [
        entry for entry in data
        if entry.get("claim") and entry.get("fact_checking_article")
    ]
    
    if I < 0 or I >= len(valid):
        raise IndexError(f"I={I} out of range [0, {len(valid)-1}]")
    
    main = valid[I]
    
    other_indices = [idx for idx in range(len(valid)) if idx != I]
    
    sample_size = min(K, len(other_indices))
    chosen = random.sample(other_indices, k=sample_size)  # unique picks :contentReference[oaicite:1]{index=1}
    
    related = [
        {
            "related_claim": valid[idx]["claim"],
            "fact_checking_article": valid[idx]["fact_checking_article"],
        }
        for idx in chosen
    ]
    result = []
    result.append({
        "main_claim": {
            "text": main["claim"],
            "fact_checking_article": main["fact_checking_article"],
        },
        "related_articles": related
    })
    return  result

def main():
    result = getIthDataFromTrainData(3, 2)
    print(result)

if __name__ == "__main__":
    main()