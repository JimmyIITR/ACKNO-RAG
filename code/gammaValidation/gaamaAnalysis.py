import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

DATAPATH = "/Users/jimmyaghera/Downloads/Thesis/ACKNO-RAG/results/gamma/validation_results(4).xlsx"
df = pd.read_excel(f"{DATAPATH}") 

# TESTDATAPATH = "/Users/jimmyaghera/Downloads/Thesis/ACKNO-RAG/results/gamma/testdata_results.xlsx"
# dfTest = pd.read_excel(f"{TESTDATAPATH}") 

def getScore(model, kValue):
    valid = df[[f"{model}_{kValue}_first"]].copy()
    valid['label'] = 1  
    invalid = df[[f"{model}_{kValue}_max"]].copy()
    invalid['label'] = 0 

    combined = pd.concat([valid.rename(columns={f"{model}_{kValue}_first": 'score'}), invalid.rename(columns={f"{model}_{kValue}_max": 'score'})])
    combined = combined[combined['score'] != 0]

    scaler = MinMaxScaler()
    combined['normalized_score'] = scaler.fit_transform(combined[['score']])

    X = combined[['normalized_score']]
    y = combined['label']

    logReg = LogisticRegression()
    logReg.fit(X, y)

    coefficient = logReg.coef_[0][0]
    intercept = logReg.intercept_[0]
    thresholdNormalized = -intercept / coefficient

    threshold = scaler.inverse_transform([[thresholdNormalized]])[0][0]
    return threshold, thresholdNormalized



def thresholdNormalized(model, kValue, score):
    valid = df[['BM25_2_first']].copy()
    valid['label'] = 1  
    invalid = df[['BM25_2_max']].copy()
    invalid['label'] = 0 
    
    combined = pd.concat([valid.rename(columns={'BM25_2_first': 'score'}), invalid.rename(columns={'BM25_2_max': 'score'})])
    combined = combined[combined['score'] != 0]

    scaler = MinMaxScaler()
    combined['normalized_score'] = scaler.fit_transform(combined[['score']])

    X = combined[['normalized_score']]
    y = combined['label']

    logReg = LogisticRegression()
    logReg.fit(X, y)

    coefficient = logReg.coef_[0][0]
    intercept = logReg.intercept_[0]
    thresholdNormalized = -intercept / coefficient

    threshold = scaler.inverse_transform([[thresholdNormalized]])[0][0]
    return thresholdNormalized

if __name__ == "__main__":
    models = ["SBERT", "TFIDF", "BM25"]
    for kValue in range(8,11):
        for model in models:
            tNormalizedScore,tScore  = getScore(model, kValue)
            print(f"50*=")
            print(model , kValue)
            print(tScore, tNormalizedScore)
            print(f"50*=")
            
    
    # kValue = "10"
    # print(model, kValue)
    # tScore = getScore(model, kValue)
    # tNormalScore = getThresholdRaw(model, kValue)
    # print(tScore, tNormalScore)
    # accuracy = thresholdNormalized(model, kValue, tScore)
