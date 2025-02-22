import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

DATAPATH = "/Users/jimmyaghera/Downloads/Thesis/ACKNO-RAG/results/gamma/validation_results(1).xlsx"
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
    print(thresholdNormalized)

    threshold = scaler.inverse_transform([[thresholdNormalized]])[0][0]
    print(f"Optimal threshold between valid and invalid scores: {threshold:.2f}") 

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
    print(thresholdNormalized)

    threshold = scaler.inverse_transform([[thresholdNormalized]])[0][0]
    print(f"Optimal threshold between valid and invalid scores: {threshold:.2f}") 
    return thresholdNormalized

if __name__ == "__main__":
    model = "BM25"
    kValue = "2"
    print(model, kValue)
    tScore = getScore(model, kValue)
    # accuracy = thresholdNormalized(model, kValue, tScore)
