import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) #code path is set

import selectData
import articleExtrection
import graphData
import nltk
import pandas as pd
from dataFetch import getLengthOfData
import json

nltk.download('punkt')

"""
will be adding articleExtrection and graphData join in this file
This file will work as follow:
1. get loop data of database for train
2. for every data extrect 2 article extra other then main one ### handeled in articleExtection.py
3. for combined document make graph data and retrive nodes also from graphData.py ##done 
4. calculate distance between co realted nodes both ways x and y both ##done
5. store it in excel file ##pending

Note: this training will take lot of time so handel try catch to avoid block if any  single data fails 
"""

RESULT_PATH = selectData.resultsAlphaBeta(0) #f'../result/alphaBeta/results.csv'

COLUMNS = [
    'claim',
    'true_true_horizontal',
    'true_true_vertical',
    'true_false_horizontal',
    'true_false_vertical'
]

def initialize_results(length):
    df = pd.DataFrame('', index=range(length), columns=COLUMNS)
    return df

def save_results(df):
    try:
        df.to_excel(RESULT_PATH, index=False)
        print(f"Results saved to {RESULT_PATH}")
    except Exception as e:
        print(f"Failed to save results: {e}")


def logisticRegression():
    pass

        
def main():
    total = getLengthOfData()
    print(f"Total items to process: {total}")

    results_df = initialize_results(total)

    for i in range(5):
        try:
            claim = articleExtrection.main(i)
            trueFalseRes = graphData.handleDataIngestion(i)

            results_df.loc[i, 'claim'] = claim
            results_df.loc[i, 'true_true_horizontal'] = json.dumps(
                trueFalseRes.get('true_true_horizontal', [])
            )
            results_df.loc[i, 'true_true_vertical'] = json.dumps(
                trueFalseRes.get('true_true_vertical', [])
            )
            results_df.loc[i, 'true_false_horizontal'] = json.dumps(
                trueFalseRes.get('true_false_horizontal', [])
            )
            results_df.loc[i, 'true_false_vertical'] = json.dumps(
                trueFalseRes.get('true_false_vertical', [])
            )
            print(f"Processed index {i}")
        except Exception as e:
            print(f"Error at index {i}: {e}. Skipping.")
        finally:
            save_results(results_df)
    

if __name__ == "__main__":
    main()

