
from dataBase import queries
import prompts
import selectData
import articleExtrection
import graphData
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import nltk
import re
import pandas as pd
from collections import defaultdict
from dataFetch import getLengthOfData
from sentence_transformers import SentenceTransformer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from dataBase.queries import twoNodeConnection

nltk.download('punkt')

"""
will be adding articleExtrection and graphData join in this file
This file will work as follow:
1. get loop data of database for train
2. for every data extrect 2 article extra other then main one 
3. for combined document make graph data and retrive nodes also from graphData.py
4. calculate distance between co realted nodes both ways x and y both
5. store it in excel file

Note: this training will take lot of time so handel try catch to avoid block if any  single data fails 
"""

RESULT_PATH = selectData.resultsAlphaBeta(0)

def main():
    l = getLengthOfData()
    for i in range(0,l):
        articleExtrection.main(i) #part 1
        print(f"files created successfully for {i}")
        trueFalseRes = graphData.handleDataIngestion(i) #part 2
        true = trueFalseRes["true"]
        false = trueFalseRes["false"]
        #calculate x and y for both true and false for all the strings 
        #and store in some data formet
        print(trueFalseRes)

    #apply logistic regression on this 
    #get X and Y value by 2 seprate logisctic regression function
    #make graph for analyssi on the data generated
    

if __name__ == "__main__":
    main()

