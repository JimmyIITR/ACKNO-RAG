import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..'))) 

import pandas as pd
from sklearn.model_selection import train_test_split
import selectData

PATH = selectData.getStrData()

def main():
    # Load dataset
    df = pd.read_csv("PATH") #csv files only

    # Convert labels to numerical IDs
    label_map = {"supported": 0, "refuted": 1, "cherrypicking": 2}
    df["label"] = df["label"].map(label_map)

    # Split into train/validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df,val_df

if  __name__ == "__main__":
    main()