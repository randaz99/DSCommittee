from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
import joblib


def readData():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = readData()

def dropColumns(file, cols: list):
    file = file.drop(cols, axis=1, inplace=True)

    # print(file.isna().sum())

# Mapping seasons to numbers
season_mapping = {
    'Spring' : 0,
    'Summer' : 1,
    'Fall'   : 2,
    'Winter' : 3
}
def map_seasons(file):
    season_mapping: dict[str, int] = {
        'Spring': 0,
        'Summer': 1,
        'Fall': 2,
        'Winter': 3
    }
    # Iterate over each column in the DataFrame
    for column in file.columns:
        # Check if any value in the column is a season (e.g., if the column has any value in the mapping keys)
        if file[column].isin(season_mapping.keys()).any():
            # Apply the mapping
            file[column] = file[column].map(season_mapping)



def makeSNS():
    sns.set_style("whitegrid")
    for column in train.columns:
        sns.countplot(train, x=column, hue="sii")
        plt.show()
    print("done")

def unCacheOrLoad(file):
    # Path for the cache file
    name = file.replace(".csv", '').replace(".parquet", '')
    path = 'cache/' + name + '_cache.joblib'

    # Check if the cache file exists
    if os.path.exists(path):
        # Load from joblib cache
        data = joblib.load(path)
        print("Read from joblib cache")
    else:
        # Load from the original source
        if file.endswith(".csv"):
            data = pd.read_csv(file)
        elif file.endswith(".parquet"):
            data = pq.read_table(file).to_pandas()
        else:
            raise ValueError("Unsupported file format")

        # Save to joblib cache
        os.makedirs('cache', exist_ok=True)
        joblib.dump(data, path)
        print("Data loaded and cached with joblib")

    return data

# Find out what columns are in train df but not in test df
def findMissingCols():
    missingCols = []
    for col in train.columns[:-1]:
        if col not in test.columns:
            missingCols.append(col)
    return missingCols