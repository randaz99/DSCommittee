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

def dropID(train, test):
    train = train.drop("id", axis=1)
    test = test.drop("id", axis=1)
    print(train.isna().sum())
    return train, test

season_mapping = {
    'Spring' : 0,
    'Summer' : 1,
    'Fall'   : 2,
    'Winter' : 3
}
def plotCounts(train):
    # Iterate over each column in the DataFrame
    for column in train.columns:
        # Check if any value in the column is a season (e.g., if the column has any value in the mapping keys)
        if train[column].isin(season_mapping.keys()).any():
            # Apply the mapping
            train[column] = train[column].map(season_mapping)

def makeSNS(train):
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

def fill_NA(train, test, fill=0):
    for col in train.columns:
        train[col] = train[col].fillna(fill)
    for col in test.columns:
        test[col] = test[col].fillna(fill)
    return train, test
