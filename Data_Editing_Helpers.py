import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def readData():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = readData()

def dropID():
    train = train.drop("id", axis=1)
    print(train.isna().sum())
    # Mapping seasons to numbers

season_mapping = {
    'Spring' : 0,
    'Summer' : 1,
    'Fall'   : 2,
    'Winter' : 3
}
def plotCounts():
    # Iterate over each column in the DataFrame
    for column in train.columns:
        # Check if any value in the column is a season (e.g., if the column has any value in the mapping keys)
        if train[column].isin(season_mapping.keys()).any():
            # Apply the mapping
            train[column] = train[column].map(season_mapping)

def makeSNS():
    sns.set_style("whitegrid")
    for column in train.columns:
        sns.countplot(train, x=column, hue="sii")
        plt.show()
    print("done")

makeSNS()
