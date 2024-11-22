import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def readData():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def dropID(train, test, y_name):
    columns = train.columns.difference(test.columns)
    column_list = columns.to_list()
    column_list.append('id')
    column_list.remove(y_name)
    print(column_list)

    train = train.drop(column_list, axis=1)
    test = test.drop('id', axis=1)
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
        sns.countplot(train, x=column)
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

def find_best_params(train, y_name):
    y_train = train['sii']
    X_train = train.drop(y_name, axis=1)
    dt_model = DecisionTreeClassifier(random_state=0)
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    random_search = RandomizedSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

def generate_submission():
    pass

def convert_strings(train, test):
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    return train, test

def remove_blank_rows(train):
    return train.dropna(subset=['sii'])
