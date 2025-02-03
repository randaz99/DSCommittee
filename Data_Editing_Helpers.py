import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Mapping for seasons
season_mapping = {
    'Spring' : 1,
    'Summer' : 2,
    'Fall'   : 3,
    'Winter' : 4
}

def unCacheOrLoad(file):
    # Path for the cache file
    name = file.replace(".csv", '').replace(".parquet", '')
    path = 'cache/' + name + '_cache.joblib'

    # Check if the cache file exists
    if os.path.exists(path):
        # Load from joblib cache
        data = joblib.load(path)
        print(f"Read {file} from joblib cache")
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
        print(f"Data {file} loaded and cached with joblib")

    return data


def dropUnusedColumns(train, test, y_name, x_name):
    columns = train.columns.difference(test.columns)
    column_list = columns.to_list()
    column_list.append(x_name)
    column_list.remove(y_name)
    print(f'\nDeleting unsililare rows: {column_list}')

    train = train.drop(column_list, axis=1)
    test = test.drop(x_name, axis=1)
    #print(train.isna().sum())
    return train, test


def map_seasons(train, test):
    # Iterate over each column in the DataFrame
    for column in train.columns:
        # Check if any value in the column is a season (e.g., if the column has any value in the mapping keys)
        if train[column].isin(season_mapping.keys()).any():
            # Apply the mapping
            train[column] = train[column].map(season_mapping)
    for column in test.columns:
        # Check if any value in the column is a season (e.g., if the column has any value in the mapping keys)
        if test[column].isin(season_mapping.keys()).any():
            # Apply the mapping
            test[column] = test[column].map(season_mapping)
    return train, test


def fill_NA(train, test, fill):
    for col in train.columns:
        train[col] = train[col].fillna(fill)
    for col in test.columns:
        test[col] = test[col].fillna(fill)
    return train, test


def get_dummies(train, test):
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)    # needs to be changed from get dummies to something better!!!!
    return train, test


def remove_blank_rows(train, y_name):
    return train.dropna(subset=[y_name])


def makeSNS(train):
    sns.set_style("whitegrid")
    for column in train.columns:
        sns.countplot(train, x=column)
        plt.show()
    print("done")

def traintestslpit(train, y_name):
    x = train.drop(columns=[y_name])
    y = train[y_name]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    y_train = train[y_name]
    X_train = train.drop(y_name, axis=1)
    return X_train, X_test, y_train, y_test

def decisiontree(X_train, y_train, y_name):

    dt_model = DecisionTreeClassifier(random_state=1235)
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    random_search = RandomizedSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_

    best_model_dt = random_search.best_estimator_

    print(f'\nBest parameters Decisiontree: {best_params}')
    saveModel(best_model_dt, './TrainedModels/decisionTreeModel.pkl')


def knn(X_train, y_train,  y_name):

    dt_model = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
        "leaf_size": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "p": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    random_search = RandomizedSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=1103)
    random_search.fit(X_train, y_train)


    best_params = random_search.best_params_
    best_model_dt = random_search.best_estimator_

    print(f'\nBest parameters KNN: {best_params}')
    saveModel(best_model_dt, './TrainedModels/knnModel.pkl')

def adaboostClassifier(X_train, y_train, y_name):

    dt_model = AdaBoostClassifier()

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME']
    }
    random_search = GridSearchCV(dt_model, param_grid)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model_dt = random_search.best_estimator_

    print(f'\nBest parameters: {best_params}')

    saveModel(best_model_dt, './TrainedModels/adaModel.pkl')


def generate_submission(y_predictions, x_name, y_name):
    test = pd.read_csv('test.csv')
    submission = pd.DataFrame({x_name: test[x_name],
                               y_name : y_predictions})

    print(f'\nSubmission Preview:\n {submission}')
    submission.to_csv('submission.csvy', index=False)


def saveModel(model, file_name):
    joblib.dump(model, file_name)

def makePredictionUsingModel(file_path, X_test):
    model = joblib.load(file_path)
    y_pred_optimized = model.predict(X_test)
    return y_pred_optimized

def accuracy (file_path,y_pred_optimized, y_test):
    accuracy = accuracy_score(y_test, y_pred_optimized)
    print("Accuracy of the :" + str(file_path) + "{:.2f}%".format(accuracy * 100))