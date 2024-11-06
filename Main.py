import Data_Editing_Helpers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
import joblib


# Load data
test = Data_Editing_Helpers.unCacheOrLoad("test.csv")
train = Data_Editing_Helpers.unCacheOrLoad("train.csv")
train_parq = Data_Editing_Helpers.unCacheOrLoad("series_train.parquet")
test_parq = Data_Editing_Helpers.unCacheOrLoad("series_test.parquet")

# Was thinking maybe we shouldn't drop ID because we might need to merge parquet and csv by ID

# Map season to corresponding integer value
#     'Spring' : 0
#     'Summer' : 1
#     'Fall'   : 2
#     'Winter' : 3
Data_Editing_Helpers.plotCounts()

# Uncomment this part when need to plot
# Data_Editing_Helpers.makeSNS()

# To drop column in train
# Based on the plots, these columns might not be valuable
toDrop = ['Physical-BMI',
          'Physical-Height',
          'Physical-Weight',
          'Physical-Diastolic_BP',
          'Physical-Systolic_BP',
          'FGC-FGC_GSND',
          'FGC-FGC_GSD',
          'FGC-FGC_PU',
          'BIA-BIA_BMC',
          'BIA-BIA_BMI',
          'BIA-BIA_BMR',
          'BIA-BIA_DEE',
          'BIA-BIA_ECW',
          'BIA-BIA_FFM',
          'BIA-BIA_FFMI',
          'BIA-BIA_FMI',
          'BIA-BIA_Fat',
          'BIA-BIA_ICW',
          'BIA-BIA_LDM',
          'BIA-BIA_LST',
          'BIA-BIA_SMM',
          'BIA-BIA_TBW',
          'PAQ_A-PAQ_A_Total',
          'PAQ_C-PAQ_C_Total']

Data_Editing_Helpers.dropColumns(train, toDrop)
Data_Editing_Helpers.dropColumns(test, toDrop)


print(train.drop('id', axis = 1).head())



# Find out what features is missing in test csv
missingCols = Data_Editing_Helpers.findMissingCols()
Data_Editing_Helpers.dropColumns(train, missingCols)

# Data_Editing_Helpers.makeSNS()
# print(train.columns)

# Replace na values with corresponding column's median value
# train = train.fillna(train.drop('id', axis = 1).median())
# test = test.fillna(test.drop('id', axis = 1).median())
# print(train.isna().sum())

# Plot a heatmap to decide which features to drop
# corr_matrix = train.drop('id', axis = 1).corr()
# plt.figure(figsize = (20,20))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title("Correlation Train")
# plt.show()


# Processing parquet files
# toDropCols = ['step',
#               'non-wear_flag',
#               'battery_voltage']
#
# train_parq = Data_Editing_Helpers.dropColumns(train_parq, toDropCols)

# print(train_parq['id'].head())
