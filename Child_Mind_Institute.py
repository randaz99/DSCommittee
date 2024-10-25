import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.isna().sum())
train = pd.get_dummies(train)
train["Basic_Demos-Enroll_Season"] = [0 if i=="Spring"  i in train["Basic_Demos-Enroll_Seaso"]]
train.dropna(inplace=False)

sns.heatmap(train, cmap='coolwarm', annot=True)
