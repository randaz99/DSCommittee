from odbc import dataError

import Data_Editing_Helpers
from Data_Editing_Helpers import accuracy

## Loading ##
test = Data_Editing_Helpers.unCacheOrLoad("test.csv")
train = Data_Editing_Helpers.unCacheOrLoad("train.csv")
#train_parq = Data_Editing_Helpers.unCacheOrLoad("series_train.parquet")
#test_parq = Data_Editing_Helpers.unCacheOrLoad("series_test.parquet")


## Wrangling ##
train, test = Data_Editing_Helpers.map_seasons(train, test)
train, test = Data_Editing_Helpers.dropID(train, test, 'sii')
train = Data_Editing_Helpers.remove_blank_rows(train)
train, test = Data_Editing_Helpers.fill_NA(train, test, fill=99)
X_train, X_test, y_train, y_test, y_name = Data_Editing_Helpers.traintestslpit(train, test, "sii")

## Visualizing ##
#Data_Editing_Helpers.makeSNS(train)   # This oputputs all graphs, can be anoying
#print(train)


## Training Models ##
Data_Editing_Helpers.decisiontree(X_train, y_train, y_name)
#Data_Editing_Helpers.knn(train, test, "sii")  #Modle Sucks
#Data_Editing_Helpers.adaboostClassifier(train, test, "sii")  #Modle good

## Making Predictions ##
# Change the path name to reflect the model you wish to make predictions with       vvvvvvvvvvvvvi
modelpath = "./TrainedModels/decisionTreeModel.pkl"
predictions = Data_Editing_Helpers.makePredictionUsingModel(modelpath, test)


#analising models
accuracy_prediction = Data_Editing_Helpers.makePredictionUsingModel(modelpath, X_test)
Data_Editing_Helpers.accuracy(y_test, accuracy_prediction)


## Submitting ##
Data_Editing_Helpers.generate_submission(predictions)



