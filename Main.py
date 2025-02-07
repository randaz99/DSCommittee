import Data_Editing_Helpers

## Loading ##
test = Data_Editing_Helpers.unCacheOrLoad("test.csv")
train = Data_Editing_Helpers.unCacheOrLoad("train.csv")
y_name = 'sii' #What your trying to predict
x_name = 'id' #Usser id. Drop this column


#train_parq = Data_Editing_Helpers.unCacheOrLoad("series_train.parquet")
#test_parq = Data_Editing_Helpers.unCacheOrLoad("series_test.parquet")


## Wrangling ##
train, test = Data_Editing_Helpers.map_seasons(train, test)
train, test = Data_Editing_Helpers.dropUnusedColumns(train, test, y_name, x_name)
train = Data_Editing_Helpers.remove_blank_rows(train, y_name)

#Using 99 as a filler for NA's Wil Change to use Random access forest for filling NA's 
train, test = Data_Editing_Helpers.fill_NA(train, test, fill=99)
X_train, X_test, y_train, y_test = Data_Editing_Helpers.traintestslpit(train, y_name)

## Visualizing ##
#Data_Editing_Helpers.makeSNS(train)   # This outputs all graphs, can be annoying
#print(train)


## Training Models ##
Data_Editing_Helpers.decisiontree(X_train, y_train, y_name) #Model Good
#Data_Editing_Helpers.knn(X_train, y_train, y_name)  #Modle Sucks
#Data_Editing_Helpers.adaboostClassifier(X_train, y_train, y_name)  #Modle good

## Making Predictions ##
# Change the path name to reflect the model you wish to make predictions with
modelpath = "./TrainedModels/decisionTreeModel.pkl"
predictions = Data_Editing_Helpers.makePredictionUsingModel(modelpath, test)


#analising models
accuracy_prediction = Data_Editing_Helpers.makePredictionUsingModel(modelpath, X_test)
Data_Editing_Helpers.accuracy(modelpath ,y_test, accuracy_prediction)


## Submitting ##
Data_Editing_Helpers.generate_submission(predictions, x_name, y_name)



