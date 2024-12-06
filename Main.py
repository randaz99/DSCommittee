import Data_Editing_Helpers


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


## Visualizing ##
#Data_Editing_Helpers.makeSNS(train)   # This oputputs all graphs, can be anoying
#print(train)


## Analyzing ##
predictions = Data_Editing_Helpers.decisiontree(train, test, 'sii')
#predictions = Data_Editing_Helpers.knn(train, test, "sii")  #Modle Sucks


## Submitting ##
Data_Editing_Helpers.generate_submission(predictions)



