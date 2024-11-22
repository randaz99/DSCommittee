import Data_Editing_Helpers

# Loading
test = Data_Editing_Helpers.unCacheOrLoad("test.csv")
train = Data_Editing_Helpers.unCacheOrLoad("train.csv")
train_parq = Data_Editing_Helpers.unCacheOrLoad("series_train.parquet")
#test_parq = Data_Editing_Helpers.unCacheOrLoad("series_test.parquet")

# Wrangling
train, test = Data_Editing_Helpers.dropID(train, test, 'sii')
train, test = Data_Editing_Helpers.fill_NA(train, test, fill=0)
train, test = Data_Editing_Helpers.convert_strings(train, test)
train = Data_Editing_Helpers.remove_blank_rows(train)

# Visualizing
#Data_Editing_Helpers.makeSNS(train)
#print(train)

# Analyzing
Data_Editing_Helpers.find_best_params(train, 'sii')



