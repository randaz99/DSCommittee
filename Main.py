import Data_Editing_Helpers
from Data_Editing_Helpers import dropID

# Loading
test = Data_Editing_Helpers.unCacheOrLoad("test.csv")
train = Data_Editing_Helpers.unCacheOrLoad("train.csv")
train_parq = Data_Editing_Helpers.unCacheOrLoad("series_train.parquet")
#test_parq = Data_Editing_Helpers.unCacheOrLoad("series_test.parquet")

# Wrangling
train, test = Data_Editing_Helpers.dropID(train, test)
train, test = Data_Editing_Helpers.fill_NA(train, test, fill=0)

# Visualizing
Data_Editing_Helpers.makeSNS()


