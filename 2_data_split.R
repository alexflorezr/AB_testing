## Coursera ####
# cross-validation using K-fold
# creates a list of vectors with the indexes for the folds, if returnTrain == T
# ... it will return the values for the training set, k indicate the number of folds
set.seed(32323)
folds <- createFolds(y=spam$type, k=10, list=T, returnTrain = T)

# create resample, in this case with replacement, like bootstrap
folds <- createResample(y=spam$type, times=10, list=T)
str(folds)

# create time slices
tme <- 1:1000
folds <- createTimeSlices(y=tme, initialWindow = 20, horizon = 10)
names(folds)
folds$train[1:10]
folds$test[1:10]

## Applied predictive modeling ####
## two class data ####
library(AppliedPredictiveModeling)
library(caret)
data(twoClassData)
str(predictors)
head(predictors)
str(classes)
head(classes)
head(twoClassData)

trainingRows <- createDataPartition(classes, p=.80, list=F)
trainPredictors <- predictors[trainingRows,]
trainClasses <- classes[trainingRows]
testPredictors <- predictors[-trainingRows,]
testClasses <- classes[-trainingRows]
# create the knn model
knnfit <- knn3(x=trainPredictors, y=trainClasses, k=5)
knnfit$theDots

# use the predict function to predict the test data
testPredictions <- predict(knnfit, newdata = testPredictors, type = "class")
confusionMatrix(testPredictions, testClasses)



