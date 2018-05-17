library(caret); library(kernlab); data(spam)
# creates a vector of random numbers equivalent to the p (proportion indicated)
# in this specific case 75% for the training and 25 for the testing
inTrain <- createDataPartition(y=spam$type, p=0.75, list=F)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

# create a model using the training data
set.seed(32343)
modelFit <- train(type ~., data=training, method="glm")
modelFit$finalModel
# use the created model to predict data in the testing data
predictions <- predict(modelFit, newdata=testing)
head(predictions)
# create a confusion matrix (contingency); also the specificity and sensitivity
confusionMatrix(predictions, testing$type)
