## ---- quiz W3 ----
# Quiz for week 3 in practical machine learning
library(AppliedPredictiveModeling)
data(segmentationData)
library(caret)
summary(segmentationData)
inTrain <- createDataPartition(y=segmentationData$Class, p = .8, list = F)
training <- segmentationData[inTrain,]
testing <- segmentationData[-inTrain,]
set.seed(125)
model <- train(Class ~., data=training, method="rpart")
toPred <- read.delim("./toPred.txt", header = T, stringsAsFactors = F, sep = "\t")
predict(model, newdata=testing)
toPredict <- testing[1:4,]
toPredict[] <- NA
toPredict$Case <- as.factor(toPredict$Case)
xtoPred <- merge(toPredict, toPred, all=T)
predict(model, newdata=xtoPred[1:4,] )
plot(model$finalModel, uniform = T)
text(model$finalModel, use.n = T, all = T, cex=.8)
summary(model)

# olive oil 
library(pgmm)
data(olive)
olive = olive[,-1]
inTrain <- createDataPartition(y=olive$Area, p = .8, list = F)
training <- olive[inTrain,]
testing <- olive[-inTrain,]
model <- train(Area ~., data=training, method="rpart")
newdata = as.data.frame(t(colMeans(olive)))
plot(model$finalModel, uniform = T)
text(model$finalModel, use.n = T, all = T, cex=.8)
predict(model, newdata = newdata)

# Africa hearth disease
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
trainSA$chd <- as.factor(trainSA$chd)
testSA = SAheart[-train,]
testSA$chd <- as.factor(testSA$chd)
set.seed(13234)
model <- train(chd ~ age + alcohol+obesity+ tobacco+typea +ldl, data=trainSA, 
               method="glm",
               family="binomial")
missClass <- function(values,prediction){
        sum(((prediction > 0.5)*1) != values)/length(values)
}
predstest <- predict(model, newdata = testSA)
predstrain <- predict(model, newdata = trainSA)
missClass(testSA$chd, predstest)
missClass(trainSA$chd, predstrain)
## ---- quiz W4 ----
library(ElemStatLearn)
library(caret)
data(vowel.train)
data(vowel.test)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)

mod1 <- train(y ~., method="rf", data=vowel.train)
mod2 <- train(y ~., method="gbm", data=vowel.train)
pred1 <- predict(mod1,vowel.test)
pred2 <- predict(mod2,vowel.test)

predDF <- data.frame(pred1, pred2, y=vowel.test$y)
combModFit <- train(y ~., method="gam", data=predDF)
combPred <- predict(combModFit, predDF)

sum(pred1 == vowel.test$y) / length(pred1)
sum(pred2 == vowel.test$y) / length(pred2)

confusionMatrix(pred1, vowel.test$y)
confusionMatrix(pred2, vowel.test$y)
confusionMatrix(combPred, vowel.test$y)

sqrt(sum((as.numeric(pred1)-as.numeric(vowel.test$y))^2))
sqrt(sum((as.numeric(pred2)-as.numeric(vowel.test$y))^2))
sqrt(sum((as.numeric(combPred)-as.numeric(vowel.test$y))^2))
accuracy(pred1,vowel.test$y)

# second question
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
set.seed(62433)
mod1 <- train(diagnosis ~., method="rf", data=training)
pred1 <- predict(mod1,testing)
mod2 <- train(diagnosis ~., method="gbm", data=training)
pred2 <- predict(mod2,testing)
mod3 <- train(diagnosis ~., method="lda", data=training)
pred3 <- predict(mod3,testing)
predDF <- data.frame(pred1, pred2, pred3, diagnosis=testing$diagnosis)
combMod <- train(diagnosis ~., method="rf", data=predDF)
combPred <- predict(combMod, predDF)

confusionMatrix(pred1,testing$diagnosis)
confusionMatrix(pred2,testing$diagnosis)
confusionMatrix(pred3,testing$diagnosis)
confusionMatrix(combPred,testing$diagnosis)
sum(combPred  == predDF$diagnosis) / length(combPred)


set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
set.seed(233)
model = train(CompressiveStrength ~ ., method = 'lasso', data = training)
plot.enet(model$finalModel)


