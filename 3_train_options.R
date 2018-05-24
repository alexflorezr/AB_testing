# create the data partition
data(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=.7, list=F)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

# take a look at the variables in the training set
featurePlot(x=training[,c("age", "education", "jobclass")], y=training$wage, plot = "pairs")
featurePlot(x=training[,c("age", "education", "jobclass")], y=training$wage, plot = "scatter")

# preprocess methods
preProc <- preProcess(log10(spam[,-58] + 1), method = "pca", pcaComp = 2)
str(preProc)
typeColor <- ifelse(spam$type == "spam", "grey", "lightblue2")
spamPC <- predict(preProc, log10(spam[,-58]+1))
head(spamPC)
plot(spamPC$PC1, spamPC$PC2, col=typeColor)

# after creating the principal components then you create a model
# you can use the normal train method, but add the option preProcess="pca", then the pca will be automatically included
inTrain <- createDataPartition(y=spam$type, p=.75, list=F)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
modelFit <- train(training$type ~ ., method="glm", preProcess="pca", data = training)
confusionMatrix(testing$type, predict(modelFit, testing))
str(modelFit)
