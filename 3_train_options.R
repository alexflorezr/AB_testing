# create the data partition
data(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=.7, list=F)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

# take a look at the variables in the training set
featurePlot(x=training[,c("age", "education", "jobclass")], y=training$wage, plot = "pairs")
featurePlot(x=training[,c("age", "education", "jobclass")], y=training$wage, plot = "scatter")
