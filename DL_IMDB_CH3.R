## ---- imdb ----
library(keras)
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
# reverse the comment to words 
# preparing the data, neural networks can not be feed with lists, we need tensors
vectorize_sequences <- function(sequences, dimension=10000){
        results <- matrix(0, nrow = length(sequences), ncol = dimension)
        for (i in 1:length(sequences)){
                # sends 1 to the columns with word values
                results[i, sequences[[i]]] <- 1
        }
        results
}
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
# convert the labels to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# create the model
model <- keras_model_sequential() %>%
        layer_dense(units = 16, activation="relu", input_shape = c(10000)) %>% 
        layer_dense(units = 16, activation="relu") %>% 
        layer_dense(units=1, activation="sigmoid")
# compile the model
model %>%  compile(
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy")
)
# validate the approach
# take 10000 samples from the training data to validate the results
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]
# training the model
history <- model %>% fit(
        partial_x_train, 
        partial_y_train,
        epochs = 20,
        batch_size = 512,
        validation_data = list(x_val, y_val)
)
# observing overfitting 
# in the plot(history) -epoch = 20 - the max accuracy for the validation data
#       is at epoch 4, similarly the lowest loss is at epoch 4

# A new model using only 4 epochs
model %>% fit(
        x_train,
        y_train,
        epochs = 4,
        batch_size = 512
)
results <- model %>% evaluate(x_test, y_test)
results
x_pred <- model %>%  predict(x_test[1:10,])
plot(y_test[1:10], x_pred)



## ---- reuters ----
# import the data
reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters
str(train_data)
# vectorize the data
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
# vectorize the labels
one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)
# create validation data
val_indices <- 1:1000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- one_hot_train_labels[val_indices,]
partial_y_train <- one_hot_train_labels[-val_indices,]
     

# create the model
model <- keras_model_sequential() %>% 
        layer_dense(units = 64, activation="relu", input_shape = c(10000)) %>% 
        layer_dense(units = 64, activation="relu") %>% 
        layer_dense(units = 46, activation="softmax")

# compile model
model %>% compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss="categorical_crossentropy"
)

history20 <- model %>% fit(
        partial_x_train,
        partial_y_train,
        epochs = 20, 
        batch_size = 512,
        validation_data = list(x_val, y_val)
)
# with 20 epochs is possible to observe that overfit starts at the nineth epoch
model <- keras_model_sequential() %>% 
        layer_dense(units = 64, activation="relu", input_shape = c(10000)) %>% 
        layer_dense(units = 64, activation="relu") %>% 
        layer_dense(units = 46, activation="softmax")

# compile model
model %>% compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss="categorical_crossentropy"
)

history09 <- model %>% fit(
        partial_x_train,
        partial_y_train,
        epochs = 9, 
        batch_size = 512,
        validation_data = list(x_val, y_val)
)
# evaluate the results in the test data
results <- model %>%  evaluate(x_test, one_hot_test_labels)
results
# predict the values using the NN 
predictions <- model %>% predict(x_test)
# plot the predictions
x_max <- apply(predictions, 1, which.max)
plot((as.numeric(test_labels)+1), x_max, pch=16, col="#A3A3A385")
sum((as.numeric(test_labels)+1) == x_max)
# testing the constraints in the size of the layers
# for the reuters data the final layers is 46, then the previous ones must be larger
# here is an example using a second layer with size 4
model <- keras_model_sequential() %>% 
        layer_dense(units = 64, activation="relu", input_shape = c(10000)) %>% 
        layer_dense(units = 4, activation="relu") %>% 
        layer_dense(units = 46, activation="softmax")

# compile model
model %>% compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss="categorical_crossentropy"
)

history09 <- model %>% fit(
        partial_x_train,
        partial_y_train,
        epochs = 20, 
        batch_size = 512,
        validation_data = list(x_val, y_val)
)
## ---- Boston housing ----
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset
# normalize and scale the data
mean <- apply(train_data, 2, mean)
sd <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = sd)
test_data <- scale(test_data, center=mean, scale = sd)
# build the model 
build_model <- function(){
        model <- keras_model_sequential() %>% 
                layer_dense(units = 64, activation = "relu", 
                            input_shape = dim(train_data)[2]) %>% 
                layer_dense(units = 64, activation = "relu") %>% 
                layer_dense(units = 1)
        
        model %>% compile(
                optimizer = "rmsprop",
                loss = "mse",
                metrics = c("mae")
        )
}
# k-fold crossvalidation
k <- 4
indices <- 1:nrow(train_data)
folds <- cut(indices, breaks = k, labels = F)
num_epochs <- 100
all_scores <- c()
for (i in 1:k){
        cat("processing fold #", i, "\n")
        
        val_indices <- which(folds == i, arr.ind = T)
        val_data <- train_data[val_indices,]
        val_targets <- train_targets[val_indices]
        partial_train_data <- train_data[-val_indices,]
        partial_train_targets <- train_targets[-val_indices]
        
        model <- build_model()
        model %>%  fit(
                partial_train_data,
                partial_train_targets,
                epochs = num_epochs,
                batch_size = 1, 
                verbose = 0
        )
        results <- model %>% evaluate(val_data, val_targets, verbose = 0)
        all_scores <- c(all_scores, results$mean_absolute_error)
}
# saving the validation logs at each fold
num_epochs <- 500
all_mae_histories <- NULL
for (i in 1:k){
        cat("processing fold #", i, "\n")
        
        val_indices <- which(folds == i, arr.ind = T)
        val_data <- train_data[val_indices,]
        val_targets <- train_targets[val_indices]
        partial_train_data <- train_data[-val_indices,]
        partial_train_targets <- train_targets[-val_indices]
        
        model <- build_model()
        history <- model %>%  fit(
                partial_train_data,
                partial_train_targets,
                validation_data = list(val_data, val_targets),
                epochs = num_epochs,
                batch_size = 1, 
                verbose = 0
        )
        mae_history <- history$metrics$val_mean_absolute_error
        all_mae_histories <- rbind(all_mae_histories, mae_history)
}
# plot the epochs results
average_mae_history <- data.frame(
        epoch = seq(1,ncol(all_mae_histories)),
        validation_mae = apply(all_mae_histories, 2, mean)
)
ggplot(average_mae_history, aes(x=epoch, y=validation_mae)) + geom_line()
ggplot(average_mae_history, aes(x=epoch, y=validation_mae)) + geom_smooth()
# the minimun value for mae is around the 80th epoch, the model starts overfitting
#       after around 120 epochs
# build the final model
model <- build_model()
model %>% fit(
        train_data,
        train_targets,
        epochs = 80, 
        batch_size = 16,
        verbose = 0
)
results <- model %>% evaluate(test_data, test_targets)
results
