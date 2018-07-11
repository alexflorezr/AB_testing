# neural network to classify clothing items
# upload the data set
fmnist <- dataset_fashion_mnist()
# assign the data and labels
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% fmnist
# plot one of the items in the training data
plot(as.raster(train_data[1,,]/255))
# data must have the shape (sample, rows, colums, channels)
# divide the data into training and validation
# for this case I will use 70:30
# dividing by 255 to rescale the data
train_sample <- sample(1:dim(train_data)[1], dim(train_data)[1]*.7)
train_data_x <- train_data[train_sample,,]/255
train_labels_y <- to_categorical(train_labels[train_sample])
validation_data_x <- train_data[-train_sample,,]/255
validation_labels_y <- to_categorical(train_labels[-train_sample])
# reshape
size <- function(x) array_reshape(x, dim=c(dim(x)[1], 28* 28))
train_data_x <- size(train_data_x)
validation_data_x <- size(validation_data_x)
dim(train_data_x)
## ---- densely_connected_layers ----
# create the model
model <- keras_model_sequential() %>%
        layer_dense(units=64, activation="relu", input_shape = c(784)) %>% 
        layer_dense(units=32, activation="relu") %>% 
        layer_dense(units = 10, activation="softmax")

model %>% compile(
        metrics = c("accuracy"),
        optimizer = "rmsprop",
        loss = "categorical_crossentropy"
)

history <- model %>% fit(
        train_data_x, 
        train_labels_y, 
        epochs=20,
        batch_size = 128, 
        validation_data = list(validation_data_x, validation_labels_y)
)        
max(history$metrics$val_acc)
## ---- convolutional_net ----
# in this case the images will be storages in arrays of dim = samples, w, h, channel
size_cnn <- function(x) array_reshape(x, dim=c(dim(x)[1], 28, 28,1))
train_data_x <- size_cnn(train_data_x)
validation_data_x <- size_cnn(validation_data_x)
dim(train_data_x)
# create the model
model_cnn <- keras_model_sequential() %>%
        layer_conv_2d(filters=32, kernel_size = c(3,3), input_shape = c(28,28, 1),
                      activation = "relu") %>% 
        layer_max_pooling_2d(pool_size = c(2,2)) %>% 
        layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu" ) %>% 
        layer_max_pooling_2d(pool_size = c(2,2)) %>% 
        layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu" ) %>% 
        layer_max_pooling_2d(pool_size = c(2,2)) %>%
        layer_flatten() %>% 
        layer_dropout(rate = .5) %>% 
        layer_dense(units=128, activation = "relu") %>%
        layer_dense(units=10, activation = "softmax")

# compile the model
model_cnn %>% compile(
        loss="binary_crossentropy",
        metrics = c("accuracy"),
        optimizer = "rmsprop"
)

history <- model_cnn %>% fit(
        train_data_x,
        train_labels_y,
        epochs = 20,
        batch_size = 128, 
        validation_data = list(validation_data_x, validation_labels_y)
)
max(history$metrics$val_acc)
test_data <- array_reshape(test_data, dim=c(dim(test_data), 1))
test_labels <- to_categorical(test_labels)
results <- model_cnn %>% evaluate(test_data, test_labels)
