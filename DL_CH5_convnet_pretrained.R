# improving convolutional networks
## ---- data_augmentation ----
# with the data generator the images are modified using several functions
# create an image generator
datagen <- image_data_generator(
        rescale = 1/255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = TRUE,
        fill_mode = "nearest"
)
## visualize some of the new images
fnames <- list.files(train_cats_dir, full.names = TRUE)
img_path <- fnames[[3]]                                             
img <- image_load(img_path, target_size = c(150, 150))               
img_array <- image_to_array(img)                                      
img_array <- array_reshape(img_array, c(1, 150, 150, 3))              
augmentation_generator <- flow_images_from_data(
        img_array,
        generator = datagen,
        batch_size = 1
)
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))            
for (i in 1:4) {                                                      
        batch <- generator_next(augmentation_generator)                    
        plot(as.raster(batch[1,,,]))                                        
}                                                                     
par(op) 
# because a new image generator was created a new model has to be ensambled
model <- keras_model_sequential() %>%
        # alternation between conv and pooling
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                      input_shape = c(150, 150, 3)) %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        # flatten out the network
        layer_flatten() %>%
        # here a dropout is added to regularize the network
        layer_dropout(rate = 0.5) %>%
        layer_dense(units = 512, activation = "relu") %>%
        layer_dense(units = 1, activation = "sigmoid")
# model compiation
model %>% compile(
        loss = "binary_crossentropy",
        optimizer = optimizer_rmsprop(lr = 1e-4),
        metrics = c("acc")
)
# training the model
# basically the "same" generator than before
datagen <- image_data_generator(
        rescale = 1/255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = TRUE
)
# generator for the test set
test_datagen <- image_data_generator(rescale = 1/255)
# flow images for the training data
train_generator <- flow_images_from_directory(train_dir,
                                              datagen,
                                              target_size = c(150, 150),
                                              batch_size = 32,
                                              class_mode = "binary"
)
# flow images for the validation data
validation_generator <- flow_images_from_directory(validation_dir,
                                                   test_datagen,
                                                   target_size = c(150, 150),
                                                   batch_size = 32,
                                                   class_mode = "binary"
)
# fit the network
history <- model %>% fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = 50
)
plot(history)
str(model)
str(history)
which.max(history$metrics$acc)
max(history$metrics$acc)
#       use previous networks: extract features
# save the pretrained net vgg16(basic), top layers are not included in this example
# the input_shape must be the same than the question data (not the pretrained data)
conv_base <- application_vgg16(
        weights = "imagenet",
        include_top = FALSE,
        input_shape = c(150, 150, 3)
)
# extranting features from the pretrained net
# defining variables for the paths of the training and validation data
base_dir <- "~/Downloads/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")
# data generator with scaling of the data
datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20
# function to extract the features from the vgg16
extract_features <- function(directory, sample_count) {
        # creates an empty array with the size of the last layer of vgg16
        features <- array(0, dim = c(sample_count, 4, 4, 512))
        # an empty array for the labels
        labels <- array(0, dim = c(sample_count))
        # flow image to get the images
        generator <- flow_images_from_directory(
                directory = directory,
                generator = datagen,
                target_size = c(150, 150),
                batch_size = batch_size,
                class_mode = "binary"
        )
        # the variable [i] is used to control the loop, because the generator can
        #       generate images infinetly 
        i <- 0
        while(TRUE) {
                # gets the images
                batch <- generator_next(generator)
                inputs_batch <- batch[[1]]
                labels_batch <- batch[[2]]
                # use the conv_base to predict using the focal images
                features_batch <- conv_base %>% predict(inputs_batch)
                index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
                # save the predictions in the empty array
                features[index_range,,,] <- features_batch
                # save the labels of conv_net in the labels array
                labels[index_range] <- labels_batch
                i <- i + 1
                # this is a control structure to stop the loop
                if (i * batch_size >= sample_count)
                        break                                                
        }
        list(
                features = features,
                labels = labels
        )
}
train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)
# the features are already extracted, now we have to flatten them
reshape_features <- function(features) {
        array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)
# once the features are flatten, then we have to use them to feed a densely
#       connected layer
# create the model
model <- keras_model_sequential() %>%
        layer_dense(units = 256, activation = "relu",
                    input_shape = 4 * 4 * 512) %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(units = 1, activation = "sigmoid")
# compile the model
model %>% compile(
        optimizer = optimizer_rmsprop(lr = 2e-5),
        loss = "binary_crossentropy",
        metrics = c("accuracy")
)
# fit the model
history <- model %>% fit(
        train$features, train$labels,
        epochs = 30,
        batch_size = 20,
        validation_data = list(validation$features, validation$labels)
)
# fine-tunning steps
# 1) Add your custom network on top of an already-trained base network.
# 2) Freeze the base network.
# 3) Train the part you added.
# 4) Unfreeze some layers in the base network.
# 5) Jointly train both these layers and the part you added.

# step 1, adding densely conected layers on top of your pretrained net
model <- keras_model_sequential() %>%
        conv_base %>%
        layer_flatten() %>%
        layer_dense(units = 256, activation = "relu") %>%
        layer_dense(units = 1, activation = "sigmoid")
# step 2, freeze the pretrained network
freeze_weights(conv_base)
cat("This is the number of trainable weights after freezing",
      "the conv base:", length(model$trainable_weights), "\n")
# step 3, train the model end to end with a frozen convolutional base
# image generator with augmentation
train_datagen = image_data_generator(
        rescale = 1/255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = TRUE,
        fill_mode = "nearest"
)
# re-scaled image generator for the test data
test_datagen <- image_data_generator(rescale = 1/255)
# getting the training images
train_generator <- flow_images_from_directory(train_dir,train_datagen,
                                              target_size = c(150, 150),
                                              batch_size = 20,
                                              class_mode = "binary"
                                              )
# getting the validation images
validation_generator <- flow_images_from_directory(validation_dir,test_datagen,
                                                   target_size = c(150, 150),
                                                   batch_size = 20,
                                                   class_mode = "binary"
                                                   )
# compile the model, very small value for the optimizer
model %>% compile(
        loss = "binary_crossentropy",
        optimizer = optimizer_rmsprop(lr = 2e-5),
        metrics = c("accuracy")
)
# NOT RUN 
# the model fiting must be run on GPU
# run until here to extract features with image augmentation
history <- model %>% fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 50
)
# step 4
unfreeze_weights(conv_base, from = "block3_conv1")
# step 5, fine tune the model
model %>% compile(
        loss = "binary_crossentropy",
        optimizer = optimizer_rmsprop(lr = 1e-5),
        metrics = c("accuracy")
)
history <- model %>% fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = 50
)


