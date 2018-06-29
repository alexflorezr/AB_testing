## ---- download the files ----
# create the directories and paths
original_dataset_dir <- "~/Downloads/kaggle_original_data"
base_dir <- "~/Downloads/cats_and_dogs_small"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)
# Divide the original data into training, validation and testing data
fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(train_cats_dir))
fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_cats_dir))
fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_cats_dir))
fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_dogs_dir))
fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_dogs_dir))
fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_dogs_dir))
# check the amount of data for each data set
cat("total training cat images:", length(list.files(train_cats_dir)), "\n")
cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")
cat("total validation cat images:", length(list.files(validation_cats_dir)), "\n")
cat("total validation dog images:", length(list.files(validation_dogs_dir)), "\n")
cat("total test cat images:", length(list.files(test_cats_dir)), "\n")
cat("total test dog images:", length(list.files(test_dogs_dir)), "\n")
## ---- create the model ----
model <- keras_model_sequential() %>%
        # first conv layer with relu activation, using a kernel of 3x3
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                      input_shape = c(150, 150, 3)) %>%
        # first max_pooling with a factor reduction of 2
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        # second conv layer, now with 64 filters
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
        # second max_pooling, also reduction of factor 2
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        # third conv layer, now with 128 filters
        layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
        # third max_pooling, also reduction of factor 2
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        # fourth conv layer, also 128 filters
        layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
        # fourth max_pooling, also reduction of factor 2
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        # now we have to flatten out the tensor
        # this will leave us with a vector of (width*height*depth)
        layer_flatten() %>%
        # now it's possible to apply a dense layer 
        layer_dense(units = 512, activation = "relu") %>%
        # and the last layer has a sigmoidal activation: binary classification
        layer_dense(units = 1, activation = "sigmoid")
## ---- compiling the model ----
model %>% compile(
        optimizer = optimizer_rmsprop(lr = 1e-4),
        # binary classification problem
        loss="binary_crossentropy",
        metrics = c("accuracy")
)
## ---- data preprocessing ----
# creates the data generator used in flow_images_from...
# default generator does not scale/normalize
train_datagen <- image_data_generator(rescale = 1/255)      
validation_datagen <- image_data_generator(rescale = 1/255)        

# creates the generator of training data to fit the model
train_generator <- flow_images_from_directory(
        train_dir,                                                       
        train_datagen,                                                   
        target_size = c(150, 150),                                       
        batch_size = 20,
        # need to specify the classes, in this case binary (2)
        class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
        validation_dir,
        validation_datagen,
        target_size = c(150, 150),
        batch_size = 20,
        class_mode = "binary"
)
## ---- fit the model ----
# because the data is the output of generator then a fit_generator must be used
history <- model %>% fit_generator(
        # the generator of the training data
        train_generator,
        # with 100 steps per epoch and batches of size 20, we get 2000 samples
        steps_per_epoch = 100,
        epochs = 30,
        # generator of the validation data
        validation_data = validation_generator,
        validation_steps = 50
)
model %>% save_model_hdf5("cats_and_dogs_small.h5")
plot(history)
