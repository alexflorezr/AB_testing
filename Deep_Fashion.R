# describing the data
# neural network for DeepFashion data
root <- "~/Desktop/A/Me/DeepFashion"
# there are 5620 directories in total (categories)
length(dir(file.path(root, "img")))
# with 51 images in average
x <- sapply(dir(file.path(root, "img"), full.names = T), function(x) v <- c(v,length(dir(x))))
summary(x)
sd(x)
# there are 289219 images in total 
length(dir(file.path(root, "img"), recursive = T))

## ---- train_val_test ----
# segmentate the data into 3 different directories

train_dir
val_dir
test_dir 

       