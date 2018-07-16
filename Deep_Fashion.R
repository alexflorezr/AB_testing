## ---- describing the data ----
# neural network for DeepFashion data
root <- "~/Desktop/A/Me/DeepFashion"
dir <- dir(file.path(root, "img"))
# there are 5620 directories in total (categories)
length(dir(file.path(root, "img")))
# with 51 images in average
x <- sapply(dir(file.path(root, "img"), full.names = T), function(x) v <- c(v,length(dir(x))))
summary(x)
sd(x)
# there are 289219 images in total 
length(dir(file.path(root, "img"), recursive = T))
img <- image_load(dir(file.path(root, "img", dir[1]), full.names = T)[1])
img <- image_to_array(img)
plot(as.raster(img,max = 255))

## ---- upload labels ----
# the data is read after replacing multiple spaces for single spaces
attr_img_file <- file.path(root, "attr_img_short.txt")
attr_img <- read.delim(attr_img_file, skip = 2, h=F, stringsAsFactors = F, sep=" ",
                       row.names=1)
# have to transpose the dataframe to convert it into an array
img_array <- array_reshape(t(attr_img), dim=c(98, 1000))
read.delim(attr_img_file, skip = 2, h=F, stringsAsFactors = F, sep=" ",
           row.names=1)
## ---- upload images ----
# function to read the images from the directory 
# use the file list_eval_partition to create train, val and test directories
path_partition <- file.path(root, "list_eval_partition_single_space.txt")
partition  <- read.delim(path_partition, skip = 2, h=F, stringsAsFactors = F, sep=" ")
head(partition)
partition_train <- partition[which(partition[,2] == "train"),]
partition_val <- partition[which(partition[,2] == "val"),]
partition_test  <- partition[which(partition[,2] == "test"),]
# function to create the data partition 
create_partition <- function(name_dir, file, whereto){
        directories <- dir(file.path(whereto, "img"))
        setwd(whereto)
        for(d in 1:length(directories)){
                data_dir <- file.path(whereto, name_dir, directories[d])
                if(!dir.exists(data_dir)){
                        dir.create(data_dir)
                }
                which(as.vector(file[,1]))
                file.copy(as.vector(file[,1]), file.path(whereto, name_dir))
        }
}
create_partition("Train", partition_train, root)
create_partition("Validation", partition_val, root)
create_partition("Test", partition_test, root)



image_generator <- image_data_generator(rescale = 1/255)
generator <- flow_images_from_directory(directory=root,
                                        generator = image_generator,
                                        target_size = c(300,205),
                                        batch_size = 32, 
                                        class_mode = "binary"
                                        )
batch <- generator_next(generator)
par(mfrow=c(6,5), mar=c(0,0,0,0))
for(b in 1:30){
        plot(as.raster(batch[[1]][b,,,]))
}

