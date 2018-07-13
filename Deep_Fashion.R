# describing the data
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
plot(as.raster(img,max = 255)) termi


# read the category text
# I changed the end-line character for \n and multiple spaces by \t
# d -e 's/\/n/\\n/g' list_attr_img.txt > list_attr_img_end.txt
# sed -e 's/jpg */jpg    /g' list_attr_img_end.txt > list_attr_img_end2.txt
attr_img_file <- file.path(root, "attr_img_short.txt")
attr_img <- read.delim(attr_img_file, skip = 2, h=F, stringsAsFactors = F, sep=" ",
                       row.names=1)
dim(attr_img)
attr_img2 <- as.integer(attr_img)
dim(attr_img)
# create an empty dataframe
to_df <- function(attr_img){
        df <- data.frame(matrix(nrow=dim(attr_img)[1], ncol=1000))
        row.names(df) <- attr_img[,1]
        for(i in 1:dim(attr_img)[1]){
                tmp_labels <- attr_img[,2]
                tmp_vec <- as.vector(strsplit(tmp_labels, split = " ")[[1]])
                tmp_null <- which(tmp_vec == "")
                if(length(tmp_null) > 0){
                        tmp_vec <- tmp_vec[-tmp_null]
                }
                df[i,] <- tmp_vec
                print(i)
        }
}
DF_df <- to_df(attr_img)

        ## ---- train_val_test ----
# segmentate the data into 3 different directories

train_dir
val_dir
test_dir 

       