# visualizing intermediate activations
## ---- upload_image ----
img_path <- "~/Downloads/cats_and_dogs_small/test/cats/cat.1700.jpg"
img <- image_load(img_path, target_size = c(150, 150))                 
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255
# the first dimension is the object, then rows, columns and the colors
# in this case is a single image
dim(img_tensor)                                                        
# view the image
plot(as.raster(img_tensor[1,,,]))
## ---- create_model ----
# we have to create a model that takes a tensor as input and gives other tensor
#       as output after every layer
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)  
# the step were we save the intermediate activation layers
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
# we use the model to save the activation  based on our image
activations <- activation_model %>% predict(img_tensor)
# activations is a list of of 8 tensors with different dimensions
# the first dimension only reduces one row and one column
first_layer_activation <- activations[[1]]
dim(first_layer_activation)
## ---- plotting_channels ----
plot_channel <- function(channel) {
        rotate <- function(x) t(apply(x, 2, rev))
        image(rotate(channel), axes = FALSE, asp = 1,
              col = terrain.colors(12))
}
par(mfrow=c(6,7), mar=c(0,0,0,0))
for(i in 1:32){
        plot_channel(first_layer_activation[1,,,i])
}
## ---- view_all_channels ----
image_size <- 58
images_per_row <- 16

for (i in 1:8) {
        layer_activation <- activations[[i]]
        layer_name <- model$layers[[i]]$name
        n_features <- dim(layer_activation)[[4]]
        n_cols <- n_features %/% images_per_row
        png(paste0("cat_activations_", i, "_", layer_name, ".png"),
            width = image_size * images_per_row,
            height = image_size * n_cols)
        op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
        
        for (col in 0:(n_cols-1)) {
                for (row in 0:(images_per_row-1)) {
                        channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
                        plot_channel(channel_image)
                }
        }
        
        par(op)
        dev.off()
}
