# create a VAE encoder
## ---- VAE network ---- 
library(keras)
img_shape <- c(28, 28, 1)
batch_size <- 16
latent_dim <- 2L                                               
input_img <- layer_input(shape = img_shape)
x <- input_img %>%
        layer_conv_2d(filters = 32, kernel_size = 3, padding = "same",
                      activation = "relu") %>%
        layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                      activation = "relu", strides = c(2, 2)) %>%
        layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                      activation = "relu") %>%
        layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                      activation = "relu")
shape_before_flattening <- k_int_shape(x)
x <- x %>%
        layer_flatten() %>%
        layer_dense(units = 32, activation = "relu")
z_mean <- x %>%                                              
layer_dense(units = latent_dim)                              
z_log_var <- x %>%                                           
layer_dense(units = latent_dim)     

