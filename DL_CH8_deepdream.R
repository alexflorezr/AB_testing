# deep dreaming 
## ---- load inception v3 ----
# load pretrained network (inception v3)
library(keras)
k_set_learning_phase(0)                
model <- application_inception_v3(weights = "imagenet", include_top = F)

layer_contributions <- list(mixed2 = 0.2,
                            mixed3 = 3,
                            mixed4 = 2,
                            mixed5 = 1.5                    
)
