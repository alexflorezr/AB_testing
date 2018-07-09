# word embeddings
# Instantiating an embedding layer
embedding_layer <- layer_embedding(input_dim = 1000, output_dim = 64)
# Loading the IMDB data for use with an embedding layer
# to use only the 10000 more common words in the text
max_features <- 10000
# maximun number of words per description 
maxlen <- 20                                                  
imdb <- dataset_imdb(num_words = max_features)
# uses %<-% to create a list of training and test sets (data and labels)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb 
# pad the sequences to have the same length (in this case 20)
x_train <- pad_sequences(x_train, maxlen = maxlen)            
x_test <- pad_sequences(x_test, maxlen = maxlen)
#Using an embedding layer and classifier on the IMDB data
# Create the model 
model <- keras_model_sequential() %>%
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>%
        layer_flatten() %>%
        layer_dense(units = 1, activation = "sigmoid")
# Compile the model
model %>% compile(
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("acc")
)
summary(model)
history <- model %>% fit(
        x_train, 
        y_train,
        epochs = 10,
        batch_size = 32,
        validation_split = 0.2
)
##%######################################################%##
#                                                          #
####               All together for text                ####
#                                                          #
##%######################################################%##
# Processing the labels of the raw IMDB data
## ---- upload the database ----
imdb_dir <- "~/Downloads/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
# create a vector with all the raw texts
for (label_type in c("neg", "pos")) {
        label <- switch(label_type, neg = 0, pos = 1)
        dir_name <- file.path(train_dir, label_type)
        for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                                 full.names = TRUE)) {
                texts <- c(texts, readChar(fname, file.info(fname)$size))
                labels <- c(labels, label)
        }
}
## ---- tokenizing ----
# maximum number of words to keep in each review
maxlen <- 100
# restrics the dara to only 200 reviews
training_samples <- 200                                             
validation_samples <- 10000                                         
max_words <- 10000                                                  
tokenizer <- text_tokenizer(num_words = max_words) %>%
        fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")
# 1 to 25000 organized randomly 
indices <- sample(1:nrow(data))
# selects the first 10000 numbers in indices
training_indices <- indices[1:training_samples]
# selects the next 200 numbers in indices
validation_indices <- indices[(training_samples + 1):
                        (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

## ---- parsing_glove ----
glove_dir = "~/Downloads/glove.6B"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
        line <- lines[[i]]
        values <- strsplit(line, " ")[[1]]
        word <- values[[1]]
        embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index), "word vectors.\n")
## ---- glove_matrix ----
# this is the size of the embedding we chose > glove.6B.100d.txt
embedding_dim <- 100
# an empty array to save the data
embedding_matrix <- array(0, c(max_words, embedding_dim))
# loop over each word in imdb data
for (word in names(word_index)) {
        # get the index
        index <- word_index[[word]]
        # if the index is greater than 10000 we discard it
        if (index < max_words) {
                # get the embedding vector for the word
                embedding_vector <- embeddings_index[[word]]
                if (!is.null(embedding_vector))
                        # save the embedding in the array
                        embedding_matrix[index+1,] <- embedding_vector           
        }
}
## ---- model definition ----
model <- keras_model_sequential() %>%
        # embedding layer
        layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                        input_length = maxlen) %>%
        # flat the 3d tensor into a 2D
        layer_flatten() %>%
        # a densly connected layer with relu
        layer_dense(units = 32, activation = "relu") %>%
        # a classification layer using sigmoid
        layer_dense(units = 1, activation = "sigmoid")
summary(model)
## ---- glove_into_layer ----
# get the embedding layer of the model
# set the weights to be the ones of glove
# freeze the weights
get_layer(model, index = 0) %>%
        set_weights(list(embedding_matrix)) %>%
        freeze_weights()
## ---- compile_model ---- 
model %>% compile(
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("acc")
)
## ---- fit_model ---- 
history <- model %>% fit(
        x_train,
        y_train,
        epochs = 20,
        batch_size = 32,
        validation_data = list(x_val, y_val)
)
save_model_weights_hdf5(model, "pre_trained_glove_model.h5")
## ---- no_glove_embedding ----
model <- keras_model_sequential() %>%
        layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                        input_length = maxlen) %>%
        layer_flatten() %>%
        layer_dense(units = 32, activation = "relu") %>%
        layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("acc")
)

history <- model %>% fit(
        x_train, y_train,
        epochs = 20,
        batch_size = 32,
        validation_data = list(x_val, y_val)
)
## ---- testing ----
test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
        label <- switch(label_type, neg = 0, pos = 1)
        dir_name <- file.path(test_dir, label_type)
        for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                                 full.names = TRUE)) {
                texts <- c(texts, readChar(fname, file.info(fname)$size))
                labels <- c(labels, label)
        }
}
sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)
# evaluate the model
model %>%
        load_model_weights_hdf5("pre_trained_glove_model.h5") %>%
        evaluate(x_test, y_test)