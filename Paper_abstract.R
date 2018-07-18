# tokenize all the available abstracts from:
#       Ecography
#       Journal of biogeography
#       Global ecology and biogeography
#       Diversity and distributions
# the first NN will be a binary classification using divdis and jbio
root <- "~/Desktop/A/Me/Papers_abstracts"
journals_dir <- c("diversity_distributions", "Jbiogeography")
# file connection
file_divdis <- file.path(root, journals_dir[1], "divdis_abstract.txt")
file_jbio <- file.path(root, journals_dir[2], "jbio_abstract.txt")
# read the lines in the connection
text_divdis <- readLines(file_divdis)
n_divdis <- length(text_divdis)
text_jbio <- readLines(file_jbio)
n_jbio <- length(text_jbio)
labels_divdis <- rep(1, n_divdis)
labels_jbio <- rep(2, n_jbio )
# segmentation of the data into training, val and test
# 25% of the data for testing
set.seed(5)
sample_div_dis <- sample(1:n_divdis, round(n_divdis*.25))
sample_jbio <- sample(1:n_jbio , round(n_jbio *.25))
# test data and labels
data_test <- c(text_divdis[sample_div_dis], text_jbio[sample_jbio])
label_test <- c(labels_divdis[sample_div_dis], labels_jbio[sample_jbio])
# train data and labels
data_train <- c(text_divdis[-sample_div_dis], text_jbio[-sample_jbio])
label_train <- c(labels_divdis[-sample_div_dis], labels_jbio[-sample_jbio])
# tokenizer
maxlen <- 100
training_samples <- 1800
validation_samples <- 1163
max_words <- 10000

tokenizer <- text_tokenizer(num_words = max_words) %>%
        fit_text_tokenizer(data_train)
sequences <- texts_to_sequences(tokenizer, data_train)
# pad the sequences
data <- pad_sequences(sequences, maxlen = maxlen)
# convert the vector of labels into an array
labels <- as.array(label_train)
labels <- to_categorical(labels)
# divide the dat into val and train
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples+1):(training_samples+validation_samples)]

x_train <- data[training_indices,]
dim(x_train)
x_val <- data[validation_indices,]
dim(x_val)

y_train <- labels[training_indices,]
dim(y_train)
#y_train <- to_categorical(y_train)
y_val <- labels[validation_indices,]
#y_val <- to_categorical(y_val)
# define model
model <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_flatten() %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

history <- model %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
