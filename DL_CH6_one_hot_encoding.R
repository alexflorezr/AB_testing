# preprocesing text data
## ---- one_hot_encoding ---- 
# word-level one hot encoding
# creates an array where each matrix encode a word, if the word is in the sentence, 
#       then a 1 will appear in the position of the word in the sentence
samples <- c("The cat sat on the mat.", "The dog ate my homework.")       
token_index <- list() 
# loop over the sentences
for (sample in samples){
        # loop over the words in the sentence
        for (word in strsplit(sample, " ")[[1]]){
                # if the word in not in the token list
                if (!word %in% names(token_index))
                # assign an index (from 2) to each word
                token_index[[word]] <- length(token_index) + 2
        }
}
# defines the maximum length of a vector word
max_length <- 10
# creates an array of (samples, max length and number of words) size
results <- array(0, dim = c(length(samples),max_length,
                            max(as.integer(token_index))))

for (i in 1:length(samples)) {
        sample <- samples[[i]]
        words <- head(strsplit(sample, " ")[[1]], n = max_length)
        for (j in 1:length(words)) {
                index <- token_index[[words[[j]]]]
                results[[i, j, index]] <- 1
        }
}

# character-level one hot encoding
# the sample text
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
# all possible characters for letters
ascii_tokens <- c("", sapply(as.raw(c(32:126)), rawToChar))
# the indexes for the letters
token_index <- c(1:(length(ascii_tokens)))
names(token_index) <- ascii_tokens
# maximum length of the sentence
max_length <- 50
# creates and empty array 
results <- array(0, dim = c(length(samples), max_length, length(token_index)))
for (i in 1:length(samples)) {
        sample <- samples[[i]]
        characters <- strsplit(sample, "")[[1]]
        for (j in 1:length(characters)) {
                character <- characters[[j]]
                results[i, j, token_index[[character]]] <- 1
        }
}
# tonkezation on Keras 
library(keras)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
# the text_tokenizer vectorize the data, the fit_text_tokenizer fit the tokens to
#       to the actual data.
# the tokenizer has a complex structure with multiple attributes
# use the text_tokenizer and the fit_text_tokenizer together using %>%
tokenizer <- text_tokenizer(num_words = 1000) %>% fit_text_tokenizer(samples)
# list of the word level tokens for each sentence
sequences <- texts_to_sequences(tokenizer, samples)
# array of binary vectors
one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
word_index <- tokenizer$word_index
# to get back the sentence
names(word_index[match(sequences[[1]], unlist(word_index))])
# count the unique number of tokens in the text
cat("Found", length(word_index), "unique tokens.\n")


## ---- hashFunction ---- 
library(hashFunction)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
dimensionality <- 1000                                                  
max_length <- 10
# creates an empty array
results <- array(0, dim = c(length(samples), max_length, dimensionality))
for (i in 1:length(samples)) {
        sample <- samples[[i]]
        words <- head(strsplit(sample, " ")[[1]], n = max_length)
        for (j in 1:length(words)) {
                index <- abs(spooky.32(words[[i]])) %% dimensionality               
                results[[i, j, index]] <- 1
        }
}

