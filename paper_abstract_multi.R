# paper abstract classification for 6 journals
#       Ecography
#       Journal of biogeography
#       Global ecology and biogeography
#       Diversity and distributions
#       Ecology
#       Molecular ecology
## ---- upload abstracts ----
root <- "~/Desktop/A/Me/Papers_abstracts"
journals_dir <- dir(root)
# file connection
# abstract parser reads the text files and create the numeric labels 
abstract_parser <- function(dir_path){
        all_journal <- list()
        journals_dir <- dir(dir_path)
        attr(journals_dir, "names") <- dir(dir_path)
        label_count <- 1
        for (j in 1:length(journals_dir)){
                file_dir <- file.path(dir_path, journals_dir[j])
                file_name <- dir(file_dir, pattern="*abstract.txt")
                data <- strsplit(file_name, split="_")[[1]][1]
                assign(data,readLines(file.path(file_dir, file_name)))
                all_journal[[(j*2)-1]] <- get(data)
                all_journal[[j*2]] <- rep(j, length(get(data)))
        }
        all_journal
}
all_journal <- abstract_parser(root)
## ---- data segmentation ----
# 25% of the data is for testing
# perc_test: how much of the total data is allocated for testing
# perc_val: how much of the training data is allocated for validation
# name_segment: test, val, train
# seed: value to define set.seed()
segment_data <- function(abstract_list, perc_test, perc_val, seed){
        set.seed(seed)
        test <- list()
        train <- list()
        val <- list()
        all_data <- list()
        for(j in 1:(length(abstract_list)/2)){
                tmp_data <- abstract_list[[(j*2)-1]]
                tmp_label <- abstract_list[[(j*2)]]
                tmp_aux <- 1:length(tmp_data)
                tmp_sample_test <- sample(tmp_aux, round(length(tmp_data)*perc_test))
                test[[(j*2)-1]] <- as.vector(tmp_data[tmp_sample_test])
                test[[(j*2)]] <- tmp_label[tmp_sample_test]
                tmp_sample <- tmp_aux[-tmp_sample_test]
                tmp_sample_val <- sample(tmp_sample, length(tmp_sample)*perc_val)
                val[[(j*2)-1]] <- tmp_data[tmp_sample_val]
                val[[(j*2)]] <- tmp_label[tmp_sample_val]
                train[[(j*2)-1]] <- tmp_data[-c(tmp_sample_val,tmp_sample_test)]
                train[[(j*2)]] <- tmp_label[-c(tmp_sample_val,tmp_sample_test)]
        }
        all_data[[1]] <- train
        all_data[[2]] <- val
        all_data[[3]] <- test
        all_data
}
all_segmented <- segment_data(all_journal, .25, .25, 55)
str(all_segmented)
str(all_segmented[[1]])

